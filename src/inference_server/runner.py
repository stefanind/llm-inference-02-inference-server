"""Run the same request schedule against this repo or a vLLM HTTP server."""

import asyncio
import json
import math
import time

import httpx

from inference_server.gpu_metrics import GpuSampler
from inference_server.metrics import ClientMetrics, elapsed_ms, summarize_http_results
from inference_server.streaming import read_sse
from inference_server.workloads import ScheduledRequest


def build_request(request: ScheduledRequest, backend, model):
    """Benchmark adapter: send the same prompt and output limit to either API."""
    if backend == "repo":
        return "/generate", {
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
            "temperature": 0.0,
        }

    return "/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request.prompt},
        ],
        "max_tokens": request.max_new_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_event(body, backend):
    """Return text, metadata updates, and whether generation has finished."""
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON stream event")
    if "error" in body:
        raise ValueError(f"Server stream error: {body['error']}")

    text = ""
    updates = {}
    finished = False
    if backend == "repo":
        if body.get("type") == "text":
            text = body.get("text")
        elif body.get("type") == "done":
            finished = True
            for field in ("input_tokens", "output_tokens", "ttft_ms", "generation_tpot_ms", "queue_wait_ms", "batch_size", "finish_reason"):
                updates[field] = body.get(field)
        else:
            raise ValueError("Expected a text, done, or error event")
    else:
        # Role-only chunks have no content; the final usage chunk has no choices.
        try:
            choices = body["choices"]
            if not isinstance(choices, list):
                raise ValueError("Expected a list of streaming choices")
            if choices:
                choice = choices[0]
                text = choice["delta"].get("content")
                if text is None:
                    text = ""
                reason = choice.get("finish_reason")
                if reason is not None:
                    updates["finish_reason"] = reason
                    finished = True
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Expected a vLLM chat completion chunk") from error
        usage = body.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("Expected usage to be a JSON object")
            updates["input_tokens"] = usage.get("prompt_tokens")
            updates["output_tokens"] = usage.get("completion_tokens")

    if not isinstance(text, str):
        raise ValueError("Expected response text to be a string")
    if finished and updates.get("finish_reason") not in ("stop", "length"):
        raise ValueError(f"Unexpected finish reason: {updates.get('finish_reason')}")
    # A malformed count should fail this request, not crash the run's summary.
    for field in ("input_tokens", "output_tokens"):
        count = updates.get(field)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError(f"Expected {field} to be a nonnegative integer")
    return text, updates, finished


async def send_request(client, request: ScheduledRequest, experiment_started_at, backend, model, client_metrics):
    """Wait for one request's deadline, send it, and return its observed result."""
    # Open-loop arrivals: schedule each request from the same experiment
    # clock, independently of when earlier requests finish.
    deadline = experiment_started_at + request.arrival_ms / 1000
    remaining_seconds = deadline - time.perf_counter()
    if remaining_seconds > 0:
        await asyncio.sleep(remaining_seconds)

    started_at = time.perf_counter()
    result = {
        "request_id": request.request_id,
        "user_id": request.user_id,
        "scheduled_arrival_ms": request.arrival_ms,
        # This is when the client starts its call, not when the server receives it.
        "actual_arrival_ms": elapsed_ms(experiment_started_at, started_at),
        "success": False,
        "status_code": None,
        "input_tokens": None,
        "output_tokens": None,
        "ttft_ms": None,  # Server generation timing, separate from client TTFT.
        "generation_tpot_ms": None,
        "queue_wait_ms": None,
        "batch_size": None,
        "finish_reason": None,
        "client_ttft_ms": None,
        "text_chunk_intervals_ms": [],
    }
    text_parts = []
    last_text_at = None

    client_metrics.request_started()
    try:
        path, payload = build_request(request, backend, model)
        async with client.stream("POST", path, json=payload) as response:
            result["status_code"] = response.status_code
            if not response.is_success:
                # Error responses are ordinary HTTP bodies, not SSE streams.
                await response.aread()
            response.raise_for_status()
            if response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
                raise ValueError("Expected a text/event-stream response")
            finished = False
            stream_ended = False
            async for data in read_sse(response):
                if data == "[DONE]":
                    if backend != "vllm" or not finished:
                        raise ValueError("Stream ended without a completion event")
                    stream_ended = True
                    break
                text, updates, generation_finished = parse_event(json.loads(data), backend)
                result.update(updates)
                finished = finished or generation_finished
                if text:
                    received_at = time.perf_counter()
                    if last_text_at is None:
                        # Client TTFT includes request delivery, server wait,
                        # generation, and the first nonempty chunk's transit.
                        result["client_ttft_ms"] = elapsed_ms(started_at, received_at)
                    else:
                        result["text_chunk_intervals_ms"].append(elapsed_ms(last_text_at, received_at))
                    last_text_at = received_at
                    text_parts.append(text)
                if backend == "repo" and finished:
                    stream_ended = True
                    break
            if not stream_ended:
                raise ValueError("Stream ended before completion")
        result["success"] = True
    except httpx.HTTPStatusError:
        result["error"] = f"HTTP {response.status_code}: {response.text}"
        result["error_body"] = response.text
        result["error_type"] = "rejected" if response.status_code == 429 else "error"
        if backend == "vllm" and response.status_code == 503:
            # vLLM also uses 503 for engine failures. Recognize its explicit
            # admission messages so those failures are not counted as overload.
            message = response.text.lower()
            if (
                "engine is currently busy and cannot accept new requests" in message
                or "prefill token backlog limit" in message
            ):
                result["error_type"] = "rejected"
    except httpx.TimeoutException as error:
        result["error"] = str(error) or "HTTP request timed out"
        result["error_type"] = "timeout"
    except (httpx.RequestError, ValueError) as error:
        result["error"] = str(error) or type(error).__name__
        result["error_type"] = "error"
    finally:
        client_metrics.request_finished()

    # Retain partial text for debugging failed streams, without counting it as success.
    result["text"] = "".join(text_parts)
    # Includes connection setup, all streamed events, parsing, and connection cleanup.
    result["latency_ms"] = elapsed_ms(started_at, time.perf_counter())
    return result


async def run_workload(
    requests: list[ScheduledRequest],
    server_url="http://127.0.0.1:8000",
    timeout_seconds=60,
    backend="repo",
    model=None,
    workload_name=None,
    workload_settings=None,
    server_info=None,
    gpu_uuid=None,
):
    """Run one schedule, drain all calls, then return results and their summary.

    Start the server separately. Startup and any warmup happen before this call.
    timeout_seconds applies to HTTPX's connect, read, write, and pool operations;
    it is not a deadline for the whole experiment. Requests are not retried.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    if backend not in ("repo", "vllm"):
        raise ValueError("backend must be 'repo' or 'vllm'")
    if backend == "vllm" and (model is None or not model.strip()):
        raise ValueError("--model is required with --backend vllm")
    if backend == "repo" and model is not None:
        raise ValueError("--model selects a vLLM served model; omit it for --backend repo")
    if server_info is not None and (backend != "vllm" or not isinstance(server_info, dict)):
        raise ValueError("server_info must be a JSON object supplied for the vLLM backend")

    # Save the exact inputs before starting the timer. A seed alone would not
    # reproduce a run after the workload definitions or prompts change.
    schedule = []
    for request in requests:
        schedule.append({
            "request_id": request.request_id,
            "user_id": request.user_id,
            "arrival_ms": request.arrival_ms,
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
        })
    run_info = {
        "backend": backend,
        "stream": True,
        "server_url": server_url,
        # This is a requested API name, not a verified checkpoint revision.
        # Repo requests do not select a model; its identity stays server-side.
        "requested_model": model,
        "timeout_seconds": timeout_seconds,
        "workload": workload_name,
        "workload_settings": workload_settings if workload_settings is not None else {},
        "schedule": schedule,
        "server": server_info,
        "server_info_source": "provided" if server_info is not None else None,
    }

    # Allow every request to overlap. A smaller pool would add client-side waiting
    # and change W5's 400-request burst. These schedules are deliberately small.
    connection_limit = max(1, len(requests))
    limits = httpx.Limits(
        max_connections=connection_limit,
        max_keepalive_connections=connection_limit,
    )
    engine_metrics = None
    engine_metrics_error = None
    # Count outstanding HTTP calls at the client. The engine separately
    # counts active requests with an assigned cache slot.
    client_metrics = ClientMetrics()
    async with httpx.AsyncClient(
        base_url=server_url, timeout=timeout_seconds, limits=limits,
    ) as client:
        if backend == "repo":
            # A dedicated server is required: reset excludes earlier warmup/work.
            response = await client.get("/info")
            response.raise_for_status()
            run_info["server"] = response.json()
            run_info["server_info_source"] = "server"
            response = await client.post("/metrics/reset")
            response.raise_for_status()
        else:
            # Repo metadata already warms its HTTP connection. Do the same for
            # vLLM, so its first measured request does not pay a different setup cost.
            response = await client.get("/health")
            response.raise_for_status()
        server_gpu = (run_info["server"] or {}).get("gpu_uuid")
        if server_gpu and gpu_uuid and server_gpu != gpu_uuid:
            raise ValueError("--gpu-uuid does not match the server's recorded GPU UUID")
        sampler = GpuSampler(server_gpu or gpu_uuid)
        async with sampler:
            experiment_started_at = time.perf_counter()
            tasks = []
            for request in requests:
                tasks.append(asyncio.create_task(
                    send_request(client, request, experiment_started_at, backend, model, client_metrics)
                ))

            # gather preserves schedule order while every call runs concurrently.
            results = await asyncio.gather(*tasks)
            elapsed_seconds = time.perf_counter() - experiment_started_at
        if backend == "repo":
            # Drain cancellations too. This wait and metadata collection are
            # outside the client experiment timer and its throughput calculation.
            try:
                response = await client.get("/metrics", params={"wait_for_idle": "true"})
                response.raise_for_status()
                engine_metrics = response.json()
            except (httpx.HTTPError, ValueError) as error:
                # Keep the collected client results if the stats request fails.
                engine_metrics_error = str(error)

    return {
        "run_info": run_info,
        "results": results,
        "summary": summarize_http_results(results, elapsed_seconds),
        "engine_metrics": engine_metrics,
        "engine_metrics_error": engine_metrics_error,
        "client_metrics": client_metrics.snapshot(),
        "gpu_metrics": sampler.snapshot(),
    }
