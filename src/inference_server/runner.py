"""Submit a workload to /generate and collect client timings and responses."""

import asyncio
import math
import time

import httpx

from inference_server.metrics import elapsed_ms, summarize_http_results
from inference_server.workloads import ScheduledRequest


async def send_request(client, request: ScheduledRequest, experiment_started_at):
    """Wait for one request's deadline, send it, and return its observed result."""
    # Every task uses the same experiment start, so earlier requests cannot
    # shift this request's arrival time by taking longer to finish.
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
    }

    try:
        response = await client.post("/generate", json={
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
        })
        result["status_code"] = response.status_code
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            raise ValueError("Expected a JSON object with a text string")

        result["success"] = True
        result["text"] = body["text"]
        # Simulation omits these. Missing measurements remain None, never zero.
        result["output_tokens"] = body.get("output_tokens")
        result["ttft_ms"] = body.get("ttft_ms")  # Server generation TTFT only.
        result["queue_wait_ms"] = body.get("queue_wait_ms")
        result["batch_size"] = body.get("batch_size")
        result["scheduler"] = body.get("scheduler")
        result["simulated_steps"] = body.get("simulated_steps")
    except httpx.HTTPStatusError:
        result["error"] = f"HTTP {response.status_code}: {response.text}"
        # Only 429 explicitly means rejection here; other HTTP failures are errors.
        result["error_type"] = "rejected" if response.status_code == 429 else "error"
    except httpx.TimeoutException as error:
        result["error"] = str(error) or "HTTP request timed out"
        result["error_type"] = "timeout"
    except (httpx.RequestError, ValueError) as error:
        result["error"] = str(error) or type(error).__name__
        result["error_type"] = "error"

    # Includes connection setup, HTTP waiting, receiving the body, and JSON decoding.
    result["latency_ms"] = elapsed_ms(started_at, time.perf_counter())
    return result


async def run_workload(
    requests: list[ScheduledRequest],
    server_url="http://127.0.0.1:8000",
    timeout_seconds=60,
):
    """Run one schedule, drain all calls, then return results and their summary.

    Start the server separately. Startup and any warmup happen before this call.
    timeout_seconds applies to HTTPX's connect, read, write, and pool operations;
    it is not a deadline for the whole experiment. Requests are not retried.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")

    # Allow every request to overlap. A smaller pool would add client-side waiting
    # and change W5's 400-request burst. These schedules are deliberately small.
    connection_limit = max(1, len(requests))
    limits = httpx.Limits(
        max_connections=connection_limit,
        max_keepalive_connections=connection_limit,
    )
    async with httpx.AsyncClient(
        base_url=server_url, timeout=timeout_seconds, limits=limits,
    ) as client:
        experiment_started_at = time.perf_counter()
        tasks = []
        for request in requests:
            tasks.append(asyncio.create_task(
                send_request(client, request, experiment_started_at)
            ))

        # Awaiting each request in the loop above would make the workload serial.
        # gather waits for all tasks and returns results in the schedule's order.
        results = await asyncio.gather(*tasks)
        elapsed_seconds = time.perf_counter() - experiment_started_at

    return {
        "results": results,
        "summary": summarize_http_results(results, elapsed_seconds),
    }
