"""Check scheduled overlap and collection of successful and rejected calls."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from inference_server.runner import run_workload
from inference_server.streaming import encode_event
from inference_server.workloads import ScheduledRequest


def mock_client(respond, fail_metrics=False):
    async def route(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/info":
            return httpx.Response(200, json={"model_id": "test-model", "revision": "test-commit"})
        if request.url.path in ("/metrics", "/metrics/reset"):
            if fail_metrics and request.url.path == "/metrics":
                return httpx.Response(503, text="Metrics unavailable")
            return httpx.Response(200, json={"active_requests": 0, "reserved_cache_tokens": 0})
        return await respond(request)
    return httpx.AsyncClient(transport=httpx.MockTransport(route), base_url="http://test")


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduled_arrival_does_not_wait_for_an_earlier_response(self):
        second_started = asyncio.Event()

        async def respond(request):
            prompt = json.loads(request.content)["prompt"]
            if prompt == "second":
                second_started.set()

            class Stream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield encode_event({"type": "text", "text": prompt}).encode()
                    if prompt == "first":
                        # Text arrives before completion; a serial client stalls.
                        await asyncio.wait_for(second_started.wait(), timeout=2)
                    yield encode_event({"type": "done", "finish_reason": "length"}).encode()

            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

        requests = [
            ScheduledRequest("1", 1, 0, "first"),
            ScheduledRequest("2", 2, 50, "second"),
        ]
        client = mock_client(respond)
        with patch("inference_server.runner.httpx.AsyncClient", return_value=client):
            report = await run_workload(requests)

        self.assertEqual([result["text"] for result in report["results"]], ["first", "second"])
        # The second deadline is 50 ms; allow a little room for timer resolution.
        self.assertGreaterEqual(report["results"][1]["actual_arrival_ms"], 40)
        self.assertLess(report["results"][0]["client_ttft_ms"], report["results"][0]["latency_ms"])
        self.assertEqual(report["client_metrics"], {"in_flight_requests": 0, "peak_in_flight_requests": 2})

    async def test_rejection_does_not_discard_successful_results(self):
        async def respond(request):
            if json.loads(request.content)["prompt"] == "busy":
                return httpx.Response(429, text="Queue full")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=(
                encode_event({"type": "text", "text": "reply"})
                + encode_event({"type": "done", "finish_reason": "stop", "input_tokens": 9, "queue_wait_ms": 12})
            ))

        requests = [
            ScheduledRequest("1", 1, 0, "hello"),
            ScheduledRequest("2", 2, 0, "busy"),
        ]
        client = mock_client(respond)
        with patch("inference_server.runner.httpx.AsyncClient", return_value=client):
            report = await run_workload(requests)

        self.assertEqual(report["summary"]["successful_requests"], 1)
        self.assertEqual(report["results"][1]["error_type"], "rejected")
        self.assertEqual(report["results"][0]["queue_wait_ms"], 12)
        self.assertEqual(report["results"][0]["input_tokens"], 9)
        self.assertEqual(report["run_info"]["server"]["revision"], "test-commit")

    async def test_vllm_request_and_response_use_the_shared_runner(self):
        async def respond(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(json.loads(request.content), {
                "model": "Qwen/Qwen3-4B",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "hello"},
                ],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            })
            events = [
                {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "reply"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4}},
            ]
            content = "".join(encode_event(event) for event in events) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=content)

        requests = [ScheduledRequest("1", 1, 0, "hello", max_new_tokens=16)]
        client = mock_client(respond)
        with patch("inference_server.runner.httpx.AsyncClient", return_value=client):
            report = await run_workload(
                requests, backend="vllm", model="Qwen/Qwen3-4B", workload_name="W1",
            )

        result = report["results"][0]
        self.assertEqual((result["text"], result["input_tokens"], result["output_tokens"]), ("reply", 12, 4))
        self.assertIsNone(result["ttft_ms"])
        self.assertIsNone(result["queue_wait_ms"])
        self.assertEqual(report["summary"]["total_output_tokens"], 4)
        self.assertEqual(report["run_info"]["requested_model"], "Qwen/Qwen3-4B")
        self.assertEqual(report["run_info"]["schedule"], [{
            "request_id": "1", "user_id": 1, "arrival_ms": 0,
            "prompt": "hello", "max_new_tokens": 16,
        }])

    async def test_vllm_overload_and_bad_responses_are_collected_separately(self):
        async def respond(request):
            prompt = json.loads(request.content)["messages"][-1]["content"]
            if prompt == "busy":
                return httpx.Response(503, json={"error": {
                    "message": "The engine is currently busy and cannot accept new requests.",
                }})
            if prompt == "faulted":
                return httpx.Response(503, text="The engine has faulted and cannot accept new requests.")
            # HTTP 200 alone cannot make this incomplete stream a success.
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=encode_event({"choices": []}))

        requests = [ScheduledRequest(prompt, None, 0, prompt) for prompt in ("busy", "faulted", "malformed")]
        client = mock_client(respond)
        with patch("inference_server.runner.httpx.AsyncClient", return_value=client):
            report = await run_workload(requests, backend="vllm", model="Qwen/Qwen3-4B")

        self.assertEqual([result["error_type"] for result in report["results"]], ["rejected", "error", "error"])
        self.assertEqual(report["results"][1]["error_body"], "The engine has faulted and cannot accept new requests.")
        self.assertEqual(report["summary"]["rejected_requests"], 1)

    async def test_truncated_streams_and_stream_errors_are_failures(self):
        async def respond(request):
            content = encode_event({"type": "text", "text": "partial"})
            if json.loads(request.content)["prompt"] == "error":
                content += encode_event({"type": "error", "error": "Model step failed"})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=content)

        requests = [ScheduledRequest(prompt, None, 0, prompt) for prompt in ("truncated", "error")]
        client = mock_client(respond, fail_metrics=True)
        with patch("inference_server.runner.httpx.AsyncClient", return_value=client):
            report = await run_workload(requests)
        self.assertEqual(report["summary"]["successful_requests"], 0)
        self.assertEqual([result["text"] for result in report["results"]], ["partial", "partial"])
        self.assertIn("Model step failed", report["results"][1]["error"])
        self.assertIn("503", report["engine_metrics_error"])
        self.assertEqual(report["client_metrics"]["in_flight_requests"], 0)
