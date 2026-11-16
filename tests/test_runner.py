"""Check scheduled overlap and collection of successful and rejected calls."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from inference_server.runner import run_workload
from inference_server.workloads import ScheduledRequest


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduled_arrival_does_not_wait_for_an_earlier_response(self):
        second_started = asyncio.Event()

        async def respond(request):
            prompt = json.loads(request.content)["prompt"]
            if prompt == "first":
                # A serial runner would time out: the second call must start first.
                await asyncio.wait_for(second_started.wait(), timeout=2)
            else:
                second_started.set()
            return httpx.Response(200, json={"text": prompt})

        requests = [
            ScheduledRequest("1", 1, 0, "first"),
            ScheduledRequest("2", 2, 50, "second"),
        ]
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test")
        with patch("inference_server.runner.httpx.AsyncClient", return_value=client):
            report = await run_workload(requests)

        self.assertEqual([result["text"] for result in report["results"]], ["first", "second"])
        # The second deadline is 50 ms; allow a little room for timer resolution.
        self.assertGreaterEqual(report["results"][1]["actual_arrival_ms"], 40)

    async def test_rejection_does_not_discard_successful_results(self):
        async def respond(request):
            if json.loads(request.content)["prompt"] == "busy":
                return httpx.Response(429, text="Queue full")
            return httpx.Response(200, json={"text": "reply", "queue_wait_ms": 12})

        requests = [
            ScheduledRequest("1", 1, 0, "hello"),
            ScheduledRequest("2", 2, 0, "busy"),
        ]
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test")
        with patch("inference_server.runner.httpx.AsyncClient", return_value=client):
            report = await run_workload(requests)

        self.assertEqual(report["summary"]["successful_requests"], 1)
        self.assertEqual(report["results"][1]["error_type"], "rejected")
        self.assertEqual(report["results"][0]["queue_wait_ms"], 12)
