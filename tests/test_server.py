"""Check the two main queue behaviors through the HTTP server."""

import asyncio
import unittest
from unittest.mock import patch

import httpx

from inference_server.server import create_app


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_inference_runs_one_request_at_a_time_in_fifo_order(self):
        events = []

        async def generate(prompt, delay_seconds):
            events.append("start " + prompt)
            await asyncio.sleep(0.01)
            events.append("finish " + prompt)
            return prompt

        app = create_app(mode="simulated")
        with patch("inference_server.server.simulate_inference", side_effect=generate):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test",
                ) as client:
                    responses = await asyncio.gather(
                        client.post("/generate", json={"prompt": "first"}),
                        client.post("/generate", json={"prompt": "second"}),
                    )

        self.assertEqual(events, ["start first", "finish first", "start second", "finish second"])
        self.assertEqual([response.json()["text"] for response in responses], ["first", "second"])
        self.assertGreater(responses[1].json()["queue_wait_ms"], 0)

    async def test_full_queue_returns_429(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def generate(prompt, delay_seconds):
            started.set()
            await release.wait()
            return prompt

        app = create_app(mode="simulated", queue_capacity=1)
        with patch("inference_server.server.simulate_inference", side_effect=generate):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test",
                ) as client:
                    first = asyncio.create_task(client.post("/generate", json={"prompt": "first"}))
                    try:
                        await asyncio.wait_for(started.wait(), timeout=2)
                        second = asyncio.create_task(client.post("/generate", json={"prompt": "second"}))
                        await asyncio.sleep(0)  # Let the second request fill the one waiting slot.
                        rejected = await client.post("/generate", json={"prompt": "third"})
                    finally:
                        release.set()  # Always unblock the worker so shutdown can finish.
                    accepted = await asyncio.gather(first, second)

        self.assertEqual(rejected.status_code, 429)
        self.assertEqual([response.status_code for response in accepted], [200, 200])
