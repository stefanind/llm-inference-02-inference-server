"""Check full batches, timed partial batches, and full-queue rejection."""

import asyncio
import unittest
from unittest.mock import patch

import httpx

from inference_server.server import create_app


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_batches_run_in_fifo_order_without_waiting_for_the_timer(self):
        batches = []

        async def generate(prompts, delay_seconds):
            batches.append(prompts)
            await asyncio.sleep(0)
            return prompts

        # A long window makes it clear that full batches skip the timer.
        app = create_app(mode="simulated", batch_wait_ms=1000)
        with patch("inference_server.server.simulate_inference", side_effect=generate):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test",
                ) as client:
                    calls = [client.post("/generate", json={"prompt": str(index)}) for index in range(8)]
                    responses = await asyncio.wait_for(asyncio.gather(*calls), timeout=0.5)

        self.assertEqual(batches, [["0", "1", "2", "3"], ["4", "5", "6", "7"]])
        self.assertEqual([response.json()["text"] for response in responses], [str(index) for index in range(8)])

    async def test_partial_batch_runs_after_the_collection_window(self):
        app = create_app(mode="simulated", delay_seconds=0)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                response = await asyncio.wait_for(
                    client.post("/generate", json={"prompt": "alone"}), timeout=1,
                )

        self.assertEqual(response.json()["batch_size"], 1)
        self.assertGreaterEqual(response.json()["queue_wait_ms"], 8)

    async def test_full_queue_returns_429(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def generate(prompts, delay_seconds):
            started.set()
            await release.wait()
            return prompts

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
