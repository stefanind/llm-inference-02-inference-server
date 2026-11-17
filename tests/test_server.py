"""Check batch collection, continuous slot reuse, and full-queue rejection."""

import asyncio
import unittest
from unittest.mock import patch

import httpx

from inference_server.server import create_app, simulate_step


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

    async def test_continuous_worker_returns_short_requests_and_refills_their_slots(self):
        steps = []
        third_step = asyncio.Event()
        release = asyncio.Event()

        async def step(active, delay_seconds):
            steps.append([job.request.prompt for job in active])
            if len(steps) == 3:
                third_step.set()
                await release.wait()  # Hold the long requests before their final step.
            await simulate_step(active, delay_seconds)

        app = create_app(mode="simulated", scheduler="continuous", step_delay_seconds=0)
        with patch("inference_server.server.simulate_step", side_effect=step):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test",
                ) as client:
                    calls = []
                    for prompt, limit in [("short", 1), ("long1", 3), ("long2", 3), ("long3", 3), ("next", 1)]:
                        calls.append(asyncio.create_task(client.post("/generate", json={
                            "prompt": prompt, "max_new_tokens": limit,
                        })))
                    try:
                        await asyncio.wait_for(third_step.wait(), timeout=2)
                        await asyncio.wait_for(calls[0], timeout=1)
                        await asyncio.wait_for(calls[4], timeout=1)
                        self.assertFalse(any(call.done() for call in calls[1:4]))
                    finally:
                        release.set()
                    responses = await asyncio.gather(*calls)

        self.assertEqual(steps[:2], [
            ["short", "long1", "long2", "long3"],
            ["long1", "long2", "long3", "next"],
        ])
        self.assertEqual([response.json()["simulated_steps"] for response in responses], [1, 3, 3, 3, 1])

    async def test_full_queue_returns_429(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def step(active, delay_seconds):
            started.set()
            await release.wait()
            await simulate_step(active, 0)

        app = create_app(mode="simulated", scheduler="continuous", queue_capacity=1)
        with patch("inference_server.server.simulate_step", side_effect=step):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test",
                ) as client:
                    first = asyncio.create_task(client.post("/generate", json={"prompt": "first", "max_new_tokens": 1}))
                    try:
                        await asyncio.wait_for(started.wait(), timeout=2)
                        second = asyncio.create_task(client.post("/generate", json={"prompt": "second", "max_new_tokens": 1}))
                        await asyncio.sleep(0)  # Let the second request fill the one waiting slot.
                        rejected = await client.post("/generate", json={"prompt": "third"})
                    finally:
                        release.set()  # Always unblock the worker so shutdown can finish.
                    accepted = await asyncio.gather(first, second)

        self.assertEqual(rejected.status_code, 429)
        self.assertEqual([response.status_code for response in accepted], [200, 200])
