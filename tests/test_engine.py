"""Check scheduling directly. Model loading is covered by gpu_smoke.py."""

import asyncio
import time
import unittest

from inference_server.engine import ActiveRequest, RequestQueue, collect_batch, run_worker, schedule_step, submit_request
from inference_server.workloads import ScheduledRequest


class EngineTests(unittest.IsolatedAsyncioTestCase):
    def test_token_budget_prioritizes_decode_then_resumes_fifo_prefill(self):
        long = ActiveRequest("long", 0, prompt_token_ids=list(range(8)))
        short = ActiveRequest("short", 0, prompt_token_ids=[1, 2])
        decoding = ActiveRequest("decoding", 0, prompt_token_ids=[1], prefill_tokens=1)
        active = [long, decoding, short]

        decode_jobs, chunks = schedule_step(active, max_batched_tokens=6)
        self.assertEqual(decode_jobs, [decoding])
        self.assertEqual(chunks, [(long, 5)])

        long.prefill_tokens = 5
        _, chunks = schedule_step(active, max_batched_tokens=6)
        self.assertEqual(chunks, [(long, 3), (short, 2)])

    async def test_full_batch_preserves_fifo_and_skips_the_timer(self):
        queue = asyncio.Queue()
        for index in range(4):
            queue.put_nowait(index)
        batch = await asyncio.wait_for(collect_batch(queue, 4, 1000), timeout=0.5)
        self.assertEqual(batch, [0, 1, 2, 3])

    async def test_partial_batch_runs_after_the_collection_window(self):
        queue = asyncio.Queue()
        queue.put_nowait("alone")
        started_at = time.perf_counter()
        batch = await asyncio.wait_for(collect_batch(queue, 4, 10), timeout=1)
        self.assertEqual(batch, ["alone"])
        self.assertGreaterEqual(time.perf_counter() - started_at, 0.008)

    async def test_finished_requests_return_and_their_slots_are_refilled(self):
        frames = []
        third_step = asyncio.Event()
        release = asyncio.Event()

        async def step(active):
            frames.append([job.request.prompt for job in active])
            if len(frames) == 3:
                third_step.set()
                await release.wait()  # Hold long requests before their last step.
            for job in active:
                job.token_ids.append(1)
                if len(job.token_ids) == job.request.max_new_tokens:
                    job.response = {"finish_reason": "length"}

        queue = RequestQueue()
        jobs = []
        for prompt, limit in [("short", 1), ("long1", 3), ("long2", 3), ("long3", 3), ("next", 1)]:
            request = ScheduledRequest(prompt, None, 0, prompt, limit)
            jobs.append(submit_request(queue, request, [1]))
        worker = asyncio.create_task(run_worker(queue, step))
        try:
            await asyncio.wait_for(third_step.wait(), timeout=2)
            self.assertEqual((await jobs[0].events.get())["type"], "done")
            self.assertEqual((await jobs[4].events.get())["type"], "done")
            self.assertTrue(all(job.events.empty() for job in jobs[1:4]))
        finally:
            release.set()
            await queue.queue.join()
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self.assertEqual(frames[:2], [
            ["short", "long1", "long2", "long3"],
            ["long1", "long2", "long3", "next"],
        ])

    async def test_full_queue_rejects_without_waiting_for_space(self):
        queue = RequestQueue(capacity=1)
        request = ScheduledRequest("1", None, 0, "first")
        submit_request(queue, request, [1])
        with self.assertRaises(asyncio.QueueFull):
            submit_request(queue, request, [1])

    async def test_requests_that_can_never_fit_are_rejected_before_queueing(self):
        queue = RequestQueue(max_context_tokens=8, max_cache_tokens=6)
        request = ScheduledRequest("1", None, 0, "prompt", max_new_tokens=3)
        with self.assertRaisesRegex(ValueError, "context limit"):
            submit_request(queue, request, [1] * 6)
        with self.assertRaisesRegex(ValueError, "can never fit"):
            submit_request(queue, request, [1] * 4)
        self.assertEqual(queue.waiting, 0)

    async def test_cache_waiting_preserves_fifo_and_counts_toward_queue_capacity(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        frames = []

        async def step(active):
            frames.append([job.request.prompt for job in active])
            if len(frames) == 1:
                entered.set()
                await release.wait()
            for job in active:
                job.response = {"finish_reason": "length"}

        queue = RequestQueue(capacity=3, max_cache_tokens=8)
        # The first two each reserve six tokens. A small request behind them
        # could fit alongside the first, but FIFO makes it wait for the second.
        for prompt, prompt_tokens, output_tokens in [("first", 3, 3), ("next", 3, 3), ("small", 1, 1)]:
            request = ScheduledRequest(prompt, None, 0, prompt, output_tokens)
            submit_request(queue, request, [1] * prompt_tokens)
        worker = asyncio.create_task(run_worker(queue, step, batch_wait_ms=0))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            extra = ScheduledRequest("extra", None, 0, "extra", 1)
            submit_request(queue, extra, [1])
            # The worker is holding waiting requests outside asyncio.Queue.
            # They still occupy waiting capacity, so this request is rejected.
            with self.assertRaises(asyncio.QueueFull):
                submit_request(queue, extra, [1])
            release.set()
            await asyncio.wait_for(queue.queue.join(), timeout=2)
        finally:
            release.set()
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self.assertEqual(frames, [["first"], ["next", "small"], ["extra"]])
        await asyncio.wait_for(queue.queue.join(), timeout=1)

    async def test_cancellation_and_failed_steps_release_cache_reservations(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def step(active):
            job = active[0]
            if job.request.prompt == "cancel":
                entered.set()
                await release.wait()
            elif job.request.prompt == "fail":
                raise RuntimeError("Failed model step")
            else:
                job.response = {"finish_reason": "length"}

        queue = RequestQueue(max_cache_tokens=6)
        jobs = []
        for prompt in ("cancel", "fail", "finish"):
            request = ScheduledRequest(prompt, None, 0, prompt, 3)
            jobs.append(submit_request(queue, request, [1] * 3))
        worker = asyncio.create_task(run_worker(queue, step, batch_wait_ms=0))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            jobs[0].cancelled = True
            release.set()
            failure = await asyncio.wait_for(jobs[1].events.get(), timeout=2)
            finished = await asyncio.wait_for(jobs[2].events.get(), timeout=2)
        finally:
            release.set()
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self.assertEqual(failure, {"type": "error", "error": "Failed model step"})
        self.assertEqual(finished["type"], "done")
        await asyncio.wait_for(queue.queue.join(), timeout=1)
