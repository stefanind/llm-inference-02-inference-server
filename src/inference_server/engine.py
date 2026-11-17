"""A FIFO queue with fixed-batch and continuous workers for /generate."""

import asyncio
import time
from dataclasses import dataclass, field

from inference_server.metrics import elapsed_ms


async def submit_request(queue, request):
    """Enqueue without waiting for space, then wait for this request's answer.

    A Future is a place for the worker to deliver an answer later. QueueFull is
    allowed to reach the HTTP handler, which turns it into a 429 response.
    """
    answer = asyncio.get_running_loop().create_future()
    queue.put_nowait((request, answer, time.perf_counter()))
    return await answer


async def collect_batch(queue, batch_size, batch_wait_ms):
    """Wait for the first request, then collect until full or the deadline."""
    batch = [await queue.get()]
    deadline = time.perf_counter() + batch_wait_ms / 1000
    while len(batch) < batch_size:
        try:
            batch.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
    return batch


async def run_worker(queue, generate, batch_size=4, batch_wait_ms=10):
    """Finish one fixed batch before starting the next."""
    while True:
        batch = await collect_batch(queue, batch_size, batch_wait_ms)
        try:
            jobs = [job for job in batch if not job[1].cancelled()]
            if not jobs:
                continue
            started_at = time.perf_counter()
            responses = await generate([request for request, _, _ in jobs])
            if len(responses) != len(jobs):
                raise RuntimeError("Inference must return one response per request")

            for (_, answer, enqueued_at), response in zip(jobs, responses):
                # Queue time includes waiting for this batch to form.
                response["queue_wait_ms"] = elapsed_ms(enqueued_at, started_at)
                response["batch_size"] = len(jobs)
                if not answer.done():
                    answer.set_result(response)
        except Exception as error:
            # A failed batch must not stop the worker or leave its callers waiting.
            for _, answer, _ in batch:
                if not answer.done():
                    answer.set_exception(error)
        finally:
            # join() uses this count to wait for accepted work during shutdown.
            for _ in batch:
                queue.task_done()


@dataclass
class ActiveRequest:
    """State kept between decode steps. The Future belongs to its HTTP handler."""
    request: object
    answer: asyncio.Future
    enqueued_at: float
    started_at: float | None = None
    batch_size: int = 0
    steps_completed: int = 0
    response: dict | None = None
    # Real inference initializes these lazily. Simulation does not need them.
    prompt_token_ids: list[int] | None = None
    token_ids: list[int] = field(default_factory=list)
    generation_started_at: float | None = None
    first_token_at: float | None = None


async def run_continuous_worker(queue, step, batch_size=4, batch_wait_ms=10):
    """Advance active requests one step, release finished ones, then refill."""
    active = []
    while True:
        # Cancelled handlers no longer need a slot. A queued cancellation is
        # checked below when we take it from the queue.
        remaining = []
        for job in active:
            if job.answer.cancelled():
                queue.task_done()
            else:
                remaining.append(job)
        active = remaining

        if not active:
            jobs = await collect_batch(queue, batch_size, batch_wait_ms)
        else:
            # Existing requests keep moving. Never pause their next step just
            # to wait for another arrival to fill an empty slot.
            jobs = []
            while len(active) + len(jobs) < batch_size:
                try:
                    jobs.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

        for request, answer, enqueued_at in jobs:
            if answer.cancelled():
                queue.task_done()
            else:
                active.append(ActiveRequest(request, answer, enqueued_at))
        if not active:
            continue

        started_at = time.perf_counter()
        for job in active:
            if job.started_at is None:
                job.started_at = started_at
                # Membership can change later. This is the first step's size.
                job.batch_size = len(active)
        try:
            await step(active)
        except Exception as error:
            # A failed step releases its callers; the worker can serve new work.
            for job in active:
                if not job.answer.done():
                    job.answer.set_exception(error)
                queue.task_done()
            active = []
            continue

        remaining = []
        for job in active:
            if job.response is None:
                remaining.append(job)
                continue
            job.response["queue_wait_ms"] = elapsed_ms(job.enqueued_at, job.started_at)
            job.response["batch_size"] = job.batch_size
            if not job.answer.done():
                job.answer.set_result(job.response)
            queue.task_done()
        active = remaining
