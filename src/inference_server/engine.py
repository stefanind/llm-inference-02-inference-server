"""A FIFO queue and one worker that runs small batches of /generate requests."""

import asyncio
import time

from inference_server.metrics import elapsed_ms


async def submit_request(queue, request):
    """Enqueue without waiting for space, then wait for this request's answer.

    A Future is a place for the worker to deliver an answer later. QueueFull is
    allowed to reach the HTTP handler, which turns it into a 429 response.
    """
    answer = asyncio.get_running_loop().create_future()
    queue.put_nowait((request, answer, time.perf_counter()))
    return await answer


async def run_worker(queue, generate, batch_size=4, batch_wait_ms=10):
    """Collect up to batch_size requests, then finish that batch before the next."""
    while True:
        batch = [await queue.get()]
        try:
            # Start one deadline when the first request is taken. New arrivals
            # fill spare slots, but never restart the batch's waiting window.
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
