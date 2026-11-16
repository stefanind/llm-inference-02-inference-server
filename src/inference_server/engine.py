"""A FIFO queue and one worker for complete /generate requests."""

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


async def run_worker(queue, generate):
    """Finish one request before taking the next. Queue.get() preserves FIFO."""
    while True:
        request, answer, enqueued_at = await queue.get()
        try:
            # If its waiting handler was cancelled, skip this queued request.
            if answer.cancelled():
                continue
            queue_wait_ms = elapsed_ms(enqueued_at, time.perf_counter())
            response = await generate(request)
            response["queue_wait_ms"] = queue_wait_ms
            # A handler can also be cancelled while inference is already running.
            if not answer.done():
                answer.set_result(response)
        except Exception as error:
            # A failed inference belongs to this request; the worker must survive
            # so it can serve the requests still waiting behind it.
            if not answer.done():
                answer.set_exception(error)
        finally:
            # join() uses this count to wait for accepted work during shutdown.
            queue.task_done()
