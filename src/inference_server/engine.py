"""A FIFO queue and continuous inference worker for /generate."""

import asyncio
import time
from dataclasses import dataclass, field

from inference_server.metrics import EngineMetrics, elapsed_ms


class RequestTooLarge(ValueError):
    """A request that exceeds the context limit or the entire cache budget."""


@dataclass
class RequestQueue:
    """Shared limits and waiting count for the HTTP handlers and worker."""
    capacity: int = 20
    max_context_tokens: int = 4096
    max_cache_tokens: int = 8192
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Include requests collected by the worker but still waiting for cache space.
    waiting: int = 0
    metrics: EngineMetrics = field(default_factory=EngineMetrics)


def submit_request(requests, request, prompt_token_ids):
    """Validate and enqueue before HTTP headers are sent; return its stream state."""
    required_tokens = len(prompt_token_ids) + request.max_new_tokens
    if required_tokens > requests.max_context_tokens:
        raise RequestTooLarge(
            f"Prompt plus output limit needs {required_tokens} tokens; "
            f"context limit is {requests.max_context_tokens}"
        )
    if required_tokens > requests.max_cache_tokens:
        raise RequestTooLarge(
            f"Request needs {required_tokens} cache tokens; "
            f"cache budget is {requests.max_cache_tokens}, so it can never fit"
        )
    if requests.waiting >= requests.capacity:
        raise asyncio.QueueFull

    job = ActiveRequest(
        request, time.perf_counter(), prompt_token_ids=prompt_token_ids,
        reserved_cache_tokens=required_tokens,
    )
    requests.queue.put_nowait(job)
    requests.waiting += 1
    requests.metrics.record_queue(requests.waiting)
    return job


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


@dataclass
class ActiveRequest:
    """State kept from enqueue through generation, with an outgoing event queue."""
    request: object
    enqueued_at: float
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancelled: bool = False
    started_at: float | None = None
    batch_size: int = 0
    response: dict | None = None
    # Tokenize once before queueing; only admitted requests get a cache slot.
    prompt_token_ids: list[int] | None = None
    reserved_cache_tokens: int = 0
    prefill_tokens: int = 0  # Prompt tokens already processed by the model.
    token_ids: list[int] = field(default_factory=list)
    text_sent: int = 0  # Characters already emitted from the decoded token history.
    text_delta: str = ""  # Prepared in the GPU thread, published by the async worker.
    generation_started_at: float | None = None
    first_token_at: float | None = None
    cache_slot: int | None = None
    cache_length: int = 0

    def release_cache_slot(self):
        # The next occupant starts with an empty history in the same buffer.
        self.cache_slot = None
        self.cache_length = 0


def schedule_step(active, max_batched_tokens):
    """Give decode requests one token each, then spend the rest on FIFO prefill.

    The budget counts real input tokens, excluding padding and cached history.
    Server configuration ensures it can cover one decode per active request.
    """
    decoding = []
    for job in active:
        if job.prefill_tokens == len(job.prompt_token_ids):
            decoding.append(job)

    remaining_budget = max_batched_tokens - len(decoding)
    prefill_chunks = []
    for job in active:
        remaining_prompt = len(job.prompt_token_ids) - job.prefill_tokens
        if remaining_prompt > 0 and remaining_budget > 0:
            chunk_size = min(remaining_prompt, remaining_budget)
            prefill_chunks.append((job, chunk_size))
            remaining_budget -= chunk_size
    return decoding, prefill_chunks


async def run_worker(requests, step, batch_size=4, batch_wait_ms=10):
    """Run one scheduling iteration, release finished requests, then refill."""
    queue = requests.queue
    active = []
    waiting = []
    while True:
        # Cancelled handlers no longer need a slot. A queued cancellation is
        # checked below when we take it from the queue.
        remaining = []
        for job in active:
            if job.cancelled:
                job.release_cache_slot()
                queue.task_done()
            else:
                remaining.append(job)
        active = remaining
        requests.metrics.record_active(len(active), sum(job.reserved_cache_tokens for job in active))

        if not active and not waiting:
            waiting = await collect_batch(queue, batch_size, batch_wait_ms)

        # Gather arrivals and remove cancelled waiters at every boundary, even
        # when all active slots are full. All waiting jobs still count at admission.
        while not queue.empty():
            waiting.append(queue.get_nowait())
        remaining = []
        for job in waiting:
            if job.cancelled:
                requests.waiting -= 1
                queue.task_done()
            else:
                remaining.append(job)
        waiting = remaining

        # Reservations belong to active requests. Removing a finished, failed,
        # or cancelled request releases its reservation on the next iteration.
        reserved_tokens = sum(job.reserved_cache_tokens for job in active)
        free_slots = list(range(batch_size))
        for job in active:
            free_slots.remove(job.cache_slot)
        while waiting and free_slots:
            job = waiting[0]
            if reserved_tokens + job.reserved_cache_tokens > requests.max_cache_tokens:
                # Preserve FIFO: smaller requests behind this one also wait.
                # Existing requests keep decoding and eventually free space.
                break
            waiting.pop(0)
            requests.waiting -= 1
            reserved_tokens += job.reserved_cache_tokens
            job.cache_slot = free_slots.pop(0)
            active.append(job)
        requests.metrics.record_queue(requests.waiting)
        requests.metrics.record_active(len(active), reserved_tokens)
        if not active:
            continue

        started_at = time.perf_counter()
        for job in active:
            if job.started_at is None:
                job.started_at = started_at
                job.generation_started_at = started_at
                # This counts active requests at admission, not a model batch.
                job.batch_size = len(active)
        try:
            await step(active)
        except Exception as error:
            # A failed step releases its callers; the worker can serve new work.
            for job in active:
                job.release_cache_slot()
                if not job.cancelled:
                    job.events.put_nowait({"type": "error", "error": str(error)})
                queue.task_done()
            active = []
            requests.metrics.record_active(0, 0)
            continue

        remaining = []
        for job in active:
            if job.cancelled:
                job.release_cache_slot()
                queue.task_done()
                continue
            if job.text_delta:
                job.events.put_nowait({"type": "text", "text": job.text_delta})
                job.text_delta = ""
            if job.response is None:
                remaining.append(job)
                continue
            job.response["queue_wait_ms"] = elapsed_ms(job.enqueued_at, job.started_at)
            job.response["batch_size"] = job.batch_size
            job.response["type"] = "done"
            job.release_cache_slot()
            job.events.put_nowait(job.response)
            queue.task_done()
        active = remaining
        requests.metrics.record_active(len(active), sum(job.reserved_cache_tokens for job in active))
