import asyncio
import math
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference_server.engine import run_worker, submit_request

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    use_kv_cache: bool = True

class GenerateResponse(BaseModel):
    text: str
    queue_wait_ms: float = Field(ge=0)
    batch_size: int = Field(ge=1)
    ttft_ms: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    output_tokens_per_second: float = Field(ge=0)
    used_kv_cache: bool = True


async def simulate_inference(prompts, delay_seconds=2.0):
    """Wait once for the whole batch, then return one recognizable reply per prompt."""
    await asyncio.sleep(delay_seconds)
    return [f"Simulated response: {prompt}" for prompt in prompts]


def create_app(mode="real", delay_seconds=2.0, queue_capacity=20, batch_size=4, batch_wait_ms=10):
    """Build the same HTTP routes with either simulated or real inference."""
    if mode not in ["simulated", "real"]:
        raise ValueError("mode must be simulated or real")
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay must be a finite, nonnegative number of seconds")
    if queue_capacity <= 0:
        raise ValueError("queue_capacity must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
        raise ValueError("batch_wait_ms must be finite and nonnegative")

    def generate_real(requests):
        # These imports happen only when real inference is requested.
        from inference_server.generation import generate_batch

        return generate_batch(requests, app.state.model, app.state.tokenizer)

    async def generate(requests):
        if mode == "simulated":
            texts = await simulate_inference([request.prompt for request in requests], delay_seconds)
            return [{"text": text, "mode": "simulated"} for text in texts]
        # GPU calls block. Move the whole call to a thread so the event loop can
        # still accept requests, reject a full queue, and answer health checks.
        # The worker awaits this call, so only one batch runs at a time.
        return await asyncio.to_thread(generate_real, requests)

    @asynccontextmanager
    async def lifespan(app):
        if mode == "real":
            # Import only in real mode: simulated startup needs no model libraries.
            from inference_server.model import load_model

            app.state.model, app.state.tokenizer = load_model()
        # Capacity counts queued requests; the worker can also hold one batch.
        app.state.queue = asyncio.Queue(maxsize=queue_capacity)
        worker = asyncio.create_task(run_worker(app.state.queue, generate, batch_size, batch_wait_ms))
        try:
            yield
        finally:
            # Finish accepted work before stopping the worker or releasing weights.
            await app.state.queue.join()
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            if mode == "real":
                del app.state.model
                del app.state.tokenizer

    app = FastAPI(title="tinyinference", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse if mode == "real" else None)
    async def generate_text(request: GenerateRequest):
        try:
            return await submit_request(app.state.queue, request)
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Inference queue is full")

    # Streaming keeps its original path. Scheduling token delivery will be a
    # separate iteration; do not mix /stream with queued GPU experiments yet.
    if mode == "simulated":
        @app.post("/stream")
        async def stream_text(request: GenerateRequest):
            async def chunks():
                texts = await simulate_inference([request.prompt], delay_seconds)
                # One text chunk after the delay; these are not generated tokens.
                yield texts[0]

            return StreamingResponse(chunks(), media_type="text/plain")

    else:
        @app.post("/stream")
        def stream_text(request: GenerateRequest):
            from inference_server.generation import generate_stream

            return StreamingResponse(
                generate_stream(
                    prompt=request.prompt,
                    model=app.state.model,
                    tokenizer=app.state.tokenizer,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                ),
                media_type="text/plain",
            )

    return app


# Preserve the original Uvicorn import path. cli.py selects its mode explicitly.
app = create_app(mode="real")
