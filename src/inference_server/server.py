import asyncio
import math
import sys
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch

from inference_server.cache import SlotCache
from inference_server.decode_compile import DecodeNorms
from inference_server.decode_graph import GRAPH_BATCH_SIZE, DecodeGraph
from inference_server.engine import RequestQueue, RequestTooLarge, run_worker, submit_request
from inference_server.generation import generate_step, tokenize_prompt
from inference_server.gpu_metrics import TorchMemoryMetrics
from inference_server.model import load_model
from inference_server.metrics import EngineMetrics
from inference_server.streaming import encode_event

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)


class RequestStream(StreamingResponse):
    def __init__(self, content, job):
        self.job = job
        super().__init__(
            content, media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Cover disconnects during headers as well as during body delivery.
            self.job.cancelled = True
            await self.body_iterator.aclose()


def stream_response(job, request):
    """SSE streaming: send a request's text, completion, or error events."""
    async def events():
        async def watch_disconnect():
            # Disconnect detection also runs while generation is between chunks.
            # The engine sees the cancellation flag at its next step boundary.
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    job.cancelled = True
                    job.events.put_nowait(None)  # Wake the event reader.
                    return

        watcher = asyncio.create_task(watch_disconnect())
        try:
            while True:
                event = await job.events.get()
                if event is None:
                    break
                yield encode_event(event)
                if event["type"] in ("done", "error"):
                    break
        finally:
            job.cancelled = True
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    return RequestStream(events(), job)


def create_app(
    queue_capacity=20, batch_size=4, batch_wait_ms=10, max_batched_tokens=512,
    max_context_tokens=4096, max_cache_tokens=8192,
    revision=None, profile=False, compile_decode=False, cuda_graph_decode=False,
):
    """Build the GPU inference server. Load one BF16 model during startup."""
    if queue_capacity <= 0:
        raise ValueError("queue_capacity must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cuda_graph_decode and batch_size != GRAPH_BATCH_SIZE:
        raise ValueError(f"--cuda-graph-decode requires --batch-size {GRAPH_BATCH_SIZE}")
    if cuda_graph_decode and compile_decode:
        raise ValueError("--cuda-graph-decode and --compile-decode are separate modes")
    if max_batched_tokens < batch_size:
        raise ValueError("max_batched_tokens must be at least batch_size so every decode can advance")
    if not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
        raise ValueError("batch_wait_ms must be finite and nonnegative")
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    if max_cache_tokens <= 0:
        raise ValueError("max_cache_tokens must be positive")

    async def step(active):
        # Async serving: the GPU step runs in a thread, so the HTTP event loop
        # can accept requests and deliver text while the model runs.
        await asyncio.to_thread(
            generate_step, active, app.state.model, app.state.tokenizer, app.state.cache,
            max_batched_tokens, app.state.requests.metrics, app.state.decode_graph,
        )

    @asynccontextmanager
    async def lifespan(app):
        app.state.model, app.state.tokenizer = load_model(revision=revision)
        # Respect the model's supported context even if the CLI limit is larger.
        context_limit = min(max_context_tokens, app.state.model.config.max_position_embeddings)
        app.state.requests = RequestQueue(
            capacity=queue_capacity, max_context_tokens=context_limit,
            max_cache_tokens=max_cache_tokens,
            metrics=EngineMetrics(profile_enabled=profile),
        )
        model = app.state.model
        if compile_decode:
            model.decode_norms = DecodeNorms(model)
        # Physical cache allocation: give each active slot its own GPU row.
        # The shared admission budget separately limits active reservations.
        cache_tokens_per_slot = min(context_limit, max_cache_tokens)
        app.state.cache = SlotCache(model, batch_size, cache_tokens_per_slot)
        app.state.decode_graph = None
        if cuda_graph_decode:
            app.state.decode_graph = DecodeGraph(model, app.state.cache, app.state.tokenizer.pad_token_id)
        app.state.gpu_memory = TorchMemoryMetrics(model.device)
        gpu_uuid = None
        if model.device.type == "cuda":
            gpu_uuid = str(torch.cuda.get_device_properties(model.device).uuid)
            if not gpu_uuid.startswith("GPU-") and not gpu_uuid.startswith("MIG-"):
                gpu_uuid = "GPU-" + gpu_uuid
        resolved_revision = getattr(model.config, "_commit_hash", None)
        app.state.server_info = {
            "model_id": model.name_or_path,
            "requested_revision": revision,
            "revision": resolved_revision,
            "tokenizer_revision": resolved_revision,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "gpu_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
            "gpu_uuid": gpu_uuid,
            "cuda_runtime": torch.version.cuda,
            "python_version": sys.version.split()[0],
            "package_versions": {
                name: version(name) for name in ("torch", "transformers", "accelerate", "fastapi", "uvicorn", "httpx")
            },
            "settings": {
                "queue_capacity": queue_capacity,
                "batch_size": batch_size,
                "batch_wait_ms": batch_wait_ms,
                "max_batched_tokens": max_batched_tokens,
                "max_context_tokens": context_limit,
                "max_cache_tokens": max_cache_tokens,
                "cache_layout": "fixed_slots",
                "cache_tokens_per_slot": cache_tokens_per_slot,
                "profile_enabled": profile,
                "compile_decode": compile_decode,
                "cuda_graph_decode": cuda_graph_decode,
                "cuda_graph_context_tokens": app.state.decode_graph.batch.context_tokens if cuda_graph_decode else None,
            },
        }
        print(f"Context limit: {context_limit} tokens; cache budget: {max_cache_tokens} tokens", flush=True)
        worker = asyncio.create_task(run_worker(app.state.requests, step, batch_size, batch_wait_ms))
        try:
            yield
        finally:
            # Finish accepted work before stopping the worker or releasing weights.
            await app.state.requests.queue.join()
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            del app.state.decode_graph
            del app.state.cache
            del app.state.model
            del app.state.tokenizer

    app = FastAPI(title="tinyinference", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/info")
    async def info():
        return app.state.server_info

    @app.get("/metrics")
    async def metrics(wait_for_idle: bool = False):
        if wait_for_idle:
            await app.state.requests.queue.join()
        result = app.state.requests.metrics.snapshot()
        result.update(app.state.gpu_memory.snapshot(synchronize=wait_for_idle))
        return result

    @app.post("/metrics/reset")
    async def reset_metrics():
        requests = app.state.requests
        if requests.waiting or requests.metrics.active_requests:
            raise HTTPException(status_code=409, detail="Wait for the engine to become idle before resetting metrics")
        # Reset after warmup so the next workload starts with fresh counters
        # and a VRAM baseline that includes the model and persistent cache.
        requests.metrics = EngineMetrics(profile_enabled=profile)
        app.state.gpu_memory.reset()
        result = requests.metrics.snapshot()
        result.update(app.state.gpu_memory.snapshot())
        return result

    @app.post("/generate", response_class=StreamingResponse)
    async def generate_text(request: GenerateRequest, connection: Request):
        # Request intake: format and tokenize before admission so limits use
        # the actual model input length and rejections can use an HTTP status.
        prompt_token_ids = await asyncio.to_thread(tokenize_prompt, request.prompt, app.state.tokenizer)
        try:
            job = submit_request(app.state.requests, request, prompt_token_ids)
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Inference queue is full")
        except RequestTooLarge as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return stream_response(job, connection)

    return app


# Preserve the Uvicorn import path as an alternative to cli.py.
app = create_app()
