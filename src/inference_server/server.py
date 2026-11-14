from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference_server.generation import (
    generate_cached,
    generate_uncached,
    generate_stream,
    summarize_result,
)
from inference_server.model import load_model


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    use_kv_cache: bool = True

class GenerateResponse(BaseModel):
    text: str
    ttft_ms: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    output_tokens_per_second: float = Field(ge=0)
    used_kv_cache: bool = True


@asynccontextmanager
async def lifespan(app):
    # Load once at startup. Every request uses these same model/tokenizer objects.
    app.state.model, app.state.tokenizer = load_model()
    try:
        yield
    finally:
        # Release the server's references when it shuts down.
        del app.state.model
        del app.state.tokenizer


app = FastAPI(title="tinyinference", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if request.use_kv_cache:
        generate_fn = generate_cached
    else:
        generate_fn = generate_uncached

    result = generate_fn(
        prompt=request.prompt,
        model=app.state.model,
        tokenizer=app.state.tokenizer,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k
    )

    summary = summarize_result(
        result,
        used_kv_cache=request.use_kv_cache
    )

    return GenerateResponse(**summary)

@app.post("/stream")
def stream_text(request: GenerateRequest):
    return StreamingResponse(
        generate_stream(
            prompt=request.prompt,
            model=app.state.model,
            tokenizer=app.state.tokenizer,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        ),
        media_type="text/plain",
    )
