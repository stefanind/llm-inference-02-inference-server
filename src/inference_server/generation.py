"""Token sampling and continuous generation with fixed KV cache slots."""

import time
from typing import Optional

import torch

from inference_server.engine import schedule_step


def profile_time(device):
    # Profiling with --profile: wait for the GPU at stage boundaries so each
    # stage is charged for its own CUDA work.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def finish_timing(metrics, phase, stage, started_at, device):
    finished_at = profile_time(device)
    metrics.record_stage(phase, stage, (finished_at - started_at) * 1000)
    return finished_at


def tokenize_prompt(prompt, tokenizer):
    """Chat template and tokenization: turn the messages into model input IDs."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False,
        return_dict=False,
    )


def select_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Token selection: greedy at temperature zero, otherwise sample."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    if top_k is not None:
        # Top-k filtering drops candidates whose score is below the cutoff.
        top_values = torch.topk(scaled_logits, top_k, dim=-1).values
        cutoff = top_values[..., -1:]
        scaled_logits = scaled_logits.masked_fill(
            scaled_logits < cutoff,
            float('-inf')
        )

    probabilities = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def forward_tokens(jobs, sequences, model, tokenizer, cache, metrics, phase, decode_graph=None):
    """Process new tokens and append their keys and values to the slot cache."""
    profile = metrics is not None and metrics.profile_enabled
    if profile:
        started_at = profile_time(model.device)

    if decode_graph is None:
        input_ids, position_ids, attention_mask, output_rows = cache.prepare(
            jobs, sequences, tokenizer.pad_token_id,
        )
    else:
        output_rows = decode_graph.prepare(jobs, sequences)
    if profile:
        started_at = finish_timing(metrics, phase, "batch_setup", started_at, model.device)

    if metrics is not None:
        metrics.record_forward(phase, len(jobs), graph_replay=decode_graph is not None)
    # Model forward: score the next token. Inside each attention layer, Qwen3
    # calls SlotCache.update() to write new keys and values into the slot.
    if decode_graph is not None:
        logits = decode_graph.replay()
    else:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": cache,
            "use_cache": True,
        }
        decode_norms = getattr(model, "decode_norms", None)
        if phase == "decode" and decode_norms is not None:
            outputs = decode_norms.run_decode(model, **inputs)
        else:
            outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
    if profile:
        finish_timing(metrics, phase, "model_forward", started_at, model.device)

    # Each layer has already written its new tokens. Advance the logical
    # lengths only after the whole forward succeeds.
    for job, sequence in zip(jobs, sequences):
        job.cache_length += len(sequence)
    # The last input position predicts the next token. Return rows in request
    # order, skipping any empty slots that were padded into the batch.
    if output_rows == list(range(logits.shape[0])):
        return logits
    return logits[output_rows]


def prefill(jobs, chunk_lengths, model, tokenizer, cache, metrics):
    """Chunked prefill: feed only the prompt slice chosen by schedule_step."""
    sequences = []
    for job, length in zip(jobs, chunk_lengths):
        start = job.prefill_tokens
        sequences.append(job.prompt_token_ids[start:start + length])
    logits = forward_tokens(jobs, sequences, model, tokenizer, cache, metrics, "prefill")
    for job, length in zip(jobs, chunk_lengths):
        job.prefill_tokens += length
    return logits


def decode(jobs, model, tokenizer, cache, metrics, decode_graph=None):
    """Autoregressive decode: feed the last generated token and reuse its history."""
    sequences = [[job.token_ids[-1]] for job in jobs]
    if decode_graph is not None and not decode_graph.can_replay(jobs):
        decode_graph = None
    return forward_tokens(jobs, sequences, model, tokenizer, cache, metrics, "decode", decode_graph)


def finish_step(jobs, logits, tokenizer, stop_token_ids):
    """Select one token per request and finish requests that hit EOS or a limit."""
    for index, job in enumerate(jobs):
        # Sample one row at a time. .item() brings its token ID back to Python.
        token = select_next_token(
            logits[index:index + 1], job.request.temperature, job.request.top_k,
        ).item()
        job.token_ids.append(token)
        ready_at = time.perf_counter()
        if job.first_token_at is None:
            # Engine TTFT ends when the first token is ready. The HTTP client
            # measures time to first visible text separately.
            job.first_token_at = ready_at
        reached_eos = token in stop_token_ids
        finished = reached_eos or len(job.token_ids) >= job.request.max_new_tokens
        # Decode the history together: one token may contain only part of a UTF-8
        # character. Hold an incomplete trailing character until more bytes arrive.
        text = tokenizer.decode(
            job.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        if finished or not text.endswith("\ufffd"):
            job.text_delta = text[job.text_sent:]
            job.text_sent = len(text)
        if finished:
            reason = "stop" if reached_eos else "length"
            job.response = build_response(job, ready_at, reason)


# Inference mode keeps PyTorch from recording gradients in the serving loop.
@torch.inference_mode()
def generate_step(active, model, tokenizer, cache, max_batched_tokens=512, metrics=None, decode_graph=None):
    """Decode first, then process prompt chunks within the remaining budget."""
    profile = metrics is not None and metrics.profile_enabled
    # Qwen3 declares two end tokens. The tokenizer only names the primary one.
    stop_token_ids = {tokenizer.eos_token_id}
    model_eos = model.generation_config.eos_token_id
    if isinstance(model_eos, list):
        stop_token_ids.update(model_eos)
    elif model_eos is not None:
        stop_token_ids.add(model_eos)

    decoding, chunks = schedule_step(active, max_batched_tokens)
    # Keep these as separate forward passes. A long prefill yields to decode
    # again on the next scheduler iteration instead of running to completion.
    if decoding:
        logits = decode(decoding, model, tokenizer, cache, metrics, decode_graph)
        if profile:
            started_at = profile_time(model.device)
        finish_step(decoding, logits, tokenizer, stop_token_ids)
        if profile:
            finish_timing(metrics, "decode", "token_handling", started_at, model.device)
    if chunks:
        jobs = [job for job, length in chunks]
        chunk_lengths = [length for job, length in chunks]
        logits = prefill(jobs, chunk_lengths, model, tokenizer, cache, metrics)
        if profile:
            started_at = profile_time(model.device)
        for index, job in enumerate(jobs):
            # Intermediate chunks build the cache without generating an answer.
            if job.prefill_tokens == len(job.prompt_token_ids):
                finish_step([job], logits[index:index + 1], tokenizer, stop_token_ids)
        if profile:
            finish_timing(metrics, "prefill", "token_handling", started_at, model.device)


def build_response(job, finished_at, finish_reason):
    """Final stream metadata. Text was already emitted in incremental chunks."""
    output_tokens = len(job.token_ids)
    elapsed_seconds = finished_at - job.generation_started_at
    output_tokens_per_second = 0.0
    if elapsed_seconds > 0:
        output_tokens_per_second = output_tokens / elapsed_seconds
    generation_tpot_ms = None
    if output_tokens > 1:
        # Engine time per output token averages the gaps after the first token.
        generation_tpot_ms = (finished_at - job.first_token_at) * 1000 / (output_tokens - 1)

    return {
        "finish_reason": finish_reason,
        "generation_tpot_ms": round(generation_tpot_ms, 2) if generation_tpot_ms is not None else None,
        "ttft_ms": round((job.first_token_at - job.generation_started_at) * 1000, 2),
        "latency_ms": round(elapsed_seconds * 1000, 2),
        "input_tokens": len(job.prompt_token_ids),
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens_per_second, 2),
    }
