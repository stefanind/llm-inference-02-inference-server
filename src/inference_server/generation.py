"""Token sampling and continuous generation with per-request KV caches."""

import time
from typing import Optional

import torch

from inference_server.cache import combine_caches, save_caches
from inference_server.engine import schedule_step


def tokenize_prompt(prompt, tokenizer):
    """Prepare the full chat prompt once, before context checks and admission."""
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
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    if top_k is not None:
        top_values = torch.topk(scaled_logits, top_k, dim=-1).values
        cutoff = top_values[..., -1:]
        scaled_logits = scaled_logits.masked_fill(
            scaled_logits < cutoff,
            float('-inf')
        )

    probabilities = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def forward_tokens(jobs, sequences, model, tokenizer, metrics, phase):
    """Process prompt chunks or decode tokens, reusing each request's history."""
    cache, past_width = combine_caches(jobs)
    # Left-pad histories and new inputs separately. Position IDs count only
    # real tokens, so a resumed prompt keeps the same positions as a whole one.
    width = max(len(sequence) for sequence in sequences)
    input_ids = []
    masks = []
    positions = []
    for job, sequence in zip(jobs, sequences):
        past_length = 0
        if job.kv_cache is not None:
            past_length = job.kv_cache[0][0].shape[2]
        padding = width - len(sequence)
        input_ids.append([tokenizer.pad_token_id] * padding + sequence)
        masks.append(
            [0] * (past_width - past_length) + [1] * past_length
            + [0] * padding + [1] * len(sequence)
        )
        positions.append(
            [0] * padding + list(range(past_length, past_length + len(sequence)))
        )
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=model.device)
    position_ids = torch.tensor(positions, dtype=torch.long, device=model.device)
    if metrics is not None:
        metrics.record_forward(phase, len(jobs))
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    save_caches(jobs, outputs.past_key_values, past_width, [len(sequence) for sequence in sequences])
    return outputs.logits[:, -1, :]


def prefill(jobs, chunk_lengths, model, tokenizer, metrics=None):
    """Process just the scheduled part of each prompt and remember our progress."""
    sequences = []
    for job, length in zip(jobs, chunk_lengths):
        start = job.prefill_tokens
        sequences.append(job.prompt_token_ids[start:start + length])
    logits = forward_tokens(jobs, sequences, model, tokenizer, metrics, "prefill")
    for job, length in zip(jobs, chunk_lengths):
        job.prefill_tokens += length
    return logits


def decode(jobs, model, tokenizer, metrics=None):
    """Process only each request's newest token using its cached history."""
    sequences = [[job.token_ids[-1]] for job in jobs]
    return forward_tokens(jobs, sequences, model, tokenizer, metrics, "decode")


def finish_step(jobs, logits, tokenizer):
    """Select one token per request and finish requests that hit EOS or a limit."""
    for index, job in enumerate(jobs):
        token = select_next_token(
            logits[index:index + 1], job.request.temperature, job.request.top_k,
        ).item()
        job.token_ids.append(token)
        ready_at = time.perf_counter()
        if job.first_token_at is None:
            job.first_token_at = ready_at
        finished = token == tokenizer.eos_token_id or len(job.token_ids) >= job.request.max_new_tokens
        # Decode the history together: one token may contain only part of a UTF-8
        # character. Hold an incomplete trailing character until more bytes arrive.
        text = tokenizer.decode(
            job.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        if finished or not text.endswith("\ufffd"):
            job.text_delta = text[job.text_sent:]
            job.text_sent = len(text)
        if finished:
            reason = "stop" if token == tokenizer.eos_token_id else "length"
            job.response = build_response(job, ready_at, reason)


@torch.inference_mode()
def generate_step(active, model, tokenizer, max_batched_tokens=512, metrics=None):
    """Decode first, then process prompt chunks within the remaining budget."""
    decoding, chunks = schedule_step(active, max_batched_tokens)
    # Keep these as separate forward passes. A long prefill yields to decode
    # again on the next scheduler iteration instead of running to completion.
    if decoding:
        finish_step(decoding, decode(decoding, model, tokenizer, metrics), tokenizer)
    if chunks:
        jobs = [job for job, length in chunks]
        chunk_lengths = [length for job, length in chunks]
        logits = prefill(jobs, chunk_lengths, model, tokenizer, metrics)
        for index, job in enumerate(jobs):
            # Intermediate chunks build the cache without generating an answer.
            if job.prefill_tokens == len(job.prompt_token_ids):
                finish_step([job], logits[index:index + 1], tokenizer)


def build_response(job, finished_at, finish_reason):
    """Final stream metadata. Text was already emitted in incremental chunks."""
    output_tokens = len(job.token_ids)
    elapsed_seconds = finished_at - job.generation_started_at
    output_tokens_per_second = 0.0
    if elapsed_seconds > 0:
        output_tokens_per_second = output_tokens / elapsed_seconds
    generation_tpot_ms = None
    if output_tokens > 1:
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
