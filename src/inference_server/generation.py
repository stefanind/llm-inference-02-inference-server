"""The original prompt preparation, sampling, and generation functions."""

import time
from typing import Optional

import torch

from inference_server.model import model, tokenizer, sync_gpu


def prepare_inputs(prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    )

    assert messages, "Add the prompt messages"
    assert inputs is not None, "Render the prompt with the tokenizer"
    return {key: value.to(model.device) for key, value in inputs.items()}


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
    next_token = torch.multinomial(probabilities, num_samples=1)
    assert next_token is not None, "Sample one token"
    return next_token


@torch.inference_mode()
def generate_uncached(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> dict:
    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"] # B, S
    attention_mask = inputs["attention_mask"] # B, S
    prompt_length = generated_ids.shape[1]
    first_token_at = None

    sync_gpu()
    started_at = time.perf_counter()
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        next_token = select_next_token( # B, 1
            outputs.logits[:, -1, :], # B S vocab
            temperature,
            top_k,
        )
        if first_token_at is None:
            sync_gpu()
            first_token_at = time.perf_counter() # ttft

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    sync_gpu()
    finished_at = time.perf_counter()
    completion_ids = generated_ids[0, prompt_length:]
    return {
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "token_ids": completion_ids.tolist(),
        "input_tokens": prompt_length,
        "ttft_ms": (first_token_at - started_at) * 1000,
        "latency_ms": (finished_at - started_at) * 1000,
    }


@torch.inference_mode()
def generate_cached(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> dict:
    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]
    first_token_at = None

    sync_gpu()
    started_at = time.perf_counter()
    # prefill
    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        use_cache=True, # returns past_key_values as an attribute
    )
    # have the key values
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        next_token = select_next_token( # B, 1
            outputs.logits[:, -1, :], # B S vocab
            temperature,
            top_k,
        )
        assert next_token is not None, "Select the next token"
        if first_token_at is None:
            sync_gpu()
            first_token_at = time.perf_counter()

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

        attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(next_token),
        ], dim=1)

        # only pass next_token since we have kv cache
        # return the next kv cache
        outputs = model(
            input_ids=next_token, # B 1
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = outputs.past_key_values

    sync_gpu()
    finished_at = time.perf_counter()
    completion_ids = generated_ids[0, prompt_length:]
    return {
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "token_ids": completion_ids.tolist(),
        "input_tokens": prompt_length,
        "ttft_ms": (first_token_at - started_at) * 1000,
        "latency_ms": (finished_at - started_at) * 1000,
    }


def summarize_result(result: dict, used_kv_cache: bool) -> dict:

    output_tokens = len(result['token_ids'])
    # ITL
    # decode_time_seconds = (
    #     result["latency_ms"] - result["ttft_ms"]
    # ) / 1000

    # output_tokens_per_second = (
    #     (output_tokens - 1) / decode_time_seconds
    #     if output_tokens > 1 and decode_time_seconds > 0
    #     else 0.0
    # )
    # output_tokens = len(result["token_ids"])

    total_time_seconds = result["latency_ms"] / 1000

    output_tokens_per_second = (
        output_tokens / total_time_seconds
        if total_time_seconds > 0
        else 0.0
    )

    return {
        "text": result["text"],
        "ttft_ms": round(result["ttft_ms"], 2),
        "latency_ms": round(result["latency_ms"], 2),
        "input_tokens": result["input_tokens"],
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens_per_second, 2),
        "used_kv_cache": used_kv_cache,
    }


@torch.inference_mode()
def generate_stream(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
):
    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]

    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    previous_text = ""

    for _ in range(max_new_tokens):
        next_token = select_next_token( # B, 1
            outputs.logits[:, -1, :], # B S vocab
            temperature,
            top_k,
        )
        assert next_token is not None, "Select the next token"
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        current_text = tokenizer.decode(
            generated_ids[0, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        chunk = current_text[len(previous_text):]
        if chunk:
            yield chunk
        previous_text = current_text

        if next_token.item() == tokenizer.eos_token_id:
            break

        attention_mask = torch.cat([
            attention_mask,
            torch.ones(
                (1, 1),
                dtype=attention_mask.dtype,
                device=model.device,
            ),
        ], dim=1)
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
