"""Combine and split per-request KV caches for full-attention models like Qwen3."""

import torch
from transformers import DynamicCache


def combine_caches(jobs):
    """Batch histories, leaving empty rows for prompts with no cache yet."""
    cached = [job for job in jobs if job.kv_cache is not None]
    if not cached:
        return None, 0

    # Each layer has shape [1, heads, cached tokens, head dimension].
    width = max(job.kv_cache[0][0].shape[2] for job in cached)
    reference = cached[0].kv_cache
    layers = []
    for layer_index in range(len(reference)):
        keys = []
        values = []
        for job in jobs:
            if job.kv_cache is None:
                # A new prompt can share a prefill batch with a resumed prompt.
                key, value = reference[layer_index]
                keys.append(torch.zeros(
                    (1, key.shape[1], width, key.shape[3]), dtype=key.dtype, device=key.device,
                ))
                values.append(torch.zeros(
                    (1, value.shape[1], width, value.shape[3]), dtype=value.dtype, device=value.device,
                ))
                continue
            key, value = job.kv_cache[layer_index]
            padding = width - key.shape[2]
            key_padding = torch.zeros(
                (1, key.shape[1], padding, key.shape[3]), dtype=key.dtype, device=key.device,
            )
            value_padding = torch.zeros(
                (1, value.shape[1], padding, value.shape[3]), dtype=value.dtype, device=value.device,
            )
            keys.append(torch.cat([key_padding, key], dim=2))
            values.append(torch.cat([value_padding, value], dim=2))
        layers.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
    return DynamicCache(ddp_cache_data=layers), width


def save_caches(jobs, cache, past_width, input_lengths):
    """Keep real history and new tokens, removing padding from both sections."""
    if any(cache.is_sliding):
        raise ValueError("Continuous cache reuse currently supports full-attention caches")
    for index, job in enumerate(jobs):
        past_length = 0
        if job.kv_cache is not None:
            past_length = job.kv_cache[0][0].shape[2]
        input_length = input_lengths[index]
        job.kv_cache = []
        for layer in cache.layers:
            # The model appends padded inputs after the padded history. There
            # may be padding between the two, so select their real parts first.
            past_key = layer.keys[index:index + 1, :, past_width - past_length:past_width, :]
            past_value = layer.values[index:index + 1, :, past_width - past_length:past_width, :]
            new_key = layer.keys[index:index + 1, :, -input_length:, :]
            new_value = layer.values[index:index + 1, :, -input_length:, :]
            # cat() makes independent tensors, releasing the batch's storage.
            key = torch.cat([past_key, new_key], dim=2)
            value = torch.cat([past_value, new_value], dim=2)
            job.kv_cache.append((key, value))
