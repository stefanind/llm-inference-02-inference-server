"""Check Qwen3 cache logits and capture a short GPU trace of warmed W2 decode."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from inference_server.cache import SlotCache
from inference_server.engine import ActiveRequest
from inference_server.generation import decode, generate_step, prefill, tokenize_prompt
from inference_server.model import load_model
from inference_server.workloads import make_w2


WARMUP_STEPS = 4
TRACE_STEPS = 8


def make_job(prompt_ids, slot):
    request = SimpleNamespace(temperature=0.0, top_k=None, max_new_tokens=32)
    started_at = time.perf_counter()
    return ActiveRequest(
        request, started_at, prompt_token_ids=prompt_ids,
        cache_slot=slot, generation_started_at=started_at,
    )


def compare_logits(model, jobs, logits, step):
    comparisons = []
    for job, scores in zip(jobs, logits):
        # Recompute one sequence with plain Transformers. Both paths see
        # exactly the same context, regardless of which token they predict.
        context = job.prompt_token_ids + job.token_ids
        input_ids = torch.tensor([context], device=model.device)
        reference = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        expected = reference.logits[0, -1].float()
        scores = scores.float()
        difference = (scores - expected).abs()
        top_scores, top_ids = expected.topk(2)
        comparisons.append({
            "slot": job.cache_slot,
            "case": "reused_slot" if job.cache_slot == 0 else "continued_prompt",
            "decode_step": step,
            "context_tokens": len(context),
            "max_abs_logit_diff": difference.max().item(),
            "mean_abs_logit_diff": difference.mean().item(),
            "reference_top_token": top_ids[0].item(),
            "fixed_slot_top_token": scores.argmax().item(),
            "reference_top_two_margin": (top_scores[0] - top_scores[1]).item(),
        })
    return comparisons


def check_slot_reuse(model, tokenizer, cache, short_prompt_ids):
    # Fill slot 0 with a longer history and leave slot 3 midway through prefill.
    old = make_job(tokenize_prompt("Describe a request queue. " * 8, tokenizer), slot=0)
    peer = make_job(tokenize_prompt("Explain why a KV cache helps. " * 12, tokenizer), slot=3)
    half = len(peer.prompt_token_ids) // 2
    prefill([old, peer], [len(old.prompt_token_ids), half], model, tokenizer, cache, None)
    old.release_cache_slot()

    # The new request must ignore the old contents. Pass the rows in request
    # order (3, 0) to check the gap between slots and the returned logits order.
    reused = make_job(short_prompt_ids, slot=0)
    jobs = [peer, reused]
    remaining = len(peer.prompt_token_ids) - half
    logits = prefill(jobs, [remaining, len(short_prompt_ids)], model, tokenizer, cache, None)
    comparisons = compare_logits(model, jobs, logits, step=0)

    forced_ids = tokenizer.encode(" Explain KV caching. The tokens stay in memory.", add_special_tokens=False)[:4]
    for step, token_id in enumerate(forced_ids, start=1):
        for job in jobs:
            job.token_ids.append(token_id)
        logits = decode(jobs, model, tokenizer, cache, None)
        comparisons.extend(compare_logits(model, jobs, logits, step))
    return forced_ids, comparisons


def traced_call(name, function):
    def call(*args, **kwargs):
        with record_function(name):
            return function(*args, **kwargs)
    return call


def trace_decode(model, tokenizer, cache, jobs, output_dir):
    generate_step(jobs, model, tokenizer, cache)  # Prefill and the first token.

    def step():
        if any(job.response is not None for job in jobs):
            raise RuntimeError("A W2 request finished before the four-request trace ended.")
        generate_step(jobs, model, tokenizer, cache)

    for _ in range(WARMUP_STEPS):
        step()

    activities = [ProfilerActivity.CPU]
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        activities.append(ProfilerActivity.CUDA)

    # These markers are installed only for the trace. The serving code uses
    # the same forward, cache update, and token handling as usual.
    with (
        patch.object(model, "forward", traced_call("model_forward", model.forward)),
        patch.object(cache, "update", traced_call("kv_write", cache.update)),
        profile(activities=activities) as trace,
    ):
        for _ in range(TRACE_STEPS):
            with record_function("decode_step"):
                step()

    trace.export_chrome_trace(str(output_dir / "decode-trace.json"))
    operators = trace.key_averages()
    tables = ["Top operations by self CPU time", operators.table(sort_by="self_cpu_time_total", row_limit=15)]
    if model.device.type == "cuda":
        tables.extend(["Top operations by self CUDA time", operators.table(sort_by="self_device_time_total", row_limit=15)])
    (output_dir / "operators.txt").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    if model.device.type == "cuda" and not any(event.self_device_time_total for event in operators):
        raise RuntimeError("The trace was saved, but PyTorch did not collect CUDA timings.")


@torch.inference_mode()
def run_diagnostic(model, tokenizer, output_dir):
    # One- and two-digit W2 user IDs give different prompt lengths on Qwen3.
    requests = make_w2()
    requests = requests[:2] + requests[9:11]
    prompts = [tokenize_prompt(request.prompt, tokenizer) for request in requests]
    cache = SlotCache(model, slots=4, tokens_per_slot=min(4096, model.config.max_position_embeddings))
    output_dir.mkdir(parents=True, exist_ok=True)

    forced_ids, comparisons = check_slot_reuse(model, tokenizer, cache, prompts[0])
    report = {
        "model": model.name_or_path,
        "revision": getattr(model.config, "_commit_hash", None),
        "device": str(model.device),
        "dtype": str(model.dtype),
        "gpu_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
        "w2_requests": [request.request_id for request in requests],
        "w2_prompt_tokens": [len(prompt) for prompt in prompts],
        "warmup_decode_steps": WARMUP_STEPS,
        "trace_decode_steps": TRACE_STEPS,
        "forced_token_ids": forced_ids,
        "logit_comparisons": comparisons,
    }
    (output_dir / "logits-check.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    matches = sum(row["reference_top_token"] == row["fixed_slot_top_token"] for row in comparisons)
    print(f"Reference top token matched in {matches}/{len(comparisons)} comparisons.", flush=True)

    # New jobs start at length zero and reuse the already allocated buffers.
    jobs = [make_job(prompt, slot) for slot, prompt in enumerate(prompts)]
    trace_decode(model, tokenizer, cache, jobs, output_dir)
    print(f"Saved the logits check, decode trace, and operator tables to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Model and tokenizer checkpoint revision")
    parser.add_argument("--output-dir", type=Path, default=Path("results/run-05/decode-trace"))
    args = parser.parse_args()
    model, tokenizer = load_model(revision=args.revision)
    run_diagnostic(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
