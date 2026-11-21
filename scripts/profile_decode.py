"""Trace warmed W2 decode or compare eager decode with CUDA Graph replay."""

import argparse
import json
import time
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from inference_server.cache import SlotCache
from inference_server.decode_compile import DecodeNorms
from inference_server.decode_graph import GRAPH_CONTEXT, GRAPH_WARMUP_STEPS, FixedDecodeBatch
from inference_server.engine import ActiveRequest
from inference_server.generation import decode, generate_step, prefill, tokenize_prompt
from inference_server.model import load_model
from inference_server.workloads import make_w2


WARMUP_STEPS = 4
TRACE_STEPS = 8
GRAPH_STEPS = 31
GRAPH_TRIALS = 4
LOGIT_TOLERANCE = 0.01


def make_job(prompt_ids, slot):
    request = SimpleNamespace(temperature=0.0, top_k=None, max_new_tokens=32)
    started_at = time.perf_counter()
    return ActiveRequest(
        request, started_at, prompt_token_ids=prompt_ids,
        cache_slot=slot, generation_started_at=started_at,
    )


def make_trace_batch(tokenizer):
    # One- and two-digit W2 user IDs give different prompt lengths on Qwen3.
    requests = make_w2()
    requests = requests[:2] + requests[9:11]
    prompts = [tokenize_prompt(request.prompt, tokenizer) for request in requests]
    return requests, prompts


def make_prefilled_batch(model, tokenizer, prompts):
    cache = SlotCache(model, slots=len(prompts), tokens_per_slot=GRAPH_CONTEXT)
    jobs = [make_job(prompt, slot) for slot, prompt in enumerate(prompts)]
    prefill(jobs, [len(prompt) for prompt in prompts], model, tokenizer, cache, None)
    return FixedDecodeBatch(model, cache)


def check_graph_logits(eager, replay, graph, graph_logits, steps):
    max_difference = 0.0
    for step, (tokens, positions) in enumerate(steps, start=1):
        eager.load_step(tokens, positions)
        expected = eager.forward()
        replay.load_step(tokens, positions)
        graph.replay()
        torch.testing.assert_close(graph_logits, expected, rtol=LOGIT_TOLERANCE, atol=LOGIT_TOLERANCE)
        if not torch.equal(graph_logits.argmax(dim=-1), expected.argmax(dim=-1)):
            raise RuntimeError(f"CUDA Graph and eager chose different tokens at decode step {step}.")
        difference = (graph_logits.float() - expected.float()).abs().max().item()
        max_difference = max(max_difference, difference)
    return max_difference


def time_decode(batch, forward, steps):
    device = batch.model.device
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    for tokens, positions in steps:
        batch.load_step(tokens, positions)
        forward()
        # A real decode loop needs this step's result before it can continue.
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started_at) * 1000 / len(steps)


@torch.inference_mode()
def run_graph_diagnostic(model, tokenizer, output_dir):
    requests, prompts = make_trace_batch(tokenizer)
    prompt_lengths = [len(prompt) for prompt in prompts]
    if max(prompt_lengths) + GRAPH_STEPS >= GRAPH_CONTEXT:
        raise ValueError("The W2 prompts and decode steps must fit before the warmup position.")

    # Separate caches prevent an eager forward from filling replay's history.
    eager = make_prefilled_batch(model, tokenizer, prompts)
    replay = make_prefilled_batch(model, tokenizer, prompts)
    pattern = tokenizer.encode(" Explain KV caching. The tokens stay in memory.", add_special_tokens=False)
    forced_ids = [
        [pattern[(step + row) % len(pattern)] for row in range(len(prompts))]
        for step in range(GRAPH_STEPS)
    ]
    # Stage the inputs once. Both timings include the copies into the fixed
    # buffers and the attention mask update for each new position.
    token_steps = torch.tensor(forced_ids, device=model.device).unsqueeze(-1)
    start_positions = torch.tensor(prompt_lengths, device=model.device).unsqueeze(-1)
    steps = [(token_steps[step], start_positions + step) for step in range(GRAPH_STEPS)]
    # Warm up beyond the measured history so every checked position is fresh.
    eager.load_step(token_steps[0], torch.full_like(token_steps[0], GRAPH_CONTEXT - 1))
    for _ in range(GRAPH_WARMUP_STEPS):
        eager.forward()
    graph, graph_logits = replay.capture(token_steps[0])
    max_difference = check_graph_logits(eager, replay, graph, graph_logits, steps)

    modes = [("eager", eager, eager.forward), ("cuda_graph", replay, graph.replay)]
    samples = {"eager": [], "cuda_graph": []}
    for _ in range(GRAPH_TRIALS):
        for name, batch, forward in modes:
            samples[name].append(time_decode(batch, forward, steps))
        modes.reverse()  # Alternate which mode runs first in each trial.
    medians = {name: median(times) for name, times in samples.items()}

    report = {
        "model": model.name_or_path,
        "revision": getattr(model.config, "_commit_hash", None),
        "device": str(model.device),
        "dtype": str(model.dtype),
        "gpu_name": torch.cuda.get_device_name(model.device),
        "gpu_uuid": str(torch.cuda.get_device_properties(model.device).uuid),
        "torch_version": torch.__version__,
        "compile_decode": False,
        "w2_requests": [request.request_id for request in requests],
        "w2_prompt_tokens": prompt_lengths,
        "kv_positions": GRAPH_CONTEXT,
        "warmup_eager_steps_per_cache": GRAPH_WARMUP_STEPS,
        "decode_steps_per_trial": GRAPH_STEPS,
        "forced_token_ids_by_step": forced_ids,
        "logit_rtol": LOGIT_TOLERANCE,
        "logit_atol": LOGIT_TOLERANCE,
        "matching_top_tokens": len(prompts) * GRAPH_STEPS,
        "max_abs_logit_diff": max_difference,
        "trial_ms_per_step": samples,
        "median_ms_per_step": medians,
        "speedup": medians["eager"] / medians["cuda_graph"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cuda-graph.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved logit checks and eager/replay timings to {path}")


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
    requests, prompts = make_trace_batch(tokenizer)
    cache = SlotCache(model, slots=4, tokens_per_slot=min(4096, model.config.max_position_embeddings))
    output_dir.mkdir(parents=True, exist_ok=True)

    forced_ids, comparisons = check_slot_reuse(model, tokenizer, cache, prompts[0])
    report = {
        "model": model.name_or_path,
        "revision": getattr(model.config, "_commit_hash", None),
        "device": str(model.device),
        "dtype": str(model.dtype),
        "gpu_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
        "compile_decode": getattr(model, "decode_norms", None) is not None,
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compile-decode", action="store_true", help="Compile Qwen3's decoder-layer RMSNorms during decode")
    modes.add_argument("--cuda-graph", action="store_true", help="Compare eager and CUDA Graph replay for a fixed decode batch")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    model, tokenizer = load_model(revision=args.revision)
    if args.cuda_graph:
        run_graph_diagnostic(model, tokenizer, args.output_dir or Path("results/run-07/decode-graph"))
        return
    if args.compile_decode:
        model.decode_norms = DecodeNorms(model)
    run_diagnostic(model, tokenizer, args.output_dir or Path("results/run-05/decode-trace"))


if __name__ == "__main__":
    main()
