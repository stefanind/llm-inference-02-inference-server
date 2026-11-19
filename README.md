# Simple inference server

Learn GPU inference serving with streaming, continuous batching, and fixed KV
cache slots. The scheduler runs up to four active requests and holds up to 20
more in a FIFO queue. Each active request writes new tokens into its own slot.
By default, the server allocates four 4,096-token slots at startup and limits
total active reservations to 8,192 tokens.

[Run 01 benchmark results](BENCHMARK.md) compare this server with vLLM on an RTX 3090.
[How it works](HOW_IT_WORKS.md) follows a request through the source code.

| Workload | What it tests |
| --- | --- |
| W1 | One request for a single-request baseline. |
| W2 | 20 simultaneous requests to exercise batching and queueing. |
| W3 | 20 requests arriving at random times over 10 seconds. |
| W4 | 20 requests mixing short and long prompts with 32- and 256-token output limits. |
| W5 | A burst of 400 requests from 20 users to test overload and rejection. |
| W6 | Steady traffic at 1, 2, 5, 10, and 20 requests/s to find saturation. |

## Set up Runpod

Use one BF16 GPU with at least 24 GB of VRAM and the image
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`. Open two Bash terminals
on the Pod in the repo folder under `/workspace`. Run one server at a time
and use localhost for the client.

In the server terminal, install the repo and set a fresh `run` name for each repeat.
[Runpod image](https://github.com/runpod/runpodctl/blob/main/docs/runpodctl_pod_create.md)

```bash
export HF_HOME=/workspace/hf-cache
MODEL=Qwen/Qwen3-4B
REVISION=1cfa9a7208912126459214e8b04321603b3df60c
run=run-03
mkdir -p "results/$run"
nvidia-smi | tee "results/$run/gpu.txt"
python -m pip install -e .
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
```

## Profile W2 and W4

This is the next experiment. It uses the repo server, one W2 warmup,
then measured profiles for W2 and W4.

Start the repo server with `--profile` after stopping any server already on port 8000.

```bash
python -m inference_server.cli --host 127.0.0.1 --revision "$REVISION" --profile \
  2>&1 | tee "results/$run/profile.log"
```

When the server is ready, paste the whole block into the client terminal.
Adjust the repo path if needed, and use the same run name as in the server
terminal. The reports go under `results/run-03/profile/` by default.

```bash
(
  set -e
  cd /workspace/llm-inference-02-inference-server
  out="results/run-03/profile"
  mkdir -p "$out"

  short_prompt="Explain continuous batching in 12 numbered points."
  long_prompt="$short_prompt Here is some context:"
  for i in {1..48}; do
    long_prompt+=" A request waits in a queue until the scheduler has room to process it."
  done

  report=$(python -m inference_server.cli run W2 --timeout 300)
  printf '%s\n' "$report" > "$out/warmup.json"
  report=$(python -m inference_server.cli run W2 --timeout 300)
  printf '%s\n' "$report" > "$out/W2.json"
  report=$(python -m inference_server.cli run W4 --timeout 300 \
    --short-prompt "$short_prompt" --long-prompt "$long_prompt")
  printf '%s\n' "$report" > "$out/W4.json"

  for workload in W2 W4; do
    echo "$workload"
    python -c 'import json, sys; print(json.dumps(json.load(sys.stdin)["engine_metrics"]["profile"], indent=2))' < "$out/$workload.json"
  done
)
```

`engine_metrics.profile` shows `calls` and `total_ms` for prefill and decode.
Batch setup prepares inputs and masks. The model forward now includes KV writes
into the fixed slots. `cache_save` stays at zero because there is no separate
copy back after the forward. Compare total decode time with Run 02 and check
GPU memory separately, since the fixed slots are allocated before warmup.

Use `total_ms` to find the largest cost and divide by `calls` for the average
per batch. These timings include CPU work and wait for CUDA to finish at each
boundary. That adds overhead, so measure throughput and latency with
`--profile` off. For a before-and-after comparison, run both revisions on the
same Pod. [PyTorch CUDA synchronization](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.synchronize.html)

## Trace model forward

Stop the server to free the GPU. From the repo folder on the Pod, run:

```bash
HF_HOME=/workspace/hf-cache python scripts/profile_decode.py \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

Look in `results/run-05/decode-trace/`. `logits-check.json` compares the
fixed slots against plain Transformers after slot reuse and for four forced
decode tokens. It reports score differences and top token matches. The script
then traces eight warmed decode steps on four W2 requests. Open
`decode-trace.json` in a Chrome trace viewer and look for `model_forward` and
`kv_write`. `operators.txt` ranks CPU and CUDA operations across the captured
steps. Use the normal W2/W4 runs for throughput comparisons.
[PyTorch profiler](https://docs.pytorch.org/docs/2.9/profiler.html)

## Compare compiled decode

`--compile-decode` compiles the two hidden-state RMSNorms in each Qwen3 layer
during decode. They share a compiled function because each receives one token
per row. Run 05 showed many small operations inside these norms that the compiler
can fuse. Prefill runs eager. [PyTorch compile](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html)

On the same Pod and model revision, run the server twice. Start with the first
command, warm up and measure W2, then stop it and repeat with the second.

```bash
python -m inference_server.cli --host 127.0.0.1 --revision "$REVISION"
python -m inference_server.cli --host 127.0.0.1 --revision "$REVISION" --compile-decode
```

Run this in the client terminal for each server. Set `mode=compiled` on the
second pass. The first decode calls compile on demand, so keep the warmup
separate from the measurement.

```bash
mode=eager
out="results/run-06/$mode"
mkdir -p "$out"
python -m inference_server.cli run W2 --timeout 300 > "$out/warmup.json" &&
  python -m inference_server.cli run W2 --timeout 300 > "$out/W2.json"
```

Check for 20 successes in each warmup and measured report. Compare output
tokens per second and generation time per output token. The report records the
flag under `run_info.server.settings.compile_decode`. Stop the server before
tracing and save each trace separately.

```bash
python scripts/profile_decode.py --revision "$REVISION" \
  --output-dir results/run-06/eager/trace
python scripts/profile_decode.py --revision "$REVISION" --compile-decode \
  --output-dir results/run-06/compiled/trace
```

## Optional full comparison (W1–W6)

Use this section to reproduce the Run 01 comparison. Stop the profiling server
first. Start the repo server in the server terminal and wait for startup to finish.

```bash
python -m inference_server.cli --host 127.0.0.1 --revision "$REVISION" \
  2>&1 | tee "results/$run/repo.log"
```

In the client terminal, use the repo's Python environment and the same run
name as in the server terminal. Warm up with W2 and check for 20 completions
and GPU samples with an empty `errors` list.

```bash
run=run-03
backend=repo
backend_args=()
short_prompt="Explain continuous batching in 12 numbered points."
long_prompt="$short_prompt Here is some context:"
for i in {1..48}; do
  long_prompt+=" A request waits in a queue until the scheduler has room to process it."
done

run_workload() {
  python -m inference_server.cli run "$@" --backend "$backend" --timeout 300 "${backend_args[@]}"
}

out="results/$run/$backend"
mkdir -p "$out"
run_workload W2 > "$out/warmup.json"
python -c 'import json, sys; r = json.load(sys.stdin); print(r["summary"]); print(r["gpu_metrics"])' < "$out/warmup.json"
```

Run the measured suite. Each W6 rate sends requests for 30 seconds, then
waits for the responses.

```bash
run_workload W1 > "$out/W1.json"
run_workload W2 > "$out/W2.json"
run_workload W3 --seed 42 > "$out/W3.json"
run_workload W4 --short-prompt "$short_prompt" --long-prompt "$long_prompt" > "$out/W4.json"
run_workload W5 > "$out/W5.json"
for rps in 1 2 5 10 20; do
  run_workload W6 --rps "$rps" --duration 30 > "$out/W6-${rps}rps.json"
done
```

Stop the repo server with Ctrl+C. Install vLLM in its own environment if
needed, then start it in the same server terminal. This uses the same context,
sequence, and token limits and disables reuse of prompt prefixes across requests.
[vLLM CUDA 12.8 install](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/#pre-built-wheels) ·
[vLLM serve options](https://docs.vllm.ai/en/v0.11.2/cli/serve/)

```bash
python -m venv /workspace/vllm-venv
/workspace/vllm-venv/bin/python -m pip install --no-cache-dir vllm==0.11.2 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

```bash
/workspace/vllm-venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --revision "$REVISION" --tokenizer-revision "$REVISION" --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --enable-chunked-prefill --no-enable-prefix-caching --generation-config vllm \
  2>&1 | tee "results/$run/vllm.log"
```

When vLLM is ready, paste this in the client terminal instead of the repo
warmup block. Then repeat the **measured suite** block above.

```bash
backend=vllm
gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
backend_args=(--model Qwen/Qwen3-4B --gpu-uuid "$gpu_uuid")
out="results/$run/$backend"
mkdir -p "$out"
run_workload W2 > "$out/warmup.json"
python -c 'import json, sys; r = json.load(sys.stdin); print(r["summary"]); print(r["gpu_metrics"])' < "$out/warmup.json"
```

## Read the results

Use completed requests/s, output tokens/s, and client P95 time to first text
for comparisons. Show completion counts alongside latency for W5 and W6.
Short runs can have few GPU samples. Keep warmups out of the plots and copy
the run directory off the Pod before terminating it.
[Runpod file transfer](https://docs.runpod.io/pods/storage/transfer-files)

## Local checks

```bash
python -m unittest discover -s tests -v
python tests/local_smoke.py
```
