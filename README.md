# Simple inference server

Learn GPU inference serving with streaming, continuous batching, and a KV cache
per request. The scheduler runs up to four active requests and holds up to 20
more in a FIFO queue.

[Run 01 benchmark results](BENCHMARK.md) compare this server with vLLM on an RTX 3090.

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

In the server terminal, install the repo and vLLM in separate environments.
Use the same checkpoint for both. Set a fresh `run` name for each repeat.
[Runpod image](https://github.com/runpod/runpodctl/blob/main/docs/runpodctl_pod_create.md) ·
[vLLM CUDA 12.8 install](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/#pre-built-wheels)

```bash
export HF_HOME=/workspace/hf-cache
MODEL=Qwen/Qwen3-4B
REVISION=1cfa9a7208912126459214e8b04321603b3df60c
run=run-02
mkdir -p "results/$run"
nvidia-smi | tee "results/$run/gpu.txt"
python -m pip install -e .
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
python -m venv /workspace/vllm-venv
/workspace/vllm-venv/bin/python -m pip install --no-cache-dir vllm==0.11.2 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

## Run the comparison

Start this repo's server in the server terminal and wait for startup to finish.

```bash
python -m inference_server.cli --host 127.0.0.1 --revision "$REVISION" \
  2>&1 | tee "results/$run/repo.log"
```

Paste this in the client terminal. Use the repo's Python environment for both
backends and the same `run` name as in the server terminal.

```bash
run=run-02
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
```

Warm up with W2. Check that all 20 requests completed and `gpu_metrics` has
samples with an empty `errors` list.

```bash
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

Stop the repo server with Ctrl+C. Start vLLM in the same server terminal.
This uses the same context, sequence, and token limits and disables reuse
of prompt prefixes across requests. [vLLM serve options](https://docs.vllm.ai/en/v0.11.2/cli/serve/)

```bash
/workspace/vllm-venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --revision "$REVISION" --tokenizer-revision "$REVISION" --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --enable-chunked-prefill --no-enable-prefix-caching --generation-config vllm \
  2>&1 | tee "results/$run/vllm.log"
```

When vLLM is ready, switch the client backend. Repeat the **warmup and
measured suite** blocks above, starting with `out=` so the files go to
`results/run-02/vllm/`.

```bash
backend=vllm
gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
backend_args=(--model Qwen/Qwen3-4B --gpu-uuid "$gpu_uuid")
```

## Profile prefill and decode

Stop the running server and start the repo server with `--profile`.

```bash
python -m inference_server.cli --host 127.0.0.1 --revision "$REVISION" --profile \
  2>&1 | tee "results/$run/profile.log"
```

In the client terminal, reuse `run_workload` and the W4 prompts from above.

```bash
backend=repo
backend_args=()
out="results/$run/profile"
mkdir -p "$out"
run_workload W2 > "$out/warmup.json"
run_workload W2 > "$out/W2.json"
run_workload W4 --short-prompt "$short_prompt" --long-prompt "$long_prompt" > "$out/W4.json"

for workload in W2 W4; do
  echo "$workload"
  python -c 'import json, sys; print(json.dumps(json.load(sys.stdin)["engine_metrics"]["profile"], indent=2))' < "$out/$workload.json"
done
```

`engine_metrics.profile` shows `calls` and `total_ms` for batch setup
(including joining the KV caches), model forward, cache saving, and token
handling, tracked separately for prefill and decode. Token handling includes
the prefill check for whether a prompt is ready to produce its first token.
The runner resets the counters before each workload. Queue wait is recorded separately.

Use `total_ms` to find the largest cost and divide by `calls` for the average
per batch. These timings include CPU work and wait for CUDA to finish at each
boundary. That adds overhead, so restart without `--profile` for throughput
and latency comparisons. [PyTorch CUDA synchronization](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.synchronize.html)

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
