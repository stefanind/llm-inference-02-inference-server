# Simple inference server

A GPU inference server for learning continuous batching, with
up to four active requests, and a FIFO queue for 20 waiting requests.
Excess requests receive HTTP 429.

## Compare on Runpod

Use `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` on one BF16 GPU with
at least 24 GB of VRAM. Run both servers on that same GPU, one at a time.
[Runpod image reference](https://github.com/runpod/runpodctl/blob/main/docs/runpodctl_pod_create.md)

Keep the repo and model cache under `/workspace`. Use two Bash terminals on
the Pod in the repo folder. The benchmark connects over localhost.
[Runpod storage options](https://docs.runpod.io/pods/storage/types)

In the server terminal, install both environments and pin one Qwen3-4B
checkpoint for both servers. vLLM 0.11.2 has a CUDA 12.8 build and uses its
own PyTorch version.
[Qwen checkpoint](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c)
and [vLLM installation](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/#pre-built-wheels).

```bash
export HF_HOME=/workspace/hf-cache
MODEL=Qwen/Qwen3-4B
REVISION=1cfa9a7208912126459214e8b04321603b3df60c
mkdir -p results/run-01
nvidia-smi | tee results/run-01/gpu.txt
python -m pip install -e .
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda)'
python -m venv /workspace/vllm-venv
/workspace/vllm-venv/bin/python -m pip install --no-cache-dir vllm==0.11.2 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Start this repo's server first. Wait for startup to finish.

```bash
python -m inference_server.cli --host 127.0.0.1 --revision "$REVISION" \
  2>&1 | tee results/run-01/repo.log
```

In the client terminal, keep using the repo's Python environment for both runs.
Set up the output folder and warm up with W2. Check that all 20 requests
completed and that GPU telemetry is present before running the suite.

```bash
backend=repo
backend_args=()
run_workload() {
  python -m inference_server.cli run "$@" --backend "$backend" --timeout 300 "${backend_args[@]}"
}
mkdir -p "results/run-01/$backend"
run_workload W2 > "results/run-01/$backend/warmup.json"
python -c 'import json, sys; r = json.load(open(sys.argv[1])); print(r["summary"]); print(r["gpu_metrics"])' \
  "results/run-01/$backend/warmup.json"
```

Run the same workload block for each backend. W4 uses fixed prompts that
exercise different prefill lengths. W6 runs each offered rate for 30 seconds.

```bash
out="results/run-01/$backend"
short_prompt="Explain continuous batching in 12 numbered points."
long_prompt="$short_prompt Here is some context:"
for i in {1..48}; do
  long_prompt+=" A request waits in a queue until the scheduler has room to process it."
done

run_workload W1 > "$out/W1.json"
run_workload W2 > "$out/W2.json"
run_workload W3 --seed 42 > "$out/W3.json"
run_workload W4 --short-prompt "$short_prompt" --long-prompt "$long_prompt" > "$out/W4.json"
run_workload W5 > "$out/W5.json"
for rps in 1 2 5 10 20; do
  run_workload W6 --rps "$rps" --duration 30 > "$out/W6-${rps}rps.json"
done
```

Stop the repo server with Ctrl+C. In the same server terminal, start vLLM
with the same context, batch and token limits. Disable cross-request prefix
reuse for the shared prompts. Both servers still use a KV cache during generation.
See the [vLLM serve options](https://docs.vllm.ai/en/v0.11.2/cli/serve/).

```bash
/workspace/vllm-venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --revision "$REVISION" --tokenizer-revision "$REVISION" --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --enable-chunked-prefill --no-enable-prefix-caching --generation-config vllm \
  2>&1 | tee results/run-01/vllm.log
```

Once vLLM is ready, switch backends in the client terminal and repeat the
warmup and workload commands above, starting at `mkdir -p`.

```bash
backend=vllm
gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
backend_args=(--model Qwen/Qwen3-4B --gpu-uuid "$gpu_uuid")
```

## Read and plot the results

Use a new run directory for each repeat and copy `results/` off the Pod
before terminating it. [Runpod file transfer](https://docs.runpod.io/pods/storage/transfer-files)

For W1–W5 use side-by-side bars grouped by workload and backend. Group reports by
`run_info.backend`, `run_info.workload`, and `run_info.workload_settings.rps`
for W6. Plot completed requests/s (`summary.requests_per_second`), output
tokens/s (`summary.output_tokens_per_second`), and
`summary.p95_latency_ms` or `summary.p95_client_ttft_ms`. Plot W6 against offered RPS with
`summary.rejected_requests` and `summary.timed_out_requests` alongside latency.

W5 and W6 expose admission policies. This repo rejects when its 20 waiting
slots fill. The vLLM setup can keep more requests waiting. Latency percentiles
use completed requests, so read them together with the success and failure counts.
Check `results` for matching input token counts and compare actual output
lengths. The long W4 prompt should exceed the 512-token prefill budget.
For GPU plots use the common NVML readings in `gpu_metrics`. Short runs can have
missing utilization readings. Queue time and generation TPOT are repo diagnostics.

| ID | Requests |
| --- | --- |
| W1 | One request |
| W2 | 20 simultaneous requests |
| W3 | 20 requests with randomized arrivals over ten seconds |
| W4 | 20 requests with mixed prompt and output lengths |
| W5 | 400 simultaneous requests from 20 users |
| W6 | Sustained arrivals at increasing rates |

## Checks

Run the small checks with the tiny local model:

```bash
python -m unittest discover -s tests -v
python tests/local_smoke.py
```

The local smoke check runs W1–W6 and cancellation over HTTP. On the Pod,
`python tests/gpu_smoke.py` can check model loading and chunked generation
with the server stopped.
