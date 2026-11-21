# Simple inference server

A small GPU inference server for learning how LLM serving works. It runs
Qwen3-4B in BF16 and streams generated text over HTTP. The repo includes a
workload runner for comparing this server with vLLM.

## Quickstart

Use Python 3.11+ and a CUDA GPU with BF16 support. The experiments used an
RTX 3090 with 24 GB of VRAM and the Runpod image
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`.

From the repo directory on the Pod, install and start the server.

```bash
export HF_HOME=/workspace/hf-cache
python -m pip install -e .
python -m inference_server.cli --host 127.0.0.1 \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

In another terminal on the same host, send a streaming request.

```bash
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain KV caching in one sentence.","max_new_tokens":32}'
```

## How it works

- The scheduler admits up to four active requests and holds up to 20 waiting
  requests in a FIFO queue. Each iteration advances decode first, then uses
  the remaining token budget for prompt chunks. [Scheduler](src/inference_server/engine.py)
- Each active request has a fixed KV cache slot. New keys and values are
  appended in place as the request advances. [KV cache](src/inference_server/cache.py)
- Start with `--cuda-graph-decode` to enable replay for four-request decode
  batches that fit the 64-position KV view. [Decode graph](src/inference_server/decode_graph.py)

## Benchmarks

With the server running, send W2, a burst of 20 simultaneous requests.

```bash
mkdir -p results
python -m inference_server.cli run W2 --timeout 300 > results/W2.json
```

In Run 08, median W2 throughput rose from **58.05 to 164.99 tokens/s**
with CUDA Graph replay on the same RTX 3090. [Run 08 results](experiment_logbook/run-08-cuda-graph-throughput.md)

- [Experiment guide](docs/experiments.md) covers W1–W6, profiling, CUDA Graphs, and the vLLM comparison.
- [Experiment logbook](experiment_logbook/) records the setup and results for each run.

## Tests

These checks use a tiny local model and run on CPU.

```bash
python -m unittest discover -s tests -v
python tests/local_smoke.py
```
