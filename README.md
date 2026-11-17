# Simple inference server

A GPU inference server for learning continuous batching, with
up to four active requests, and a FIFO queue for 20 waiting requests.
Excess requests receive HTTP 429.

## Run on Runpod

Use the PyTorch **2.9.1 / CUDA 12.8.1 / Ubuntu 24.04** image, a BF16-capable GPU,
and expose HTTP port **8000**. From the repo folder:

```bash
python -m pip install -e .
python -m inference_server.cli
```

The server loads one BF16 model on `cuda:0` and listens on `0.0.0.0:8000`.
Add `--revision <commit>` to pin the model and tokenizer to the same checkpoint.
API docs are at `/docs` through the Pod's HTTP proxy.
`POST /generate` always returns SSE: `text` events, then `done` with token counts
and timings. Generation failures send an `error` event. Disconnecting cancels
the request; cache reservations are released at the next scheduling boundary.

## Run workloads

In another terminal, or with `--url` pointing at the Pod's HTTP proxy:

```bash
python -m inference_server.cli run W1
python -m inference_server.cli run W2
python -m inference_server.cli run W5 > results.json
python -m inference_server.cli run W6 --rps 2 --duration 30
python -m inference_server.cli run W2 --backend vllm --model Qwen/Qwen3-4B
```

| ID | Requests |
| --- | --- |
| W1 | One request: baseline |
| W2 | 20 simultaneous requests |
| W3 | 20 requests with randomized arrivals over ten seconds |
| W4 | 20 requests with mixed prompt and output lengths |
| W5 | 20 users submitting 20 requests each: a 400-request burst |
| W6 | Sustained arrivals at a chosen rate; run rates separately |

Start vLLM separately in its own environment. Use `--url` with its base address
(without `/v1`). Match the model revision, chat template, BF16, and stopping rules;
use vLLM's `--generation-config vllm` to avoid loading model generation defaults.
Both runner backends request greedy generation with thinking disabled.

W4 requires `--short-prompt` and `--long-prompt`. Reports save the exact schedule,
client settings, responses, throughput, latency, errors, and token counts.
Both backends stream. Reports measure client TTFT and gaps between text chunks;
one chunk can contain multiple tokens. Truncated streams count as failures.
Repo reports include observed queue/active/cache peaks, prefill/decode batch
sizes, generation TPOT, and server configuration from `/info`. Each run resets
counters while idle, so use a dedicated server. Metadata/reset collection is
outside client timings. `/metrics` exposes the current counters.
vLLM engine diagnostics stay `null`; `--server-info file.json` records supplied
server settings and versions, labelled as provided rather than verified.
Reports also include observed client in-flight peaks. For GPU sampling, run the
client on the GPU host: repo runs use `/info`'s UUID; vLLM needs `--gpu-uuid GPU-...`
or a UUID in its supplied server info. NVML samples utilization and device VRAM
every 200 ms, recording counts and errors. Missing readings stay `null`; short
runs may have no utilization samples. Sampled VRAM peaks can miss brief spikes;
the repo separately records its PyTorch tensor-memory baseline and exact peak.
Use `python -m inference_server.cli run --help` for client options.

`--batch-size` sets the maximum active requests. `--batch-wait-ms` sets the initial
collection window (default: 10 ms). Continuous scheduling refills slots between
token steps and always keeps a KV cache per request.
`--max-batched-tokens` limits real input tokens per iteration (default: 512,
at least `--batch-size`). Decode uses one token per request first; prompt chunks
use the remaining budget. Padding and cached history do not count toward it.
`--max-context-tokens` caps prompt plus requested output tokens (default: 4096,
also capped by the model). `--max-cache-tokens` sets a shared reservation budget
(default: 8192). Requests reserve their full prompt plus output limit before
activation; FIFO requests wait for space. Requests that can never fit receive
HTTP 422. This is a logical token budget; allow GPU headroom for padding,
activations, and temporary cache copies.
The cache implementation supports full-attention models such as Qwen3.

## Checks

Run the small checks, including generation with a tiny local model:

```bash
python -m unittest discover -s tests -v
python tests/local_smoke.py
```

The local smoke check runs W1-W6 and cancellation over HTTP with a tiny test
model, without GPU or downloads. Its timings are not GPU performance results.
On the GPU Pod, with the server stopped, check model loading and generation:

```bash
python tests/gpu_smoke.py
```
