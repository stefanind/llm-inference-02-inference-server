# Simple inference server

## Layout

```text
src/inference_server/
  __init__.py
  model.py        # Explicit BF16 GPU loading and sync_gpu()
  generation.py   # Prompt preparation, sampling, generation, and metrics
  server.py       # Request/response types and FastAPI endpoints
tests/
  test_server.py  # The notebook's health and generation checks
benchmark.py     # Local cache comparison and HTTP concurrency benchmark
pyproject.toml   # Dependencies and package installation
PLAN.md          # Later iterations
```

## Runpod image

Use a Runpod GPU Pod with this PyTorch image **PyTorch 2.9.1, CUDA 12.8.1, and Ubuntu 24.04**

Choose one BF16-capable GPU and a host compatible with the image's CUDA version.

Add **8000** to the Pod's exposed HTTP ports. The steps below run in the Pod's
Linux terminal.

## Install

Clone or upload this repo to `/workspace/llm-inference-02-inference-server`, then:

```bash
cd /workspace/llm-inference-02-inference-server
python -m pip install -e . 
```

Check the environment before loading the model:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
```

The server, local benchmark, and GPU smoke test each load their own model; run
them one at a time. The HTTP benchmark uses the running server's model.

## Start the server

```bash
python -m uvicorn inference_server.server:app --host 0.0.0.0 --port 8000 --workers 1
```

Inside the Pod, the API is at `http://127.0.0.1:8000`. From your browser, open
`https://POD_ID-8000.proxy.runpod.net/docs`, replacing `POD_ID` with your Pod's ID.
Binding to `0.0.0.0` makes the server reachable through Runpod's proxy.

Try a request from another terminal in the same Pod:

```bash
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain KV caching","max_new_tokens":32}'
```

The endpoints are `GET /health`, `POST /generate`, and `POST /stream`. To try streaming:

```bash
curl -N http://127.0.0.1:8000/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello","max_new_tokens":16}'
```

## Run the existing checks

These commands run their own model and do not need a running HTTP server:

```bash
python tests/test_server.py
python benchmark.py
```

The smoke test checks health and generation. The benchmark warms up the model,
compares cached and uncached generation, and prints whether their token IDs match.
Both commands need the real model and enough GPU memory to load it.

## Measure concurrent HTTP requests

Start the server in one terminal, then run this in another:

```bash
python benchmark.py --http
```

The script sends eight requests at each concurrency level: 1, 2, and 4. Concurrency
is the maximum number of requests running at once; the total stays at eight.
Every request uses the same prompt, up to 32 output tokens, greedy sampling, and
KV caching. Each experiment starts with a warmup request excluded from its timing.

One JSON summary is printed for each concurrency level:

- `average_latency_ms`: average client time to receive a successful full answer.
- `elapsed_seconds`: time for the whole group, including failed requests.
- `total_output_tokens`: actual output tokens from successful requests.
- `output_tokens_per_second`: total output tokens divided by the group time.
- `successful_requests`, `failed_requests`, and `errors`: what completed or failed.
- `average_generation_ttft_ms`: the server's first-token timing, averaged over
  successful requests. It excludes HTTP overhead and waiting before generation.

The client receives completed answers from `/generate`, so it does not measure
when the first streamed text arrives. If every request fails, average timings
are `null`. A failed warmup stops the benchmark with an error message.

The small functions in `benchmark.py` separate sending, running, summarizing,
and printing. You can change `SERVER_URL`, `REQUEST_COUNT`, or `MAX_NEW_TOKENS`
near the top of the file. Repeat the experiment a few times on the same hardware
and keep the printed summaries for comparison with later iterations.

Unit checks use mocked CUDA calls and a tiny locally created model. They require
no GPU or pretrained weight downloads:

```bash
python -m unittest discover -s tests -v
```

## Existing behavior to revisit later

- `/stream` always uses KV caching, including when `use_kv_cache=false` is supplied.
- The cached and streaming loops run an unused model forward after the last token
  when they reach the output limit without hitting EOS.
