# Simple inference server

## Layout

```text
src/inference_server/
  __init__.py
  model.py        # Model/tokenizer loading, device selection, sync_gpu()
  generation.py   # Prompt preparation, sampling, generation, and metrics
  server.py       # Request/response types and FastAPI endpoints
tests/
  test_server.py  # The notebook's health and generation checks
benchmark.py     # Local cache comparison and HTTP concurrency benchmark
pyproject.toml   # Dependencies and package installation
PLAN.md          # Later iterations
```

## Install

Use Python 3.11 or later. From the repo folder, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

`-e` installs the package in editable mode, so edits inside `src/` take effect
without reinstalling it. Dependencies are listed in `pyproject.toml`.

The loading settings are copied from the notebook: `Qwen/Qwen3-4B`, BF16, and
`device_map="auto"`. Use the same hardware setup that runs your notebook. For CUDA,
make sure your PyTorch installation supports your GPU.

Importing `model.py` loads the model. The first run downloads weights if they are
not already cached. The server, local benchmark, and smoke test each load their
own model; run them one at a time. The HTTP benchmark uses the running server's
model.

## Start the server

```powershell
python -m uvicorn inference_server.server:app --workers 1
```

The API is at `http://127.0.0.1:8000`. Interactive request examples are available
at `http://127.0.0.1:8000/docs`.

Try a request from another PowerShell window:

```powershell
$body = @{ prompt = "Explain KV caching"; max_new_tokens = 32 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/generate -Method Post -ContentType application/json -Body $body
```

The endpoints are `GET /health`, `POST /generate`, and `POST /stream`. To try
streaming, save a request to a file and use curl:

```powershell
'{"prompt":"Say hello","max_new_tokens":16}' | Set-Content -Encoding Ascii request.json
curl.exe -N http://127.0.0.1:8000/stream -H "Content-Type: application/json" --data-binary "@request.json"
```

## Run the existing checks

These commands run their own model and do not need a running HTTP server:

```powershell
python tests/test_server.py
python benchmark.py
```

The smoke test checks health and generation. The benchmark warms up the model,
compares cached and uncached generation, and prints whether their token IDs match.
Both commands need the real model and enough memory to load it.

## Measure concurrent HTTP requests

Start the server in one terminal, then run this in another:

```powershell
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

Benchmark checks that do not load model weights can be run with:

```powershell
python -m unittest discover -s tests -p test_benchmark.py -v
```

## Existing behavior to revisit later

- `/stream` always uses KV caching, including when `use_cache=false` is supplied.
- The cached and streaming loops run an unused model forward after the last token
  when they reach the output limit without hitting EOS.

These are carried over from the notebook. Queuing, scheduling, and batching are
covered by later iterations in [PLAN.md](PLAN.md).
