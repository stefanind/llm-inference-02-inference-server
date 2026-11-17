# Simple inference server

A learning project for inference scheduling, with **four active slots** and a
FIFO queue for 20 waiting requests. Excess requests receive HTTP 429.
Fixed batching finishes a whole batch before taking more work. Continuous
scheduling finishes requests individually and refills free slots between steps.

## Run locally

Simulation needs no GPU. From the repo folder:

```bash
python -m pip install -e .
python -m inference_server.cli --scheduler continuous --step-delay 0.01
```

In another terminal, run a workload:

```bash
python -m inference_server.cli run W1
python -m inference_server.cli run W2
python -m inference_server.cli run W5 > results.json
```

Reports contain individual responses and a summary of throughput, latency,
errors, and queue waiting. Continuous simulation advances each active request
one dummy step at a time; `max_new_tokens` sets its step count. Prompt lengths
do not affect step cost. Results include `simulated_steps`, but real token
measurements stay unavailable. These runs check scheduling behavior.

For fixed batching, use `--scheduler fixed --delay 0.1` (fixed is the default).
`--delay` controls one fixed batch; `--step-delay` controls one continuous step.

## Workloads

| ID | Requests |
| --- | --- |
| W1 | One request: baseline |
| W2 | 20 simultaneous requests |
| W3 | 20 requests with randomized arrivals over ten seconds |
| W4 | 20 requests with mixed prompt and output lengths |
| W5 | 20 users submitting 20 requests each: a 400-request burst |
| W6 | Sustained arrivals at a chosen requests-per-second rate |

W4 requires `--short-prompt` and `--long-prompt`. Run each W6 rate separately:

```bash
python -m inference_server.cli run W6 --rps 2 --duration 30
```

Use `--url` for another server. See `python -m inference_server.cli run --help`
for the remaining options. `--batch-size` limits batch size or active slots.
The initial collection window is 10 ms (`--batch-wait-ms`); a full set starts
immediately. Continuous refills never wait for this window while work is active.
`queue_wait_ms` ends at the request's first step; `batch_size` records that
step's membership, which can change later.

The queue serves `/generate`. `/stream` currently bypasses it; run streaming
separately from queued experiments.

## Real inference on Runpod

Use the PyTorch **2.9.1 / CUDA 12.8.1 / Ubuntu 24.04** image, a BF16-capable GPU,
and expose HTTP port **8000**.

```bash
python -m pip install -e ".[gpu]"
python -m inference_server.cli --mode real --scheduler continuous --host 0.0.0.0 --port 8000
```

The runner works with either inference mode. API docs are at `/docs` on the
server's local address or Runpod HTTP proxy URL.
Both schedulers support real inference and keep each request's output limits
and sampling settings. Continuous mode currently recomputes prefixes without
KV caching and reports `used_kv_cache=false`. Fixed mode uses caching unless
a batch member disables it. Streaming still uses its separate path.

## Checks

Run the automatic tests without a GPU:

```bash
python -m unittest discover -s tests -v
```

On a GPU, with the server stopped, check real model loading and generation:

```bash
python tests/gpu_smoke.py
```
