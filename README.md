# Simple inference server

A learning project for inference scheduling: HTTP requests enter a FIFO queue,
then one worker runs inference. The queue holds 20 waiting requests plus one
active request; excess requests receive HTTP 429.

## Run locally

Simulation needs no GPU. From the repo folder:

```bash
python -m pip install -e .
python -m inference_server.cli --delay 0.1
```

In another terminal, run a workload:

```bash
python -m inference_server.cli run W1
python -m inference_server.cli run W2
python -m inference_server.cli run W5 > results.json
```

Reports contain individual responses and a summary of throughput, latency,
errors, and queue waiting. Simulation leaves token measurements unavailable.

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
for the remaining options. Keep the same delay when comparing experiments;
with a two-second delay, W2 takes about 40 seconds with one worker.

The queue serves `/generate`. `/stream` currently bypasses it; run streaming
separately from queued experiments.

## Real inference on Runpod

Use the PyTorch **2.9.1 / CUDA 12.8.1 / Ubuntu 24.04** image, a BF16-capable GPU,
and expose HTTP port **8000**.

```bash
python -m pip install -e ".[gpu]"
python -m inference_server.cli --mode real --host 0.0.0.0 --port 8000
```

The runner works with either inference mode. API docs are at `/docs` on the
server's local address or Runpod HTTP proxy URL.

## Checks

Run the automatic tests without a GPU:

```bash
python -m unittest discover -s tests -v
```

On a GPU, with the server stopped, check real model loading and generation:

```bash
python tests/gpu_smoke.py
```
