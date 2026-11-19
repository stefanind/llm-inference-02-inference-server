# Benchmark experiments

## Run 01 — establish the vLLM comparison

We ran W1–W6 to see where the servers differed in throughput, latency, and
overload behavior. The repo used its original KV cache, which packed request
histories into a batch before each forward and copied them back afterward.
Profiling was off.

This server and vLLM 0.11.2 ran Qwen3-4B in BF16 on the same RTX 3090. They
used the same checkpoint, request schedules, four-sequence limit, 512-token
scheduling budget, and 4,096-token context limit. These are results from one
measured run per case.
Every pair is **this repo vs vLLM**.

| Workload | Offered per server | Completed | Completed req/s | Output tokens/s | P95 first text (s) | P95 total latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 · single request | 1 | 1 vs 1 | 0.57 vs 2.33 | 18.1 vs 74.7 | 0.088 vs 0.042 | 1.765 vs 0.428 |
| W2 · simultaneous | 20 | 20 vs 20 | 2.01 vs 9.50 | 64.3 vs 303.9 | 8.095 vs 1.710 | 9.896 vs 2.098 |
| W3 · random arrivals | 20 | 20 vs 20 | 1.62 vs 1.95 | 52.0 vs 62.3 | 2.476 vs 0.039 | 4.409 vs 0.441 |
| W4 · mixed lengths | 20 | 20 vs 20 | 0.37 vs 1.67 | 52.6 vs 240.5 | 36.590 vs 7.828 | 51.569 vs 11.206 |
| W5 · burst | 400 | 24 vs 400 | 1.92 vs 9.51 | 61.6 vs 304.3 | 10.754 vs 39.441 | 12.431 vs 39.840 |
| W6 · 1 req/s | 30 | 30 vs 30 | 0.98 vs 1.02 | 31.4 vs 32.6 | 0.135 vs 0.036 | 1.851 vs 0.416 |
| W6 · 2 req/s | 60 | 60 vs 60 | 1.90 vs 2.01 | 60.9 vs 64.2 | 0.412 vs 0.034 | 2.316 vs 0.415 |
| W6 · 5 req/s | 150 | 80 vs 150 | 1.97 vs 4.96 | 63.1 vs 158.8 | 10.205 vs 0.043 | 12.123 vs 0.446 |
| W6 · 10 req/s | 300 | 80 vs 300 | 1.94 vs 9.15 | 62.2 vs 292.8 | 10.735 vs 2.399 | 12.752 vs 2.811 |
| W6 · 20 req/s | 600 | 78 vs 600 | 1.90 vs 9.25 | 60.7 vs 295.9 | 10.845 vs 32.937 | 12.832 vs 33.347 |

Throughput includes the time to drain queued requests after arrivals stop.
Latency uses completed requests. W1 is a single observation. Every failed
request was an HTTP 429 from the repo. Warmups are excluded.

- **Mixed lengths.** W4 used the same 34- and 759-token prompts and 32- and
  256-token output limits on both servers. Both finished all 20 requests.
  P95 total latency was 51.57 s for the repo and 11.21 s for vLLM.
- **Saturation.** The repo completed about 2 req/s once offered load reached
  5 req/s. vLLM reached about 9.2 completed req/s at the two highest offered
  rates and eventually cleared its backlog. At 20 offered req/s, the repo
  rejected 522 of 600 requests. Its lower P95 at that rate describes the 78
  requests that finished.
- **GPU memory.** The highest sampled device VRAM was 10.7 GiB for the repo
  and 21.4 GiB for vLLM. vLLM reserved about 12.9 GiB for its KV cache at
  startup. The sampled values include those reservations.

W2 was the clearest throughput comparison. Both servers finished all 20
requests with matching token counts, and vLLM handled **4.7× as many requests
per second**. W5 and W6 showed how differently the servers admitted work under
load. We chose W2 and W4 for profiling because both completed every request
and let us compare short, concurrent requests with mixed prompt and output
lengths.

## Run 02 — find where the old cache path spends time

We profiled the repo's original implementation on W2 and W4. The profile
separated prefill from decode and timed batch setup, model forward, cache
saving, and token handling for each phase.

| Workload | Profiled prefill | Profiled decode | Model forward share of decode | Batch setup + cache saving share of decode |
| --- | ---: | ---: | ---: | ---: |
| W2 | 0.52 s | 14.50 s | 75.4% | 24.2% |
| W4 | 1.76 s | 59.29 s | 76.0% | 23.3% |

Decode accounted for over 96% of profiled time in these workloads. The
setup and copy-back stages took roughly a quarter of decode time, and
the dynamic cache also grew inside model forward. That made repeated
movement of KV history a concrete cost to target.

## Run 03 — profile persistent KV slots

We changed the server to preallocate KV slots for active requests. Each model
forward appends only the new keys and values into the assigned slot. Attention
receives a view of the stored history, and finished or cancelled requests free
their slots for reuse. With the defaults, the server allocates four 4,096-token
slots and caps active reservations at 8,192 tokens.

| Metric | W2, Run 02 → Run 03 | W4, Run 02 → Run 03 |
| --- | ---: | ---: |
| Profiled prefill | 0.52 → 0.31 s | 1.76 → 1.86 s |
| Profiled decode | 14.50 → 7.71 s (−47%) | 59.29 → 44.50 s (−25%) |
| Decode batch setup + cache saving | 3.50 → 0.08 s | 13.84 → 0.50 s |
| Output throughput with profiling | 42.1 → 78.7 tokens/s | 46.7 → 61.4 tokens/s |
| P95 time to first text | 12.44 → 6.64 s | 39.93 → 30.58 s |
| Peak sampled device VRAM | 8.57 → 10.69 GiB | 10.62 → 11.08 GiB |

The separate cache-saving stage went to zero. KV writes now happen inside
model forward, which accounted for about **98% of profiled decode time** in
both workloads. Preallocating the slots raised baseline tensor memory by
about 2.23 GiB.

Both runs completed every request with matching token counts and finish
reasons. Exact response text matched for 19 of 20 W2 requests and 9 of 20
W4 requests. The runs used different physical RTX 3090s and formed slightly
different batches. A fixed-input logit check around slot reuse would help
determine what caused the text differences.

## Run 04 — measure fixed-slot throughput with profiling off

Run 03 showed where time went with profiling enabled. In Run 04 we turned it
off and measured three W2 and three W4 trials on the same GPU as Run 03.
Every trial completed all 20 requests with the same per-request input and
output token counts as Run 01.

| Workload | Run 01 old cache, tokens/s | Run 04 fixed-slot trials, tokens/s | Run 04 median, tokens/s | Observed change |
| --- | ---: | --- | ---: | ---: |
| W2 | 64.30 | 74.87, 69.56, 71.97 | 71.97 | +11.9% |
| W4 | 52.62 | 67.14, 65.25, 60.35 | 65.25 | +24.0% |

The reports in `results/run-04/baseline/` record the fixed-slot implementation.
Run 01 is the available unprofiled
old-cache reference. It came from a different physical RTX 3090 and has one
measured trial per workload, so the percentages above are observed differences
across Pods. A comparison of both revisions on one Pod would isolate the
cache change more closely.

The next diagnostic is a short trace inside model forward. It can show how
much of the remaining decode time goes to cache writes, attention, matrix
multiplication, and launching GPU work.
