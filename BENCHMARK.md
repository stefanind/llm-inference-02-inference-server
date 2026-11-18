# Run 01 benchmark results

vLLM completed about **4.7× as many requests per second in W2**, where both
servers finished all 20 requests. At higher loads, the repo rejected excess
requests while vLLM kept them queued. The repo used less device memory.

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
