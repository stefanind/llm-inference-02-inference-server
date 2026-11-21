# Run 08 — measure CUDA Graph replay over HTTP

## Question and setup

How much of the faster decode from a CUDA Graph carries through to end-to-end
W2 throughput?

I ran both server modes on the same RTX 3090 with Qwen3-4B in BF16 and model
and tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`. Both
used four active slots, a 512-token scheduling budget, and a 4,096-token
context limit. I kept profiling and RMSNorm compilation off. The graph
server replayed four-request decode batches that fit a 64-position KV view.
Other batches used eager execution. [Run commands](../docs/experiments.md#measure-cuda-graph-serving)

W2 sends 20 simultaneous short prompts with a 32-token output limit. I ran
one warmup and three measured trials per mode. The reports record matching
GPU UUIDs, package versions, and W2 schedules for both modes.

## Results

Every measured trial completed **20 of 20 requests** and produced **640
output tokens**. Latency and throughput medians below are taken across the
three trial summaries in each mode.

| W2 measurement | Eager | CUDA Graph |
| --- | ---: | ---: |
| Output tokens/s by trial | 48.38, 58.05, 58.47 | 165.42, 163.70, 164.99 |
| Median output tokens/s | 58.05 | 164.99 |
| Median completed requests/s | 1.81 | 5.16 |
| Median mean generation time per output token (ms) | 66.46 | 19.87 |
| Median P95 time to first text (s) | 8.93 | 3.27 |
| Median P95 total latency (s) | 10.97 | 3.87 |
| Decode forwards that used replay, per trial | 0 of 160 | 150 of 160 (93.75%) |

Input and output token counts and finish reasons matched for every request.
Exact response text matched for **17, 17, and 18 of 20 requests** across
the three numbered trial pairs. The graph trials matched each other for
all 20 requests. One eager trial differed from the first eager trial on
one request.

## Interpretation and decision

CUDA Graph replay raised median HTTP throughput by **2.84×** and lowered
generation time per output token by **70.1%**. The graph handled most of the
decode forwards on this short-prompt workload. The graph path uses a fixed
attention width, and the eager path uses the current history width. The
text differences leave open how much of the variation comes from that shape
change. I kept replay opt-in while measuring this four-request path.

## Records

The warmups and three measured reports per mode are in `results/run-08/eager/`
and `results/run-08/graph/`. Each report includes the server flag, replay
count, and per-request results.
