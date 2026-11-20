# Run 03 — profile persistent KV slots

## Question and setup

How much time can I save by keeping each request's KV history in place?

On **2026-11-19**, I ran Qwen3-4B in BF16 on an RTX 3090.
The model and tokenizer revision was `1cfa9a7208912126459214e8b04321603b3df60c`.
I warmed up with W2 and then ran one measured W2 and W4 with `--profile`.
The original cache in Run 02 had packed histories before every forward
and copied them back afterward. Run 03 used four persistent 4,096-token
slots, an 8,192-token reservation budget, and a 512-token scheduling budget
per iteration. Each forward appends new keys and values into the assigned
slot. Attention reads a view of the stored history, and finished or cancelled
requests release their slots for reuse. [Slot cache implementation](../src/inference_server/cache.py)

W2 sends 20 simultaneous short prompts with a 32-token output limit.
W4 sends 20 simultaneous requests split evenly across the four combinations
of 34- or 759-token prompts and 32- or 256-token output limits.

## Results

| Metric | W2, Run 02 → Run 03 | W4, Run 02 → Run 03 |
| --- | ---: | ---: |
| Profiled prefill | 0.52 → 0.31 s | 1.76 → 1.86 s |
| Profiled decode | 14.50 → 7.71 s (−47%) | 59.29 → 44.50 s (−25%) |
| Decode batch setup + cache saving | 3.50 → 0.08 s | 13.84 → 0.50 s |
| Output throughput with profiling | 42.1 → 78.7 tokens/s | 46.7 → 61.4 tokens/s |
| P95 time to first text | 12.44 → 6.64 s | 39.93 → 30.58 s |
| Peak sampled device VRAM | 8.57 → 10.69 GiB | 10.62 → 11.08 GiB |

## Interpretation and decision

The separate cache-saving stage went to zero. Model forward now includes
KV writes and accounted for about **98% of profiled decode time** in both
workloads. Preallocating the slots raised baseline tensor memory by about
2.23 GiB.

Both runs completed every request with matching token counts and finish
reasons. Exact response text matched for 19 of 20 W2 requests and 9 of 20
W4 requests. The runs used different physical RTX 3090s and formed slightly
different batches. I kept the fixed slots and checked throughput with
profiling off in Run 04. Run 05 included a slot-reuse check with fixed inputs
to investigate the text differences.

## Records

Reports and startup configuration are under `results/run-03/`.
