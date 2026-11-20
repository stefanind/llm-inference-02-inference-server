# Run 02 — find where the old cache path spends time

## Question and setup

Which stage of the original implementation accounts for the most time?

On **2026-11-18**, I ran Qwen3-4B in BF16 on an RTX 3090 with the original
KV cache, which packed request histories before each forward and copied
them back afterward. The model and tokenizer revision was
`1cfa9a7208912126459214e8b04321603b3df60c`. The server ran up to four
active requests with a 512-token scheduling budget per iteration.

I warmed up with W2 and then ran one measured W2 and W4 with `--profile`.
The profile separated prefill from decode and synchronized
CUDA at the boundaries of batch setup, model forward, cache saving, and
token handling. Each measured workload completed all 20 requests.

W2 sends 20 simultaneous short prompts with a 32-token output limit.
W4 sends 20 simultaneous requests split evenly across the four combinations
of 34- or 759-token prompts and 32- or 256-token output limits.

## Results

| Workload | Profiled prefill | Profiled decode | Model forward share of decode | Batch setup + cache saving share of decode |
| --- | ---: | ---: | ---: | ---: |
| W2 | 0.52 s | 14.50 s | 75.4% | 24.2% |
| W4 | 1.76 s | 59.29 s | 76.0% | 23.3% |

## Interpretation and decision

Decode accounted for over 96% of profiled time in both workloads. Setup
and copy-back took roughly a quarter of decode time. The dynamic cache
also grew inside model forward. I chose to preallocate KV slots so a
request could append tokens in place instead of moving its history every
step. The synchronized timings help locate time within the pipeline.

## Records

The warmup, W2 and W4 stage totals, and server log live under
`results/run-02/`.
