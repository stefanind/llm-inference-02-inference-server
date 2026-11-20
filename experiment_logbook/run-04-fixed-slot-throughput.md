# Run 04 — measure fixed-slot throughput with profiling off

## Question and setup

How does the fixed-slot server perform without timing every stage?

I ran Qwen3-4B in BF16 on the same RTX 3090 as Run 03.
The model and tokenizer revision was `1cfa9a7208912126459214e8b04321603b3df60c`.
The server had four fixed 4,096-token KV slots and an 8,192-token reservation
budget. The earlier Run 01 server packed and copied back the KV history every
step. I warmed up with W2 and measured three W2 and three W4 trials with
profiling off. The exact run date was unrecorded. Every trial completed all
20 requests with the same per-request input and output token counts as Run 01.

W2 sends 20 simultaneous short prompts with a 32-token output limit.
W4 sends 20 simultaneous requests split evenly across the four combinations
of 34- or 759-token prompts and 32- or 256-token output limits.

## Results

| Workload | Run 01 old cache, tokens/s | Run 04 fixed-slot trials, tokens/s | Run 04 median, tokens/s | Observed change |
| --- | ---: | --- | ---: | ---: |
| W2 | 64.30 | 74.87, 69.56, 71.97 | 71.97 | +11.9% |
| W4 | 52.62 | 67.14, 65.25, 60.35 | 65.25 | +24.0% |

## Interpretation and decision

The fixed-slot medians were higher than the old cache measurements in
Run 01. Run 01 used a different physical RTX 3090 and had one trial per
workload, so the percentages describe the observed differences across Pods.
I kept Run 04 as the unprofiled fixed-slot baseline. Run 03 identified the
next measurement target inside `model_forward`.

## Records

All six trials and the warmup are in `results/run-04/baseline/`.
