# Run 06 — compile the decoder-layer RMSNorms

## Question and setup

Can fusing small operations inside RMSNorm reduce decode latency and
improve W2 throughput?

On **2026-11-19**, I ran the fixed-slot server with Qwen3-4B in BF16 on an
RTX 3090. The model and tokenizer revision was
`1cfa9a7208912126459214e8b04321603b3df60c`. I used `--compile-decode`
for the two hidden-state RMSNorms in each of Qwen3's 36 layers.

Both traces captured eight decode
steps after four warmup decode steps on identical four-request batches
with prompt lengths of 30, 30, 31, and 31 tokens. The logit check compared
the fixed-slot path against single-request Transformers after prefill and
for four forced decode tokens on two requests. The HTTP measurement used
W2 with profiling off, sending 20 simultaneous short prompts with a
32-token output limit. [Compilation entry point](../src/inference_server/decode_compile.py)

## Results

| Measurement across eight steps | Eager | Compiled |
| --- | ---: | ---: |
| GPU kernel launches | 18,584 | 14,552 |
| Time covered by GPU kernels, copies, and memory sets | 130.77 ms | 122.86 ms |
| Profiler self CPU time for CUDA launch calls | 128.91 ms | 108.56 ms |
| Elapsed time inside model forward | 586.57 ms | 627.44 ms |
| Elapsed time in decode steps | 602.97 ms | 644.56 ms |

Matching the **576 affected RMSNorm calls** by position in the model showed
where part of the regression came from. The CPU interval below spans the
norm and its dispatch, from the end of the preceding operator to the start
of the next linear projection.

| Average per affected RMSNorm | Eager | Compiled |
| --- | ---: | ---: |
| GPU kernels | 8 | 1 |
| Time executing those GPU kernels | 16.6 µs | 2.3 µs |
| Elapsed CPU time between the surrounding operators | 210.5 µs | 245.2 µs |

The logit check matched the reference's top token in 9 of 10 eager cases and
10 of 10 compiled cases. The eager mismatch was the same near tie in the
reused slot as in Run 05. Across the eight forced decode comparisons, mean
absolute logit differences averaged 0.072 eager and 0.081 compiled.

The measured compiled W2 produced 640 tokens at **63.92 tokens/s**, about
**11% below** the Run 04 eager median of **71.97 tokens/s**. Both runs
completed all 20 requests with the same W2 schedule, checkpoint, and server
limits. The warmup and measured HTTP report are stored under
`results/run-06/eager/`, and both record `compile_decode: true`. For HTTP
throughput, the eager comparison comes from Run 04 on another physical GPU.

## Interpretation and decision

Kernel launches fell by **22%**, and traced decode steps took **6.9% longer**.
The runtime reused the same compiled graph and spent 18.34 ms checking the
cache across 576 calls. At 72 norms per step, the longer CPU intervals around
the norms account for **2.5 ms of the 5.2 ms slowdown per step**. The other
time was spread across surrounding operators and gaps. I kept eager as
the default. A paired HTTP run on one Pod would isolate the throughput
effect more closely if I revisit this approach.

## Records

The paired timelines, operator tables, and logit comparisons are in
`results/run-06/eager/trace/` and `results/run-06/compiled/trace/`. The trace
metadata records the compile flag for each capture.
