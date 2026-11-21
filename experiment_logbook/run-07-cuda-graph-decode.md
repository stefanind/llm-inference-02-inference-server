# Run 07 — replay a decode forward with a CUDA Graph

## Question and setup

How much time can I save by replaying the whole decode forward instead of
launching each operation from Python?

I ran Qwen3-4B in BF16 on an RTX 3090 with model and tokenizer revision
`1cfa9a7208912126459214e8b04321603b3df60c`. The diagnostic used four W2
requests with prompt lengths of 30, 30, 31, and 31 tokens. Eager and replay
used independent KV caches with the same fixed 64-position view. I kept
RMSNorm compilation and profiling off. [Run command](../docs/experiments.md#compare-cuda-graph-decode)

After four eager warmup steps per cache, I compared logits for 31 forced
decode steps and then timed four trials of those steps per mode. Timing
included copying the staged token IDs and positions into the input buffers,
updating the mask, and waiting for each forward to finish.

## Results

| Decode measurement | Eager | CUDA Graph |
| --- | ---: | ---: |
| Trial means (ms/step) | 44.20, 44.06, 42.39, 41.13 | 15.76, 15.80, 15.79, 15.78 |
| Median of trial means (ms/step) | 43.22 | 15.78 |

Replay was **2.74× faster**. The next-token prediction matched in all
**124 comparisons** across the four requests and 31 steps. The maximum
absolute logit difference was **0.0**.

## Interpretation and decision

The two paths used the same fixed shapes, so this experiment isolates the
benefit of submitting the forward as a graph. I used the result to justify
testing replay in the streaming server on W2. That comparison includes
prefill, sampling, request queueing, and HTTP streaming. [Run 08](run-08-cuda-graph-throughput.md)

## Records

The timings and logit check are in `results/run-07/decode-graph/cuda-graph.json`.
The same directory contains `run.log`. The device snapshot is in
`results/run-07/gpu.txt`.
