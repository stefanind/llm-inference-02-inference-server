# Run 05 — look inside model forward

## Question and setup

Where does decode spend time inside `model_forward`, and how closely do
fixed-slot logits match plain Transformers after a slot is reused?

On **2026-11-19**, I traced the fixed-slot server with Qwen3-4B in BF16
on an RTX 3090. The model and tokenizer revision was
`1cfa9a7208912126459214e8b04321603b3df60c`.
The profiler captured eight decode steps after four warmup decode steps
on four W2 requests with prompts of 30, 30, 31, and 31 tokens. The diagnostic
compared fixed-slot logits with single-request Transformers after prefill
and for four forced decode tokens on two requests. [Decode profiler](../scripts/profile_decode.py)

## Results

| Trace measurement across eight steps | Observed |
| --- | ---: |
| Elapsed time in the decode steps | 631.65 ms |
| Elapsed time inside model forward | 615.31 ms |
| Time covered by GPU kernels, copies, and memory sets | 130.73 ms |
| GPU kernel launches recorded | 18,584 |
| Matrix multiplication time on the GPU | 90.55 ms |
| Attention kernel time on the GPU | 4.90 ms |
| KV-write operations' GPU time | 4.76 ms |

The GPU was doing recorded work for about **21% of the captured decode
windows**, with about **2,323 kernels per step**. CUDA kernel launch calls
took about **131 ms on the CPU** across the capture. Matrix multiplication
was the largest GPU cost. The GPU activity total uses the time covered by
kernel, copy, and memory-set events. The profiler's scope annotations can
overlap and include idle gaps.

The logit check matched the highest-scoring token in **9 of 10** comparisons.
The one mismatch came in the reused slot at forced decode step 4. The
reference's top two scores were only 0.125 apart. [Slot-reuse check](../scripts/profile_decode.py#L60)

## Interpretation and decision

The gaps between kernels pointed to CPU dispatch as a place to investigate.
The trace contained 1,160 calls each to `aten::pow` and `aten::mean`, both
used in RMSNorm. I chose the decoder-layer RMSNorms as a small region to
try fusing in Run 06. The slot-reuse mismatch also remained open. A fresh
slot and a reused slot with identical batch shapes would isolate that check
more directly.

## Records

The timeline, operator tables, and logit comparisons are in
`results/run-05/decode-trace/`.
