# Experiment 0004: MLX core chunk-8 Gated DeltaNet

## Hypothesis

MLX core commit `59d600b` provides a chunk-parallel M1 GDN kernel that reduces long-prefill latency relative to the sequential kernel in mlx-lm 0.31.3.

## Implementation

Built MLX `0.32.3.dev20260924+59d600b` from source. For sequences of at least 16 tokens, the adapter computes the existing gamma/beta values and calls `mx.fast.gated_delta_update`; M1 automatically selects the SIMD-group chunk-8 path. Short decode retains the original kernel.

## Correctness

Synthetic BF16 comparison showed finite outputs but nontrivial numerical differences from the old kernel. End-to-end greedy IDs diverged:

- 1K: first differed token index 24 versus current reference;
- 8K: first differed token index 9.

The path therefore does not satisfy the project's exact greedy-ID gate.

## Performance

- 8K prefill: about 797 tok/s versus about 756 tok/s for the current reference, roughly +5.6%.
- 8K TTFT: 10.414 s versus 10.768 s, roughly 3.3% lower.
- decode was effectively unchanged.

Raw files: `mlx-nightly-gdn-core-chunked-{1k,8k}-64-greedy.json`.

## Decision

Do not admit to the exact runtime. Keep the source-build manifest and adapter switch as a numerical research result. A future exact version needs a reduction/order contract that matches the project reference on M1, not only a mathematical equivalence claim.
