# Experiment 0002: packed routed gate/up projection

## Hypothesis

The batch-1 routed MoE path is launch/index-bound as well as weight-bandwidth-bound. Concatenating the equally quantized gate and up output rows should replace two expert gathers with one without changing selected weights, routing, or arithmetic.

## Implementation

`src/runnel/mlx.py` lazily constructs one fused quantized switch matrix per layer, performs one `mx.gather_qmm`, splits gate/up halves, applies the existing SwiGLU, and invokes the original routed down projection.

At top-8 batch-1 decode this changes routed gathers from:

- `40 layers × 3 = 120` to `40 layers × 2 = 80` per token.

## Benchmark

Greedy IDs matched the current project reference for every measured 8-bit and 4-bit trial.

| Checkpoint/context | Reference decode | Fused decode | Gain |
|---|---:|---:|---:|
| 8-bit / 1K | 52.61 | 56.53 | +7.44% |
| 8-bit / 8K | 50.09 | 52.51 | +4.84% |
| 4-bit / 1K | 68.45 | 73.15 | +6.86% |
| 4-bit / 8K | 61.94 | 65.40 | +5.58% |

Raw files are named `mlx-lm-current-*-64-greedy.json` and `mlx-lm-current-*-fused-routed-gate-up-64-greedy.json` under `bench/results/m0/`.

## Outcome

**Keep.** The gain is repeatable, exact against the project reference, representation-independent, and implemented in the product path.

It is not a 2× breakthrough. At 8K, prefill and TTFT were effectively unchanged; the benefit is concentrated in decode/launch overhead.

## Why it worked

The optimization does not reduce selected weight bytes. It removes one gather index path, one quantized matrix launch, and one intermediate boundary per layer while preserving each output row's dot-product order. The consistent gain across 8-bit and 4-bit checkpoints shows that dispatch/materialization overhead is material on M1 Max even when weight bandwidth is already significant.
