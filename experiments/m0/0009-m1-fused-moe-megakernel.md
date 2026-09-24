# Experiment 0009: M1 fused MoE megakernel

## Hypothesis

A routed MoE layer can avoid the gate/up → activation → down intermediate round trip by assigning each SIMD group a complete intermediate-channel shard, computing its gate/up values in threadgroup memory, and immediately accumulating its down-projection partials.

## Implementation

The first implementation used one 32-workgroup launch:

1. eight selected experts × 32 SIMD groups;
2. each SIMD group computed a channel shard of gate/up;
3. SwiGLU was evaluated in threadgroup memory;
4. each SIMD group computed down partials;
5. a lightweight combine kernel reduced the partials.

The kernel is in `native/shaders/fused_moe.metal` and its harness is `native/src/fused_moe_probe.mm`.

## Result

The kernel is finite and numerically close to the CPU reference after correcting:

- workgroup-local versus global SIMD storage;
- partial-buffer indexing by selected slot rather than absolute expert ID;
- affine q4 scale-group indexing for channel shards;
- reduction over all hidden rows for each channel shard.

However, the one-kernel schedule is slow:

- q4, 32 workgroups × 256 threads: approximately **1.07 ms GPU** for the synthetic routed layer;
- q8, same geometry: approximately **0.86 ms GPU**.

The separate SIMD-mapped QMV projections were approximately 0.25–0.36 ms for the same routed weights. The fused version repeats the hidden-row loop once per channel shard, creating excessive row/scale/index overhead. The combine kernel is not the cause; the main expert kernel dominates.

Raw records:

- `bench/results/m0/native-m1-fused-moe-q4.json`
- `bench/results/m0/native-m1-fused-moe-q8.json`

The first generated q4 fused raw files had a harness metadata bug that counted packed q4 bytes as if they were two bytes per element. The files above were regenerated after correcting the byte-count formula; the performance distributions were not used to hide or select a result.
The reduction order is also different from the direct reference, so this is a numerical baseline, not an exact claim.

## Decision

**Kill the one-launch channel-shard megakernel as a product path.** Keep the harness and negative evidence.

The failure is informative: a channel shard is a good way to avoid duplicate gate/up work, but it is a bad way to partition down rows. The next design will use two stages:

```text
Stage A: gate/up + SwiGLU → compact [selected_expert, 512] intermediate
Stage B: down + router-weighted combine
```

Both stages will be encoded into one command buffer and can later be replaced by readiness flags only after Metal residency is proven safe.
