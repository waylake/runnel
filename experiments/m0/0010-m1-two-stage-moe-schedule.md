# Experiment 0010: M1 two-stage routed MoE schedule

## Hypothesis

The channel-shard megakernel failed because each SIMD group repeated the full hidden-row loop for a small channel subset. A compact intermediate should make the dependency explicit instead:

```text
gate/up + SwiGLU → [selected_slot, 512] float intermediate
 down projection → [selected_slot, 2048] active rows
 router-weighted combine → [2048] output
```

All three dispatches are encoded in one command buffer; the intermediate is only 16 KiB for eight selected experts.

## Implementation

- `gate_up_swiglu` in `native/shaders/fused_moe.metal` computes four up/gate rows per iteration and writes only the activated intermediate.
- `down_router` maps active output rows to one selected expert and computes four rows per iteration.
- `combine_router` performs the eight-way router-weighted reduction.
- 32 workgroups and 256/512-thread schedules are supported.
- A direct CPU reference is retained separately from the legacy megakernel reference because the Metal reduction order differs.
- The result is currently a numerical baseline, not an exact greedy-token claim.

## Results

Raw records:

- `bench/results/m0/native-m1-two-stage-moe-q4-256.json`
- `bench/results/m0/native-m1-two-stage-moe-q4-512.json`
- `bench/results/m0/native-m1-two-stage-moe-q8-256.json`
- `bench/results/m0/native-m1-two-stage-moe-q8-512.json`

Median GPU times for the synthetic 64-expert bank, eight scattered selected experts, five warmups, and 20 trials:

| Bits | 256 threads | 512 threads |
|---|---:|---:|
| q4 | 279.1 μs | **218.0 μs** |
| q8 | 371.3 μs | **258.1 μs** |

The 512-thread q8 schedule is below the separate optimized QMV estimate and the q4 schedule is close to it. The largest remaining cost is likely the short gate/up and down launches plus the separate combine; the result should not be extrapolated to full inference until real checkpoint tensors and layer repetition are measured.

## Decision

**Keep the two-stage schedule as the leading native MoE candidate.** It fixes the megakernel's repeated-row-loop failure without claiming a resident global barrier. It is suitable for the next step: load real Ornith q4/q8 tensors, measure one layer repeatedly, and then encode a 40-layer command buffer.

Before productization:

1. add real-checkpoint tensor loading and identity capture;
2. compare against pinned MLX per layer;
3. establish greedy-ID parity for the exact lane;
4. profile stage 1, stage 2, and combine separately;
5. only then consider a residency/readiness-based fusion.
