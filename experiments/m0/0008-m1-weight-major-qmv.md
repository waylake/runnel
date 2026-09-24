# Experiment 0008: M1 weight-major quantized QMV and expert mapping

## Hypothesis

A batch-1 Ornith expert projection is too short-lived to amortize ordinary Metal dispatch and expert-index bookkeeping. A 32-workgroup weight-major QMV should stream selected expert rows directly, with the expert index resolved once per SIMD group rather than once per output row.

## Implementation

- C++20/Objective-C++ Metal host with no runtime framework dependency.
- Affine q4/q8 packed-weight QMV for `K=2048` and `K=512`.
- 32 workgroups, 64–512 threads, and 64–256 synthetic expert banks.
- Non-contiguous expert indices are supported through a SIMD-group-local expert base.
- Four output rows are loaded per inner iteration to expose memory-level parallelism.
- CPU reference validates every output, including non-contiguous expert gathers.
- The q4 packed-word oracle was corrected to use nibble shifts `0,4,…,28`; the earlier q8-style byte pattern was an oracle bug, not a Metal-kernel result.

## Results

### Large contiguous matrices

Raw records:

- `bench/results/m0/native-m1-weight-major-qmv-q4.json`
- `bench/results/m0/native-m1-weight-major-qmv-q8.json`

At `N=65536, K=2048`, the best 512-thread launch reached:

- q4: 174.29 GiB/s
- q8: 202.87 GiB/s

### Actual top-8 shapes

Raw records:

- `bench/results/m0/native-m1-top8-gather-simdmap-gate-up-qmv-q4.json`
- `bench/results/m0/native-m1-top8-gather-simdmap-gate-up-qmv-q8.json`
- `bench/results/m0/native-m1-top8-gather-simdmap-down-qmv-q4.json`
- `bench/results/m0/native-m1-top8-gather-simdmap-down-qmv-q8.json`

For scattered expert indices, SIMD-level mapping reduced gate/up latency from approximately 222 μs to 107 μs in q8. A contiguous-index control was equally slow, proving the original regression was repeated integer division/remainder bookkeeping, not expert scatter.

The optimized routed projection lower bounds are approximately:

- q4: 151 μs gate/up + 95 μs down
- q8: 107 μs gate/up + 116 μs down

These are kernel-only GPU timings for synthetic data, not end-to-end model results.

## Decision

**Keep the weight-major QMV as a component.** It is numerically validated, reaches useful M1 bandwidth, and exposes a concrete dispatch/index win.

**Do not ship it as a one-projection replacement yet.** At `N=512,K=2048`, its single-command wall time was approximately 325–335 μs versus 274–278 μs for pinned MLX `quantized_matmul`; the native advantage only appears when many projections are encoded into a persistent layer schedule.

## Next hypothesis

Fuse gate/up, SwiGLU, and down without repeating channel-shard row loops. The first whole-layer one-kernel attempt is recorded separately as Experiment 0009; its failure motivates a two-stage global-intermediate schedule.
