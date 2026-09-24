# Experiment 0007: M1 persistent token-capsule feasibility

## Hypothesis

A single 1,024-thread Metal workgroup may be able to keep the M1 Max memory pipeline busy enough to execute all 40 Ornith decoder layers inside one persistent token kernel. If true, a native C++ runtime can remove most per-operator command-buffer boundaries without first building a model-specific qmv kernel.

The expected bottleneck is memory-pipeline underutilization: a 1,024-thread workgroup has 32 SIMD groups, but a compute kernel dispatches many short-lived workgroups so that independent groups can overlap memory stalls.

## Pre-registered decision gate

The first experiment streams a 4 GiB resident buffer and computes a checksum over every 16-byte vector load. Each launch partitions disjoint buffer ranges, so the checksum also proves complete vector coverage.

- If the best one-workgroup launch reaches at least 70% of the best many-workgroup bandwidth, continue to a quantized QMV kernel and an M1 single-workgroup whole-token prototype.
- If it reaches 85% or more, prioritize the single-workgroup whole-token design; use ordinary multi-workgroup kernels only for prefill.
- If it reaches less than 50%, kill the single-workgroup whole-token design immediately.
- At 50–70%, use a small persistent workgroup set only if in-kernel residency and synchronization are demonstrably safe; otherwise use per-layer persistent MoE kernels.
- Raw streaming above 200 GB/s is necessary for a plausible 120 tok/s 4-bit weight-stream floor, and above 280 GB/s for a 2× 8-bit weight-stream target. These are feasibility thresholds, not promised end-to-end results.

The primary comparison is full trial distributions, not best-of-N.

## Implementation

- C++20 host and public interface under `native/`.
- Minimal Objective-C++ Metal bridge; no MLX, PyTorch, Core ML, or framework runtime dependency.
- Hand-written Metal kernel with eight independent 16-byte loads per loop iteration.
- 64–1,024 threads per workgroup and 1–32 workgroups on the verified 32-core M1 Max.
- GPU timestamps and wall-clock timing, two warmups, seven measured trials by default.
- Preallocated and first-touched shared buffer; no allocation or page fault occurs inside a measured launch.

## Prior-art boundary

The broad native-Metal and persistent-MoE directions are not claimed as novel here. This experiment tests a narrower M1-specific question: whether one maximum-width workgroup has enough memory-level parallelism for a whole-token persistent capsule, without relying on CUDA cooperative groups or unsupported cross-workgroup barriers.

Relevant public work inspected before implementation:

- BaseRT, native C++/Metal runtime and M1-family design context: arXiv:2607.00501.
- MonoMoE, weight-major persistent quantized MoE decode on H200: arXiv:2609.04244.
- MLX's existing affine QMV, wide QMV, quad QMV, and gather-QMM tile families.
- ZMLX and Core AI fused-MoE reports as evidence that fusion/dispatch removal can matter on Apple Silicon.

## Result

Raw record: `bench/results/m0/native-m1-persistent-stream-4gib.json`.

All launch geometries produced the same non-zero checksum over every 16-byte vector. Median GPU timings from seven trials were:

| Workgroups | Threads/workgroup | Median bandwidth |
|---:|---:|---:|
| 1 | 256 | 31.075 GiB/s |
| 1 | 1,024 | 30.953 GiB/s |
| 4 | 256 | 131.013 GiB/s |
| 8 | 256 | 162.517 GiB/s |
| 16 | 256 | 282.983 GiB/s |
| 32 | 256 | **370.341 GiB/s** |
| 32 | 1,024 | 350.309 GiB/s |

The best launch sustained approximately 397.7 GB/s in decimal units, close to the M1 Max's nominal 400 GB/s. The best one-workgroup launch reached only 8.36% of that bandwidth.

## Decision

**Kill the single-workgroup whole-token design.** It failed the pre-registered `<50%` gate by a wide margin and cannot stream Ornith's active weights at a useful rate.

**Continue with 32 persistent workgroups × 256 threads.** This geometry matches the 32 GPU cores and is the first native-runtime schedule to saturate measured memory bandwidth. The next implementation will not assume a whole-model global barrier because Metal does not guarantee cross-workgroup residency. It will begin with per-layer persistent MoE kernels and readiness-based handoff, then use the same 32-group schedule in the heterogeneous 4D prefill scheduler.
