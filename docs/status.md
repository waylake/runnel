# Status

_Last updated: 2026-09-24_

## Latest verified result

Runnel now has a working streaming MLX execution path and one exact model-specific optimization: packed routed gate/up projection. This is an alpha research result, not a stable release or the requested 2× breakthrough.

On the 32-core M1 Max 64 GB, current `mlx-lm` project reference versus the Runnel transform, batch 1, greedy, 64 output tokens:

| Checkpoint/context | Reference decode | Runnel decode | Gain | Greedy IDs |
|---|---:|---:|---:|---|
| 8-bit / 1K | 52.61 tok/s | 56.53 tok/s | +7.44% | matched |
| 8-bit / 8K | 50.09 tok/s | 52.51 tok/s | +4.84% | matched |
| official 4-bit / 1K | 68.45 tok/s | 73.15 tok/s | +6.86% | matched |
| official 4-bit / 8K | 61.94 tok/s | 65.40 tok/s | +5.58% | matched |

This is a useful exact optimization, not the requested 2× breakthrough. Raw trials are under `bench/results/m0/`.

## Fastest observed external baseline

Signed oMLX 0.7.0.dev4 reached 75.18 tok/s on the official 4-bit checkpoint at 1K and 65.44 tok/s at 8K. Its greedy output differs from the project reference, so it is a numerical/performance baseline, not the exact lane. At 8K its measured TTFT used 4,192 cached prompt tokens; the cold server TTFT was 12.53 s.

## Verified target environment

- MacBook Pro `MacBookPro18,2`, Apple M1 Max, **32 GPU cores**, 64 GB unified memory.
- macOS 27.2 (`26B5086k`), Xcode 26.6, Metal compiler 32023.883, AppleClang 21.
- MLX 0.32.2; project-reference `mlx-lm` commit `15bcf8b` (`mlx-lm` 0.32.0).
- Historical release baseline: `mlx-lm` 0.31.3.
- Signed oMLX app 0.7.0.dev4, app build 2761, MLX 0.32.2, mlx-lm 0.32.0.
- llama.cpp 0.2.0 build 10566 is installed.

## Verified checkpoints

- 8-bit: revision `02440c39bdf7365c494a7f55f2a8b104ba87562f`, manifest `4ca4e1a78b0524998f26c5039ca1882a0ed6f5f738011aa345a3b8a521a9e026`.
- official 4-bit: revision `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`, manifest `7860835dd74e4ad17a4009421b2c9f5b513450f2f07882763928733a63ae567b`.

Both have 40 layers (30 GDN + 10 full attention), 256 routed experts with top-8 plus one shared expert, hidden size 2,048, and 248,320 vocabulary entries. The config declares one MTP layer, but neither MLX checkpoint contains an MTP tensor.

The header-derived 8-bit active-weight estimate is 3.132 GB/token, giving a 127.73 tok/s 400 GB/s weight-only ceiling. The 4-bit estimate is 1.670 GB/token, giving 239.58 tok/s. Actual rates are far below these optimistic ceilings.

## Current hypothesis

Packed routed gate/up proved that M1 batch-1 MoE loses enough time in repeated gather/index/dispatch work to justify a custom execution path. The next likely decode gains are:

1. an M1-tuned fused quantized expert qmv or gate/up/activation/down pipeline that reduces command and intermediate overhead further;
2. exact complete GDN+KV prefix checkpoints for repeated coding-agent turns;
3. direct elimination of one token-level host synchronization after profiling confirms sampling is on the critical path.

MLX core's chunk-8 GDN improved 8K prefill by about 5.6% but changed greedy IDs, so it is not in the exact runtime.

## Native C++/Metal track

A native feasibility track is now active. A 4 GiB streaming probe killed the single-workgroup whole-token idea: 1 workgroup × 1,024 threads reached 30.95 GiB/s versus 370.34 GiB/s (about 397.7 GB/s) for 32 workgroups × 256 threads.

A 32-workgroup weight-major affine QMV is numerically validated for q4/q8. Resolving the selected expert once per SIMD group instead of once per output row reduced a scattered top-8 q8 gate/up projection from about 222 μs to 107 μs. The first whole-layer channel-shard megakernel was rejected: it repeated hidden-row loops per channel shard and took about 1.07 ms for the synthetic q4 routed layer.

The current native candidate is a two-stage command-buffer schedule: fused gate/up + SwiGLU into a compact selected-expert intermediate, followed by down projection and router-weighted combine. It now passes real Ornith layer-0 tensors and a 40-layer routed-MoE sweep: median layer GPU time is 214.6 μs (q4) and 255.5 μs (q8), with low outliers retained in the raw records. These are routed-MoE-only timings, not end-to-end decode or exact greedy claims. A first scalar 4D prefill QMM was also rejected: its exact r256 kernel remained slower than a shape-matched MLX control.

## Milestone

**M1 — first exact model-specific execution optimization.**

## Testing now

1. run paired same-process A/B tests to remove process/thermal order bias;
2. extend exact and external frontiers to 4K, 16K, 32K, 64K, and 128K;
3. run the multi-turn Python/TypeScript repository workload;
4. implement complete in-memory GDN+KV prefix restore and measure agent-turn TTFT;
5. load real GDN and attention tensors into native probes;
6. replace the rejected scalar prefill tile with a simdgroup matrix/ALU 4D kernel;
7. build the full 40-layer native command graph and measure 1K/8K prefill.

## Next high-value actions

- Add a stateful multi-turn benchmark with complete messages, tools, and reused-prefix accounting.
- Compare Runnel exact 4-bit decode against oMLX's 75 tok/s observation.
- Compare the real-layer two-stage native schedule against pinned MLX per layer.
- Design a matrix/tensor-native 4D prefill kernel after the scalar tile gate failed.
- Build direct full-state GDN+KV snapshot/restore tests.
- Establish greedy-ID parity before labeling any native reduction-order change exact.
- Add package/release CI only after the exact runtime path and result schema stabilize.

## Blockers/risks

- No independent full Transformers/PyTorch greedy oracle has been completed; the pinned current mlx-lm is the project reference, not the final mathematical authority.
- `xctrace` on this macOS/Xcode combination hung while saving traces; MLX Metal capture and targeted synchronization are the fallback.
- GPU counters and joules/token still require a reliable `powermetrics` capture protocol.
- GGUF and community mixed-bit checkpoints are not quality-equivalent to the 8/4-bit targets.
- The native two-stage MoE path currently changes floating-point reduction order and is a numerical baseline; it has no greedy-token parity claim.
- No native MTP tensor exists in the exact checkpoint; speculative work requires a separate drafter and hybrid rollback implementation.
