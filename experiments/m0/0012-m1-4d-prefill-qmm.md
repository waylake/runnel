# Experiment 0012: M1 4D logical prefill QMM tile

## Hypothesis

A prefill schedule can map the logical dimensions `[expert, token tile, output tile, reduction]` onto a physical Metal grid while reusing each quantized weight row across a token tile. A small `8×32×64` logical tile should beat scalar row-wise prefill by keeping the M1 SIMD lanes busy.

## Implementation

The first kernel is `qmm_4d_tile` in `native/shaders/quantized_qmm.metal`:

- physical grid: `(output_tile, token_tile)`;
- logical tile: `token=8`, `output=32`, `reduction=64/256`;
- q4/q8 affine packed weights;
- threadgroup x staging;
- CPU reference for every output.

The r256 revision changes the reduction tile to 256 so q4 uses all 32 SIMD lanes. It also computes the scale/bias group from the global reduction offset.

## Results

Raw records:

- `bench/results/m0/native-m1-qmm-4d-q4-m64-n4096-k2048-r64.json`
- `bench/results/m0/native-m1-qmm-4d-q8-m64-n4096-k2048-r64.json`
- `bench/results/m0/native-m1-qmm-4d-q4-m64-n4096-k2048-r256.json`
- `bench/results/m0/native-m1-qmm-4d-q8-m64-n4096-k2048-r256.json`
- `bench/results/m0/mlx-current-qmm-control-q4.json`
- `bench/results/m0/mlx-current-qmm-control-q8.json`

For `M=64, N=4096, K=2048`:

| Path | q4 median GPU | q8 median GPU |
|---|---:|---:|
| native r64 | 3.94 ms | 3.37 ms |
| native r256 | 1.64 ms | 1.98 ms |
| stock MLX shape control | 0.96 ms wall | 0.87 ms wall |

The MLX control uses deterministic shape-matched random tensors, not the native dataset, so it is a performance control rather than a bitwise A/B correctness comparison. The native kernel is exact against its CPU reference, but remains roughly 1.7–2.3× slower than the stock control depending on bit width.

## Decision

**Kill scalar 4D tile QMM as a product path.** The tile mapping is correct, and r256 confirms that lane utilization was a real bottleneck, but the resulting kernel still does too much scalar unpack/ALU work and does not use M1's matrix/tensor paths effectively.

The next prefill hypothesis must be structural rather than another scalar tile-size sweep:

1. use simdgroup matrix/ALU primitives for the token×output tile;
2. stage packed weights and activations in a layout that avoids per-token unpack;
3. fuse expert/token scheduling with the 4D logical mapping;
4. compare at 1K and 8K against the pinned MLX prefill baseline.

This negative result is retained as evidence; no prefill speedup is claimed.
