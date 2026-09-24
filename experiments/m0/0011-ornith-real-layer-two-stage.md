# Experiment 0011: Real Ornith layer tensors through the native two-stage MoE kernel

## Hypothesis

The synthetic two-stage schedule must work with the exact packed tensor layout and shapes from the pinned Ornith checkpoints before any end-to-end decode claim is credible.

## Method

- `bench/export_ornith_moe_layer.py` reads safetensors headers and raw tensor bytes without copying model weights into git.
- One layer is exported to a temporary ignored binary containing:
  - fused up/gate packed weights, scales, and biases;
  - down packed weights, scales, and biases;
  - BF16 activation vector;
  - deterministic router weights.
- The binary includes tensor SHA-256 values and an output SHA-256 in its manifest.
- All 40 layers were run with the same eight scattered selected experts, 32 workgroups × 512 threads, two warmups, and five measured trials.
- No model weights are included in the repository or result JSON.

Raw records:

- `bench/results/m0/native-ornith-all-layers-q4.json`
- `bench/results/m0/native-ornith-all-layers-q8.json`
- `bench/results/m0/native-ornith-representative-layers-q4.json`
- `bench/results/m0/native-ornith-representative-layers-q8.json`

## Results

The 40-layer layer-median GPU-time distribution is:

| Bits | Median layer | Range | Mean | Sum of layer medians |
|---|---:|---:|---:|---:|
| q4 | 214.6 μs | 93.3–242.2 μs | 197.8 μs | 7.91 ms |
| q8 | 255.5 μs | 134.6–261.8 μs | 229.4 μs | 9.18 ms |

The low outliers are retained in the raw records. Excluding only layers below 180 μs, the stable-layer means are 216.2 μs (q4) and 255.5 μs (q8). The q4/q8 output checks stayed finite; the maximum absolute CPU-reference error across the layer sweep was 0.04.

These are routed-MoE weight/activation timings only. They do not include GDN, full attention, shared expert, normalization, LM head, sampling, or KV-cache work. The sum of medians is therefore not an end-to-end decode prediction.

## Decision

**Pass the real-tensor integration gate.** The packed layout, expert gather, two-stage dependency, and M1 launch geometry are valid on the actual checkpoint.

Next gates are:

1. run the real GDN and attention tensors through their own native probes;
2. build a full per-layer command graph and measure 1K/8K prefill;
3. compare the full native graph against pinned MLX;
4. run greedy-token parity for the exact lane;
5. only then extrapolate decode speed or claim a 2× result.
