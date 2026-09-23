# Research map

_Cutoff: 2026-09-24. Primary sources are pinned in the final section. Measurements labeled **local** were reproduced on the target M1 Max and link to raw JSON in `bench/results/m0/`._

## 1. Target facts

The exact local 8-bit checkpoint is Hugging Face revision `02440c39bdf7365c494a7f55f2a8b104ba87562f` of `ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit`.

| Property | Verified value |
|---|---:|
| Architecture | `Qwen3_5MoeForConditionalGeneration` |
| Decoder layers | 40 = 30 Gated DeltaNet + 10 full attention |
| Hidden size | 2,048 |
| Full attention | 16 Q heads, 2 KV heads, head dimension 256 |
| MoE | 256 routed experts, top 8, plus one shared expert |
| MoE intermediate | 512 |
| Recurrent state | 30 × 32 × 128 × 128 FP32 matrices plus convolution history |
| Vocabulary | 248,320 |
| Native maximum position | 262,144 |
| Quantization | MLX affine 8-bit, group 64 |
| Logical parameters | 34,660,608,768 |
| Stored tensor bytes | 36,827,986,176 |
| MTP tensors | **None** despite `mtp_num_hidden_layers: 1` in config |

The same architecture has an official MLX 4-bit checkpoint at revision `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`. It is a different quantized target and is never compared as quality-equivalent to the 8-bit checkpoint.

### Active-weight roofline

The header-derived batch-1 estimate is:

| Checkpoint | Active parameters/token | Estimated active weight bytes/token | 400 GB/s weight-only ceiling |
|---|---:|---:|---:|
| MLX 8-bit | 2,946,429,568 | 3,131,668,736 | 127.73 tok/s |
| MLX 4-bit | 2,946,429,568 | 1,669,560,576 | 239.58 tok/s |

These are optimistic ceilings. They exclude KV, recurrent-state traffic, activations, command gaps, cache effects, and imperfect expert-weight streaming. They do not include the 248,320-way vocabulary projection for every prefill position.

The dominant logical active components are:

- 1.0116B GDN projection parameters;
- 1.0066B routed-expert parameters across top-8 selection;
- 508.6M LM-head parameters;
- 272.6M full-attention projection parameters;
- 146.9M shared-expert and router parameters.

## 2. Exactness vocabulary

| Label | Meaning |
|---|---|
| Project-reference exact | Greedy token IDs equal the pinned `mlx-lm` project reference for the same checkpoint, template, and request |
| Distribution-exact speculative | Target probabilities and non-anticipating rejection sampling preserve the target law; finite-precision greedy parity is not implied |
| Lossy cache | KV/state compression changes target computation |
| Numerical baseline | Fast path changes finite-precision results and failed greedy-ID parity; it may be mathematically equivalent but is not admitted to the exact lane |

The project reference is `mlx-lm` commit `15bcf8b929e5da67aa7f8fe6feda7fb9eda00050`. It includes the packed GDN kernel and corrected q/k L2-normalization epsilon. Legacy `mlx-lm` 0.31.3 results remain useful history but are not silently relabeled as current-reference exact.

## 3. Local performance frontier

Primary protocol: deterministic public Python fixture, exact 1,024 or 8,192 prompt tokens, batch 1, greedy, 64 output tokens, one warmup, three measured trials, median reported.

| Runtime / checkpoint | Context | TTFT | Decode tok/s | Correctness | Note |
|---|---:|---:|---:|---|---|
| `mlx-lm` 15bcf8b, 8-bit | 1K | 1.557 s | 52.61 | project reference | current source baseline |
| Runnel packed routed gate/up, 8-bit | 1K | 1.506 s | **56.53** | matched | +7.44% decode |
| `mlx-lm` 15bcf8b, 8-bit | 8K | 10.768 s | 50.09 | project reference | uncached prompt |
| Runnel packed routed gate/up, 8-bit | 8K | 10.825 s | **52.51** | matched | +4.84% decode |
| `mlx-lm` 15bcf8b, official 4-bit | 1K | 1.467 s | 68.45 | 4-bit reference | different quality target |
| Runnel packed routed gate/up, official 4-bit | 1K | 1.458 s | **73.15** | matched | +6.86% decode |
| `mlx-lm` 15bcf8b, official 4-bit | 8K | 11.001 s | 61.94 | 4-bit reference | uncached prompt |
| Runnel packed routed gate/up, official 4-bit | 8K | 11.004 s | **65.40** | matched | +5.58% decode |
| oMLX 0.7.0.dev4, 8-bit | 1K | 1.883 s | 50.06 | greedy mismatch | experimental accelerations off |
| oMLX 0.7.0.dev4, official 4-bit | 1K | 1.668 s | **75.18** | greedy mismatch | fastest observed 1K decode |
| oMLX 0.7.0.dev4, official 4-bit | 8K | 6.739 s | 65.44 | greedy mismatch | trials reused 4,192 prompt tokens; cold server TTFT was 12.53 s |

The packed projection optimization is useful but **not a breakthrough**. It is kept because it is exact, repeatable, model-specific, and consistently improves decode. It has not yet beaten oMLX's 4-bit steady decode.

## 4. Runtime and technique landscape

| Project | License | Relevant implementation | M1/Ornith evidence | Constraint |
|---|---|---|---|---|
| [MLX](https://github.com/ml-explore/mlx) | MIT | Metal quantized GEMV/GEMM, SDPA, GDN kernels | Directly measured through mlx-lm | Generic graph/runtime; custom kernels still need M1 tuning |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | Qwen3.5 model, packed GDN, MoE gather paths | Project reference and baseline | Python graph/launch overhead remains |
| [oMLX](https://github.com/jundot/omlx) | Apache-2.0 | Paged/SSD prefix cache, GDN snapshots, ANE/CPU prefill, TurboQuant, DFlash/MTP paths | Fastest local 4-bit observation; 8-bit baseline measured | Greedy output differs from project reference; optional paths need exactness tests |
| [MTPLX](https://github.com/youssofal/MTPLX) | Apache-2.0 | MTP depth tuning and hybrid rollback | No exact Ornith-1.5 35B M1 result found | Catalog centers on Qwen3.5/3.6/3.8 |
| [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) | Apache-2.0 | Rapid serving/runtime paths | No exact Ornith-1.5 35B M1 result found | Must not be ranked from proxy models |
| [DFlash](https://github.com/z-lab/dflash) | MIT | Parallel block-draft speculative decoding | Strong B200/M4 reports; Ornith mismatch issue | No released target-matched Ornith drafter; M1 results can regress |
| [DFlash2](https://inco.ai/blog/dflash2/) | MIT code / model terms vary | Candidate selector + dynamic convolutions | M1/M5 issue results are numerically sensitive | No official Ornith checkpoint |
| [DSpark](https://arxiv.org/abs/2607.05147) | MIT | Semi-autoregressive draft and scheduler | No Apple implementation found | Batch-serving gains are weak evidence for batch 1 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Metal GDN, packed MoE, bounded recurrent rollback, MTP/DFlash2 | M1 Qwen3.5-9B proxy; no exact Ornith M1 row | GGUF precision/checkpoint differs |
| [d-inference](https://github.com/Layr-Labs/d-inference) | Proprietary | Complete hybrid KV + GDN + MTP SSD prefix checkpoints | Published exact-ID cache reports | Establishes high novelty risk for generic hybrid prefix caching |

### Optimization map

| Technique | Prefill/decode | Exact? | Evidence | Runnel decision |
|---|---|---|---|---|
| Pack routed gate/up into one `gather_qmm` | both | yes, greedy IDs matched | +4.8–7.4% decode locally | **Keep** |
| Pack shared-expert gate/up | both | yes | No repeatable gain over routed fusion | Kill |
| Project LM head only at final prefill position | prefill/TTFT | generation-semantic yes | No repeatable 4-bit gain; initial 8-bit gain was confounded by a slow baseline run | Do not ship by default |
| MLX core chunk-8 GDN | prefill | no greedy parity | ~5.6% 8K prefill gain, token mismatch | Numerical research lane only |
| oMLX ANE Qwen3.5 prefill | prefill | runtime baseline | 1K neutral; 8K cached TTFT ~2.5% slower | Kill for this target |
| TurboQuant KV 8-bit | memory/decode | lossy | ~10% slower at 1K | Kill at 8-bit; quality mode only if long-context evidence reverses |
| Complete hybrid prefix reuse | TTFT/agent turns | potentially | oMLX reused 4,192/8,192 tokens and cut warm TTFT to 6.74 s | High-priority next exact runtime feature |
| MTP | decode | checkpoint-dependent | Exact MLX checkpoint has no MTP tensor | Blocked natively; external drafter requires a separate parity study |
| DFlash/DFlash2 | decode | target/drafter dependent | No target-matched Ornith release | Do not integrate before exact baseline work |

## 5. Component evidence

The synchronized component profiler uses evaluated lazy outputs and actual checkpoint weights. At 8,192 tokens, isolated medians were:

- GDN layer: 0.202 s;
- full-attention layer: 0.438 s;
- MoE block: 0.122 s;
- LM head across all positions: 1.738 s;
- LM head at one position: 0.0025 s.

Multiplying layers gives a non-additive estimate because each isolated case flushes allocator/cache state and prevents end-to-end overlap. The result is still directional: GDN, full attention, and MoE all matter; LM-head projection is large enough to test but did not survive end-to-end A/B validation as a default optimization.

## 6. Novelty boundary

Not novel by themselves:

- another MLX server or CLI;
- generic paged KV or content-addressed prefix chunks;
- another MTP implementation;
- gate/up concatenation when Transformers and llama.cpp already use packed layouts;
- generic SSD caching of Qwen hybrid state.

Potentially defensible work must add target-specific evidence and implementation, such as:

- an M1-tuned fused quantized expert path with full dispatch/coverage checks;
- a complete exact GDN+KV prefix format with verified state identity and rollback;
- a novel low-copy recurrent rollback/checkpoint kernel;
- a reproducible agent-turn benchmark that exposes where generic prefix caches fail.

## 7. Primary sources

- Target model card: <https://huggingface.co/ornith-ai/Ornith-1.5-35B-MLX-8bit>
- Official 4-bit model: <https://huggingface.co/ornith-ai/Ornith-1.5-35B-MLX-4bit>
- MLX pinned source: <https://github.com/ml-explore/mlx/tree/59d600b5e64c238427d0f8d897ab7c682ef4d3d2>
- mlx-lm pinned source: <https://github.com/ml-explore/mlx-lm/tree/15bcf8b929e5da67aa7f8fe6feda7fb9eda00050>
- mlx-lm packed GDN PR: <https://github.com/ml-explore/mlx-lm/pull/1559>
- MLX GDN core PR: <https://github.com/ml-explore/mlx/pull/4020>
- Gated Delta Networks paper: <https://arxiv.org/abs/2412.06464>
- oMLX source: <https://github.com/jundot/omlx>
- oMLX Lightning MTP PR: <https://github.com/jundot/omlx/pull/2113>
- oMLX MTP regression issue: <https://github.com/jundot/omlx/issues/2150>
- DFlash paper/code: <https://arxiv.org/abs/2602.06036>, <https://github.com/z-lab/dflash>
- DFlash2 report: <https://inco.ai/blog/dflash2/>
- DSpark paper: <https://arxiv.org/abs/2607.05147>
- SpecPrefill paper: <https://arxiv.org/abs/2502.02789>
- TurboQuant paper/code: <https://arxiv.org/abs/2504.19874>, <https://github.com/TheTom/turboquant_plus>
- llama.cpp GDN/MTP implementation: <https://github.com/ggml-org/llama.cpp/pull/20361>, <https://github.com/ggml-org/llama.cpp/pull/22673>
- M1 MTP regression evidence: <https://github.com/ggml-org/llama.cpp/issues/23752>
- Complete hybrid SSD cache report: <https://github.com/Layr-Labs/d-inference/blob/master/docs/reports/2026-09-05-ssd-prefix-cache-model-check.md>
