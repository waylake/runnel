# Changelog

## 0.1.0-alpha.1 — 2026-09-24

First target-hardware research runtime with one verified exact model-specific optimization.

### Added

- Privacy-scrubbed macOS/M1 Max environment capture.
- Full safetensors checkpoint fingerprinting and header-derived architecture/active-weight audit.
- Fixed-token Python repository workloads and streaming TTFT/decode benchmark adapters.
- Synchronized MLX component profiler with explicit lazy-graph evaluation.
- Signed oMLX control helper with reproducible local profiles.
- Streaming `runnel generate` CLI.
- Exact packed routed gate/up `gather_qmm` transform.
- Project reference/exactness ADR and raw 8-bit/4-bit target-hardware result set.

### Verified

- Greedy IDs match the pinned current `mlx-lm` reference.
- Packed routed gate/up improves decode by 4.8–7.4% across the measured 8-bit/4-bit, 1K/8K cases.
- ANE prefill, TurboQuant KV8, shared-expert fusion, and last-position LM-head-only defaulting were rejected or kept out of the product after measurement.
- MLX core chunk-8 GDN changes greedy IDs and remains outside the exact lane.

### Limitations

- No 2× breakthrough yet.
- No OpenAI-compatible server yet.
- Only 1K/8K fixed-prompt sweeps are published; long-context and multi-turn state harnesses are next.
- oMLX 0.7.0.dev4 remains the fastest observed external 4-bit performance baseline.
- The project reference is pinned mlx-lm, not yet an independent full Transformers/PyTorch oracle.
