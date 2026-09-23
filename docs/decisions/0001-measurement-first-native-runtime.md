# ADR 0001: Measurement-first, native optimized runtime

- Status: accepted provisionally
- Date: 2026-09-24

## Context

The target is a 34.66B-parameter, 8-bit hybrid MoE checkpoint on a 32-GPU-core M1 Max. The fastest credible existing implementations already include specialized MLX kernels and substantial cache/runtime machinery. Starting with a broad native rewrite would violate the project's measure-first requirement and could optimize the wrong layer.

At the same time, the final project must contain a real runtime, not only a wrapper or benchmark suite.

## Decision

1. Build a standard-library, privacy-safe benchmark and checkpoint-audit layer first.
2. Preserve stock MLX/mlx-lm as the exact semantic reference.
3. Profile oMLX and other strongest baselines on the actual M1 Max.
4. Implement a native C++/Objective-C++ + Metal optimized plane only behind measured evidence, with Python excluded from its per-token path.
5. Permit MLX in research tooling and as an optional reference kernel source, but never describe the optimized path as novel merely because it combines MLX calls.
6. Reconsider the native decision if profiling shows host overhead is negligible and MLX custom kernels dominate; record replacement evidence in a new ADR.

## Consequences

- Early commits deliver trustworthy measurement rather than a speculative runtime.
- The native runtime has a higher implementation cost and will be staged by bottleneck.
- Exact/approximate paths and external baselines remain auditable.
- The repository can end in a rigorous negative result without preserving failed native complexity.
