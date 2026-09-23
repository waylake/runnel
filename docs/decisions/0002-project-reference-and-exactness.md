# ADR 0002: project reference and exactness gate

- Status: accepted
- Date: 2026-09-24

## Context

`mlx-lm` 0.31.3 and current main produce different greedy IDs for the same 8-bit Ornith prompt. The mainline changes include corrected q/k L2-normalization epsilon and a packed GDN kernel whose tests pin it to an explicit reduction-tree comparator. oMLX also produces different IDs.

“Exact” therefore needs a named reference rather than an unqualified claim of mathematical sameness.

## Decision

1. The project-reference implementation is `mlx-lm` commit `15bcf8b929e5da67aa7f8fe6feda7fb9eda00050`, MLX 0.32.2, for a pinned checkpoint and pinned prompt/template.
2. An exact optimization must match all 64 greedy token IDs in the standard differential set and pass longer prompt/differential tests before release.
3. Logits, MoE routes, GDN state, and KV state receive separate comparisons; token parity alone is insufficient evidence for a general API.
4. oMLX is retained as a performance baseline. Faster output with different greedy IDs is not admitted to the exact lane.
5. MLX-core chunk GDN and future reordered reductions remain numerical baselines until they satisfy the same gate.
6. A full Transformers/PyTorch or equivalent independent model check is still required before calling the project reference the ultimate target-model oracle.

## Consequences

- Legacy and current-main results stay separated.
- A mathematically exact rewrite can still fail the practical exact gate because finite-precision ordering changes greedy near-ties.
- Performance claims state the reference lane explicitly.
