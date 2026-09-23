# Roadmap

The order is gated by evidence, not calendar time.

## M0 — Reproducible frontier

- Capture machine, toolchain, power, memory, and process state without private identifiers.
- Fingerprint every checkpoint shard and derive architecture facts from tensor headers.
- Implement fixed-token and multi-turn coding-agent workloads.
- Benchmark stock mlx-lm, oMLX baseline, and applicable accelerated runtimes with raw trials.
- Establish greedy token-ID and recurrent-cache differential tests.

Exit gate: the strongest exact baseline is reproducible and the bottleneck model has measured shares.

## M1 — Small exact experiments

Choose at most two from:

- route/top-k and expert-index fusion;
- grouped 8-bit MoE decode with reduced intermediate materialization;
- Gated DeltaNet recurrent state/update fusion;
- full-attention KV layout or M1-specific threadgroup tuning;
- CPU/GPU synchronization and command-buffer reduction;
- content-addressed prefix reuse, including GDN state snapshots.

Exit gate: repeatable user-visible improvement or a recorded negative result.

## M2 — Model-specific runtime

- Native C++/Objective-C++ control plane where measurements justify it.
- Direct Metal kernels only for proven hot paths.
- Differential validation against the untouched reference path.
- Streaming CLI and minimal OpenAI-compatible server after the core path is competitive.

## M3 — Long-context agent path

- Incremental prompt invalidation rather than full prompt resend where semantically valid.
- Reused-prefix ratio and avoided-prefill accounting.
- Python, TypeScript, multi-file, large-tool-schema, and long-system-prompt scenarios.
- Stable repeated-turn behavior and memory ceilings.

## Completion

The project ends only with either:

- a correct, usable runtime that materially beats the fastest measured exact baseline on the M1 Max workload; or
- a rigorous negative conclusion supported by the best implementation, bottlenecks, failed hypotheses, empirical/theoretical ceilings, and the hardware/model changes needed for another major gain.
