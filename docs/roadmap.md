# Roadmap

The order is gated by evidence, not calendar time.

## M0 — Reproducible frontier — complete for 1K/8K

- Capture machine, toolchain, power, memory, and process state without private identifiers.
- Fingerprint every checkpoint shard and derive architecture/active-weight facts from tensor headers.
- Implement fixed-token workloads, streaming TTFT, trial distributions, and correctness status.
- Measure release/current mlx-lm and signed oMLX on the actual M1 Max.
- Isolate ANE prefill and TurboQuant KV behavior.
- Establish the project reference and exactness gate.

Remaining scale-out to 4K/16K/32K/64K/128K belongs to M1 because it interacts with cache and state experiments.

## M1 — Exact model-specific execution — active

Current kept optimization:

- pack routed gate/up output rows into one quantized `gather_qmm` per layer;
- preserve greedy token IDs;
- gain 4.8–7.4% decode across 8-bit/4-bit and 1K/8K tests.

Next experiments, in order:

1. paired same-process A/B benchmark to remove run-order and thermal bias;
2. 4K–128K exact frontier;
3. multi-turn Python/TypeScript repository benchmark;
4. complete in-memory GDN+KV prefix snapshots and restore tests;
5. M1 quantized-qmv tile/SIMD microbenchmarks;
6. direct Metal kernel only if the microbenchmark identifies a material gap.

Exit gate: a repeatable user-visible improvement large enough to retain, or a recorded negative result. The 2× target is not considered met by the current 5–7% decode gain.

## M2 — Agent-oriented exact runtime

- stateful multi-turn session execution;
- content identity bound to checkpoint, template, numerical policy, KV layout, and GDN state;
- partial prefix updates with explicit semantic validation;
- CLI streaming stability and a minimal OpenAI-compatible API after the core path is competitive;
- long-context quality and memory-ceiling suite.

## M3 — Native hot paths

Proceed only if M1/M2 identifies a bottleneck worth native ownership:

- C++/Objective-C++ control plane outside the per-token Python path;
- direct Metal fused quantized MoE and/or recurrent checkpoint kernels;
- one host synchronization per generated token;
- Metal System Trace/counter evidence and complete grid-coverage tests;
- rollback/prefix state parity against the reference.

## Speculative lane

Blocked on a compatible drafter because the exact MLX checkpoint has no MTP tensor. A future drafter must pass:

- target-distribution correctness;
- greedy parity for the project exact lane;
- GDN+KV+convolution rollback;
- net M1 throughput after acceptance and snapshot cost;
- target-matched acceptance on coding prompts.

DFlash/DFlash2 results on other Qwen checkpoints are controls, not evidence for Ornith.

## Completion

The project ends only with either:

- a correct, usable runtime that materially beats the fastest measured exact baseline on the M1 Max workload; or
- a rigorous negative conclusion supported by the best implementation, bottlenecks, failed hypotheses, empirical/theoretical ceilings, and the hardware/model changes needed for another major gain.
