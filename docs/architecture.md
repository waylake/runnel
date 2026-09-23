# Architecture

## Product boundary

Runnel is not intended to be an MLX CLI wrapper. The benchmark package may use MLX as a trusted reference and loader, but the optimized runtime must own model execution, state management, scheduling, and native hot paths when profiling demonstrates a bottleneck.

## Current staged architecture

### Research control plane

Python 3.11+ owns experiment configuration, checkpoint auditing, workload generation, correctness comparison, and result capture. It is not in the per-token inference path.

### Reference plane

Stock `mlx-lm` remains the untouched semantic reference for the exact 8-bit checkpoint. Its generated token IDs, per-position logits where feasible, MoE routes, KV state, and GDN state are used for differential testing.

### Current exact experimental plane

`src/runnel/mlx.py` owns the first product optimization rather than leaving it inside benchmark code. It concatenates the equally quantized routed up/gate output rows, executes one `gather_qmm`, restores the original activation/down path, and proves greedy parity against the pinned reference. `src/runnel/cli.py` provides streaming batch-1 generation.

This path is still MLX-based and is not presented as the final native runtime. It exists to test one model-specific graph change quickly and to provide a usable exact CLI while native kernel experiments continue.

### Candidate native plane

A native C++/Objective-C++ runtime with direct Metal kernels is the leading candidate because it offers the shortest control path to Metal command buffers, explicit resource residency, counters, and profiling. Python remains outside the hot loop. MLX may supply reference kernels or tensor-loading utilities, but Runnel will not depend on Python for optimized decode.

This choice is provisional until the M0 profile identifies where time is actually spent. Evidence can reject it.

## Execution model to investigate

For one token, the target checkpoint requires model-specific treatment:

1. input normalization and layer-specific linear-attention or full-attention state update;
2. quantized router selection over 256 experts;
3. one shared expert and eight routed experts;
4. residual/norm sequence matching the checkpoint implementation;
5. final LM-head projection over 248,320 tokens.

At batch 1, per-expert dispatch and full quantized matrix reads are likely more important than server throughput. A grouped MoE path and a specialized GDN path are therefore higher-value initial hypotheses than generic scheduler features.

## State model

The durable state design must distinguish:

- full-attention KV pages;
- Gated DeltaNet recurrent state and convolution state;
- content-addressed prefix checkpoints for both;
- speculative branch checkpoints and rollback cost;
- resident unified-memory budget versus SSD spill traffic.

A cache hit may reduce end-to-end latency, but it is never reported as prefill throughput.

## Semantic modes

- `exact`: greedy IDs match the reference for the same quantized checkpoint and request semantics.
- `approximate`: quality-changing methods are isolated, named, and evaluated independently.
- `baseline`: external runtime results record quantization and implementation differences explicitly.
