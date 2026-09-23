# Experiment 0005: hybrid prefix-reuse evidence

## Hypothesis

For coding-agent turns, complete GDN+KV prefix reuse can reduce TTFT more than isolated kernel optimizations.

## Existing implementation measured

oMLX 0.7.0.dev4 was run with the same fixed 8,192-token 4-bit prompt twice after clearing its cache.

- first request: server TTFT 12.53 s, 0 cached prompt tokens;
- subsequent requests: 4,096 cached prompt tokens, server TTFT 6.71–6.80 s;
- observed decode remained about 65–66 tok/s.

The current exact 8K path without prefix reuse measured about 11.0 s TTFT and 65 tok/s decode.

## Interpretation

The cache cuts roughly half of the repeated 8K TTFT while leaving decode unchanged. This matches the target workload: agent turns resend a mostly identical repository/system/tool prefix.

This is evidence, not a Runnel implementation. A valid exact cache must atomically store:

- all full-attention KV blocks;
- every FP32 GDN recurrent state;
- convolution history;
- model/checkpoint/template/numerical-policy identity;
- any MTP state if a compatible drafter is introduced.

A GDN state is a selected complete checkpoint, not a composable content block.

## Decision

High-priority next runtime feature. Generic CAS/SSD KV is not novel; exact complete hybrid-state restore, M1-specific placement, and coding-agent measurements are the contribution surface.
