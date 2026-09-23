# Experiment 0003: rejected acceleration variants

## Shared-expert gate/up fusion

**Hypothesis:** removing 40 dense launch pairs per token adds to routed fusion.

**Result:** 1K greedy IDs matched, but decode was 55.00 tok/s versus 55.85 for routed fusion alone in the old reference environment. The difference was noise/slightly negative.

**Decision:** kill. Shared expert work is too small relative to routed execution.

## Last-position-only LM head

**Hypothesis:** avoid projecting all prefill positions through the 248,320-way head.

**Result:** the first 8-bit comparison suggested ~29% prefill gain, but a clean stock rerun reached nearly the same prefill throughput. On the 4-bit checkpoint the isolated head cost was large, but end-to-end 8K TTFT did not improve and sometimes regressed ~1–3%.

**Decision:** do not ship by default. The benchmark adapter retains the switch for reproducibility, but the runtime does not.

## oMLX Qwen3.5 ANE prefill

**Hypothesis:** splitting Qwen3.5 prefill across ANE lowers long-context TTFT.

**Result:**

- 1K TTFT 1.870 s versus 1.883 s baseline: noise.
- 8K half-reused TTFT 7.800 s versus 7.610 s: ~2.5% slower.
- cold 8K request also did not improve.

**Decision:** kill for this model/hardware/profile.

## TurboQuant KV 8-bit

**Hypothesis:** lossy KV compression reduces memory traffic enough to help decode.

**Result:** 1K decode fell from 50.06 to 45.06 tok/s and TTFT rose slightly. The cache is too small at 1K to repay transform/dequantization overhead.

**Decision:** kill at 8-bit. Reconsider only with long-context memory pressure and a quality suite.

Raw files remain in `bench/results/m0/`; failures are not deleted.
