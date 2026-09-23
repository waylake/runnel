# Runnel agent contract

## Mission

Optimize Ornith-1.5-35B-A3B for batch-1 coding-agent inference on an M1 Max 64 GB. Measure before optimizing and preserve prior art.

## Required workflow

For every significant experiment:

1. state a falsifiable hypothesis and expected bottleneck;
2. record the exact baseline, checkpoint fingerprint, configuration, and machine state;
3. implement the smallest useful experiment;
4. run warmups and repeated trials;
5. compare distributions and profile the result;
6. keep, revise, or kill the idea;
7. record negative as well as positive outcomes.

## Integrity rules

- Label `exact/lossless` and `approximate` paths separately.
- Never compare different quantization or quality requirements as if equivalent.
- Never report M1 Max numbers measured on another chip.
- Never call cached-prefix restoration prefill throughput.
- Never report best-of-N as typical performance.
- Preserve full raw data, including regressions and failed hypotheses.
- Do not invent novelty; cite primary sources and inspect prior art first.
- Do not commit model weights, local prompts, chat histories, credentials, API keys, machine serials, platform UUIDs, or cache state.

## Repository boundaries

- `bench/`: protocol, adapters, workload generation, raw results.
- `src/`: runtime and native kernels.
- `tests/`: correctness and harness tests.
- `experiments/`: concise experiment-family records.
- `docs/decisions/`: architecture and protocol decisions.
- `docs/status.md`: current resumable state; keep concise.

## Validation

Run before commit:

```bash
PYTHONPATH=src:bench python -m unittest discover -s tests -v
python -m compileall -q bench src tests
```

Add model correctness tests before claiming an exact optimization. Benchmark results on hosted macOS CI are functional evidence only, never M1 Max performance evidence.
