# Runnel

Runnel is a model-specific LLM inference runtime and reproducible performance laboratory for **Ornith-1.5-35B-A3B on Apple M1 Max 64 GB**.

It is an active research runtime, not a claim of a completed 2× breakthrough. The current exact execution path includes a model-specific packed routed gate/up projection and a streaming CLI. The first public artifact is the `v0.1.0-alpha.1` prerelease; there is no stable release yet.

## Target

- Apple M1 Max, 32 GPU cores, 64 GB unified memory
- macOS 27.2, Metal 4 / `applegpu_g13s`
- batch 1, single-user coding-agent latency
- `ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit`
- official 4-bit checkpoint as a separately labeled quality/performance lane
- 40 layers: 30 Gated DeltaNet + 10 full attention
- 256 routed experts, top-8, plus one shared expert

The exact 8/4-bit MLX checkpoints contain no MTP tensor despite MTP metadata in `config.json`.

## Current measured result

Current `mlx-lm` project reference versus Runnel, M1 Max, batch 1, greedy, 64 output tokens:

| Checkpoint/context | Reference | Runnel | Gain |
|---|---:|---:|---:|
| 8-bit / 1K decode | 52.61 tok/s | 56.53 tok/s | +7.44% |
| 8-bit / 8K decode | 50.09 tok/s | 52.51 tok/s | +4.84% |
| official 4-bit / 1K decode | 68.45 tok/s | 73.15 tok/s | +6.86% |
| official 4-bit / 8K decode | 61.94 tok/s | 65.40 tok/s | +5.58% |

All four comparisons matched greedy token IDs. These are meaningful exact optimizations, not the 2× target. oMLX 0.7.0.dev4 remains the fastest observed external 4-bit baseline at 75.18 tok/s on the 1K test, but its greedy output differs from the project reference.

See [docs/status.md](docs/status.md), [docs/research.md](docs/research.md), and [bench/results/m0](bench/results/m0/).

## Install

The audit tools require only Python 3.11+:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Install the experimental runtime dependencies for generation:

```bash
python -m pip install -e '.[runtime]'
```

For the exact reference lane, pin `mlx-lm` to commit `15bcf8b929e5da67aa7f8fe6feda7fb9eda00050` as documented in [ADR 0002](docs/decisions/0002-project-reference-and-exactness.md).

## Streaming CLI

```bash
runnel generate \
  --model ~/.omlx/models/ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit \
  --system-prompt 'You are a careful coding agent.' \
  --prompt 'Explain this repository and identify one likely bug.' \
  --max-tokens 256 \
  --temperature 0
```

The default path applies Runnel's packed routed gate/up transform. Use `--stock-mlp` only for an A/B baseline.

## Reproducibility tools

```bash
runnel-env --output /tmp/runnel-env.json

runnel-fingerprint \
  --model ~/.omlx/models/ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit \
  --revision 02440c39bdf7365c494a7f55f2a8b104ba87562f \
  --output /tmp/runnel-checkpoint.json
```

`runnel-fingerprint` hashes all 36.8 GB of checkpoint shards. Result files reference a manifest rather than repeating that cost for every trial.

Environment captures remove machine serial/UUID fields and never store credentials or private prompts.

## Architecture

- `bench/`: protocol, adapters, workload definitions, component profiler, raw results
- `src/runnel/`: streaming CLI and exact model-specific MLX transforms
- `tests/`: deterministic harness/runtime tests
- `experiments/`: hypotheses, outcomes, and negative results
- `docs/decisions/`: reference, exactness, and architecture decisions

The longer-term direction is a native C++/Objective-C++ and Metal runtime for proven hot paths. Python remains outside the optimized per-token path. See [docs/architecture.md](docs/architecture.md) and [ADR 0001](docs/decisions/0001-measurement-first-native-runtime.md).

## Validation

```bash
PYTHONPATH=src:bench python -m unittest discover -s tests -v
python -m compileall -q bench src tests
```

A number without raw trials, checkpoint identity, machine state, cache state, and correctness status is not a result.

## License

Apache License 2.0. See [LICENSE](LICENSE).
