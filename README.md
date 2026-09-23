# Runnel

Runnel is a model-specific inference runtime and reproducible performance laboratory for **Ornith-1.5-35B-A3B on Apple M1 Max 64 GB**.

The project is intentionally in its **measurement-first** phase. It does not yet claim a speedup over MLX, oMLX, MTPLX, DSpark, llama.cpp, or any other runtime. The first deliverable is a trustworthy target-hardware harness and checkpoint audit; native runtime work follows only where profiling justifies it.

## Target

- Apple M1 Max, 32 GPU cores, 64 GB unified memory
- macOS 27.2, Metal 4 / Apple GPU family `applegpu_g13s`
- batch 1, single-user coding-agent latency
- exact checkpoint: `ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit`
- model family: 40-layer hybrid linear/full attention, 256-expert top-8 MoE, shared expert

Checkpoint files, local paths, API keys, and generated private prompts are never committed.

## Current milestone

**M0 — environment, checkpoint audit, and benchmark protocol.**

See [docs/status.md](docs/status.md) for the live state and [docs/roadmap.md](docs/roadmap.md) for the measured sequence.

## Quick start

The inspection tools use only the Python standard library:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

runnel-env --output /tmp/runnel-env.json
runnel-fingerprint \
  --model ~/.omlx/models/ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit \
  --output /tmp/runnel-checkpoint.json
```

`runnel-fingerprint` hashes all 36.8 GB of checkpoint shards by default, so the first run is intentionally expensive. Benchmark result files should reference one immutable fingerprint rather than re-hashing the model for every trial.

## Reproducibility contract

Every tracked benchmark must record:

- repository commit and runtime version;
- checkpoint revision, shard manifest, quantization, and architecture facts derived from tensors;
- macOS, Xcode, Metal, compiler, chip, GPU-core count, power and memory state;
- exact prompt/output/cache/sampling protocol;
- cold/warm status, warmups, trial count, and full per-trial values;
- medians and dispersion, not best-of-N;
- exact versus approximate semantics;
- correctness output, including token IDs or a digest when an exact stream is available.

A number without its raw trial data and environment is not a result.

## License

Apache License 2.0. See [LICENSE](LICENSE).
