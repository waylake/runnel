# M0 target-hardware results

These files were measured on the target 32-core M1 Max 64 GB on 2026-09-24.

## Groups

- `checkpoint-*.json`: complete SHA-256 manifests and tensor-header audits.
- `environment-*.json`: privacy-scrubbed machine/toolchain/power/memory captures referenced by result provenance.
- `runtime-*.json`: exact runtime/native-library identities.
- `mlx-lm-*.json`: legacy 0.31.3 and current-main baselines/experiments.
- `mlx-lm-current-*.json`: current project-reference lane.
- `mlx-nightly-*.json`: MLX-core chunk-GDN numerical lane; failed greedy parity.
- `omlx-*.json`: signed oMLX 0.7.0.dev4 performance baselines.
- `mlx-component-profile-8bit.json`: evaluated lazy component timings; inclusive and non-additive.
- `invalid/`: explicitly invalid harness output retained to prevent accidental reuse.

## Interpretation

- Read medians and MAD from each JSON; do not select the minimum trial.
- Check `correctness.status` and `protocol.semantic_mode` before comparing.
- 8K oMLX trials using the same prompt report 4,192 cached prompt tokens. They are prefix-reuse measurements, not uncached prefill.
- The legacy 8-bit baseline includes one anomalously slow run and one rerun. Both are retained; final claims use the rerun/current-main lane.
- Prompt token IDs/text are private generated artifacts and are not committed. Public workload source and digest are embedded in each result.
