# Experiment 0001: M1 Max baseline frontier

## Hypothesis

Stock `mlx-lm`, current oMLX, the official 4-bit checkpoint, and proxy implementations can be distinguished on the actual 32-core M1 Max 64 GB target before attempting custom work.

## Implementation

- fingerprinted all checkpoint shards;
- captured target macOS/Xcode/Metal/compiler/power/memory state;
- used one fixed 1,024-token Python repository prompt and one 8,192-token prompt;
- greedy sampling, 64 generated tokens, batch 1, one warmup, three measured trials;
- ran release 0.31.3, current mlx-lm 15bcf8b, and signed oMLX 0.7.0.dev4;
- downloaded and pinned the official MLX 4-bit revision.

## Benchmark

See:

- `bench/results/m0/checkpoint-ornith-mlx-8bit.json`
- `bench/results/m0/checkpoint-ornith-mlx-4bit.json`
- `bench/results/m0/mlx-lm-current-1k-64-greedy.json`
- `bench/results/m0/mlx-lm-current-8k-64-greedy.json`
- `bench/results/m0/mlx-lm-current-4bit-1k-64-greedy.json`
- `bench/results/m0/mlx-lm-current-4bit-8k-64-greedy.json`
- `bench/results/m0/omlx-4bit-1k-64-greedy-baseline.json`
- `bench/results/m0/omlx-4bit-8k-64-greedy-baseline.json`

## Outcome

- Current 8-bit reference: 52.61 tok/s at 1K, 50.09 tok/s at 8K.
- Official 4-bit reference: 68.45 tok/s at 1K, 61.94 tok/s at 8K.
- oMLX 4-bit: 75.18 tok/s at 1K; 65.44 tok/s at 8K with half-prefix reuse.
- oMLX outputs differ from the current project reference, so it is a performance baseline rather than an exact-ID reference.
- An initial 8-bit baseline run was anomalously slow. A later rerun reached 48.84 tok/s at 8K instead of 40.96. The original result remains tracked and is not used for the final optimization claim.

## Conclusion

The real local frontier is higher than old web posts: current `mlx-lm` is the correctness reference, while oMLX 0.7.0.dev4 is the fastest observed 4-bit runtime. A new runtime must beat both appropriate lanes: exact 8-bit, exact official 4-bit, and oMLX numerical/performance baseline.
