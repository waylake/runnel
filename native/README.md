# Runnel native Metal probes

This directory contains the C++20/Objective-C++/Metal feasibility runtime for
Ornith-1.5-35B-A3B. It is a measurement harness first, not yet a complete
inference runtime.

## Build and test on the target Mac

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --parallel
ctest --test-dir native/build --output-on-failure
```

The native tests require Apple Metal. Performance numbers are valid only on the
recorded M1 Max environment.

## Current probes

- `runnel-metal-stream`: resident-memory bandwidth gate. The single-workgroup
  whole-token hypothesis was rejected; 32 workgroups are required to approach
  measured memory bandwidth.
- `runnel-metal-qmv`: affine q4/q8 weight-major QMV with CPU-reference checks,
  optional non-contiguous expert banks, and SIMD-level expert mapping.
- `runnel-metal-qmm`: first logical 4D prefill tile prototype. Its exact
  scalar r256 kernel is retained as a negative result; it is slower than the
  shape-matched MLX control and is not a product path.
- `runnel-metal-fused-moe`: legacy channel-shard megakernel and the current
  two-stage schedule. The two-stage command buffer is the leading candidate;
  its floating-point reduction order is not yet an exact greedy claim.

Example synthetic two-stage run:

```bash
native/build/runnel-metal-fused-moe \
  --bits q4 --total-experts 64 \
  --experts 3,17,29,42,50,61,7,55 \
  --groups 32 --threads 512 \
  --warmups 5 --trials 20 --two-stage
```

## Real checkpoint tensors

Model weights must remain outside git. Export one layer to an ignored cache
file, then pass that file to the native probe:

```bash
python bench/export_ornith_moe_layer.py \
  --model /path/to/Ornith-1.5-35B-A3B-MLX-4bit \
  --layer 0 --experts 3,17,29,42,50,61,7,55 \
  --output bench/cache/ornith-moe-q4-layer0.bin

native/build/runnel-metal-fused-moe \
  --bits q4 --total-experts 256 \
  --experts 3,17,29,42,50,61,7,55 \
  --groups 32 --threads 512 --two-stage \
  --dataset bench/cache/ornith-moe-q4-layer0.bin
```

The exporter writes a manifest containing tensor and derived-file SHA-256
values. It never writes model weights to the repository.

See `experiments/m0/0010-m1-two-stage-moe-schedule.md` and
`experiments/m0/0011-ornith-real-layer-two-stage.md` for hypotheses, negative
results, and raw-result links.
