#!/usr/bin/env python3
"""Shape-matched stock MLX quantized-GEMM control for the native 4D probe."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import numpy as np


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(
        np.uint16
    )


def summary(values: list[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    return {
        "n": len(values),
        "median": round(median, 6),
        "mad": round(statistics.median(abs(value - median) for value in values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--outputs", type=int, default=4096)
    parser.add_argument("--reduction-size", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx

    rng = np.random.default_rng(0x4D4D0000 + args.bits)
    values_per_pack = 32 // args.bits
    weights = rng.integers(
        0,
        1 << args.bits,
        size=(args.outputs, args.reduction_size // values_per_pack),
        dtype=np.uint32,
    )
    scales = rng.uniform(
        0.002 if args.bits == 4 else 0.00005,
        0.008 if args.bits == 4 else 0.00015,
        size=(args.outputs, args.reduction_size // 64),
    ).astype(np.float32)
    biases = rng.uniform(
        -0.0005, 0.0005, size=scales.shape
    ).astype(np.float32)
    x = rng.uniform(-1.0, 1.0, size=(1, args.tokens, args.reduction_size)).astype(
        np.float32
    )

    x_mx = mx.array(to_bf16(x), dtype=mx.bfloat16)
    weights_mx = mx.array(weights, dtype=mx.uint32)
    scales_mx = mx.array(to_bf16(scales), dtype=mx.bfloat16)
    biases_mx = mx.array(to_bf16(biases), dtype=mx.bfloat16)

    def run_once() -> float:
        started = time.perf_counter_ns()
        result = mx.quantized_matmul(
            x_mx,
            weights_mx,
            scales_mx,
            biases_mx,
            transpose=True,
            group_size=64,
            bits=args.bits,
            mode="affine",
        )
        mx.eval(result)
        return (time.perf_counter_ns() - started) / 1e3

    for _ in range(args.warmups):
        run_once()
    wall = [run_once() for _ in range(args.trials)]
    document = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "engine": {
            "name": "mlx.quantized_matmul",
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        },
        "protocol": {
            "semantic_mode": "baseline",
            "bits": args.bits,
            "tokens": args.tokens,
            "outputs": args.outputs,
            "reduction_size": args.reduction_size,
            "group_size": 64,
            "transpose": True,
            "warmups": args.warmups,
            "trials": args.trials,
            "data": "deterministic shape-matched random control; not the native dataset",
        },
        "summary": {"wall_micros": summary(wall)},
        "trials": [{"wall_micros": round(value, 6)} for value in wall],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
