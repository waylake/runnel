#!/usr/bin/env python3
"""Benchmark pinned MLX affine QMV on the native probe's synthetic dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HEADER = struct.Struct("<8sIIIIIIII")
_MAGIC = b"RNLQMV02"


def load_dataset(path: Path) -> dict[str, np.ndarray | int]:
    raw = path.read_bytes()
    if len(raw) < _HEADER.size:
        raise SystemExit("QMV dataset is truncated")
    (
        magic,
        version,
        bits,
        output_size,
        weight_bank_output_size,
        rows_per_expert,
        reduction_size,
        group_size,
        selected_expert_count,
    ) = _HEADER.unpack_from(raw)
    if magic != _MAGIC or version != 2:
        raise SystemExit("unsupported QMV dataset header")
    values_per_pack = 32 // bits
    if reduction_size % values_per_pack or reduction_size % group_size:
        raise SystemExit("QMV dataset shape is incompatible with packing")

    offset = _HEADER.size
    weight_count = weight_bank_output_size * (reduction_size // values_per_pack)
    parameter_count = weight_bank_output_size * (reduction_size // group_size)

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal offset
        size = count * np.dtype(dtype).itemsize
        end = offset + size
        if end > len(raw):
            raise SystemExit("QMV dataset is truncated")
        array = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
        offset = end
        return array

    weights = take("<u4", weight_count).reshape(
        weight_bank_output_size, reduction_size // values_per_pack
    )
    scales = take("<u2", parameter_count).reshape(
        weight_bank_output_size, reduction_size // group_size
    )
    biases = take("<u2", parameter_count).reshape(
        weight_bank_output_size, reduction_size // group_size
    )
    x_bits = take("<u2", reduction_size)
    selected_experts = take("<u4", selected_expert_count)
    if offset != len(raw):
        raise SystemExit("QMV dataset contains trailing bytes")
    x = (x_bits.astype(np.uint32) << 16).view(np.float32)
    return {
        "bits": bits,
        "output_size": output_size,
        "weight_bank_output_size": weight_bank_output_size,
        "rows_per_expert": rows_per_expert,
        "selected_experts": selected_experts,
        "reduction_size": reduction_size,
        "group_size": group_size,
        "weights": weights,
        "scales": (scales.astype(np.uint32) << 16).view(np.float32),
        "biases": (biases.astype(np.uint32) << 16).view(np.float32),
        "x": x,
    }


def cpu_reference(dataset: dict[str, np.ndarray | int]) -> np.ndarray:
    bits = int(dataset["bits"])
    output_size = int(dataset["output_size"])
    weight_bank_output_size = int(dataset["weight_bank_output_size"])
    rows_per_expert = int(dataset["rows_per_expert"])
    selected_experts = np.asarray(dataset["selected_experts"], dtype=np.uint32)
    reduction_size = int(dataset["reduction_size"])
    group_size = int(dataset["group_size"])
    weights = np.asarray(dataset["weights"])
    scales = np.asarray(dataset["scales"])
    biases = np.asarray(dataset["biases"])
    x = np.asarray(dataset["x"])
    values_per_pack = 32 // bits
    output = np.zeros((output_size,), dtype=np.float32)
    maximum = (1 << bits) - 1
    active_weight_rows = (
        selected_experts[:, None].astype(np.int64) * rows_per_expert
        + np.arange(rows_per_expert, dtype=np.int64)[None, :]
    ).reshape(-1)

    for group in range(reduction_size // group_size):
        start = group * group_size
        stop = start + group_size
        if bits == 4:
            shifts = np.arange(8, dtype=np.uint32) * 4
            unpacked = (weights[:, start // 8 : stop // 8, None] >> shifts) & 0x0F
        else:
            shifts = np.arange(4, dtype=np.uint32) * 8
            unpacked = (weights[:, start // 4 : stop // 4, None] >> shifts) & 0xFF
        unpacked = unpacked.reshape(weight_bank_output_size, group_size)
        quantized = unpacked[active_weight_rows].astype(np.float32)
        x_group = x[start:stop]
        output += (quantized @ x_group) * scales[active_weight_rows, group]
        output += np.sum(x_group, dtype=np.float32) * biases[active_weight_rows, group]
        if np.any(quantized > maximum):
            raise SystemExit("invalid synthetic quantized value")
    return output


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "mad": None, "min": None, "max": None}
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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()
    if args.warmups < 0 or args.trials <= 0:
        raise SystemExit("warmups must be non-negative and trials must be positive")

    import mlx.core as mx

    dataset = load_dataset(args.dataset)
    bits = int(dataset["bits"])
    output_size = int(dataset["output_size"])
    weight_bank_output_size = int(dataset["weight_bank_output_size"])
    selected_experts = np.asarray(dataset["selected_experts"], dtype=np.uint32)
    if weight_bank_output_size != output_size or selected_experts.tolist() != [0]:
        raise SystemExit("MLX dense-QMV adapter does not yet support expert gather datasets")
    reduction_size = int(dataset["reduction_size"])
    group_size = int(dataset["group_size"])
    x = mx.array(np.asarray(dataset["x"])[None, :], dtype=mx.bfloat16)
    weights = mx.array(np.asarray(dataset["weights"]), dtype=mx.uint32)
    scales = mx.array(np.asarray(dataset["scales"]), dtype=mx.bfloat16)
    biases = mx.array(np.asarray(dataset["biases"]), dtype=mx.bfloat16)

    def run_once() -> tuple[float, object]:
        mx.reset_peak_memory()
        started = time.perf_counter_ns()
        result = mx.quantized_matmul(
            x,
            weights,
            scales,
            biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        mx.eval(result)
        elapsed = (time.perf_counter_ns() - started) / 1e3
        return elapsed, result

    for _ in range(args.warmups):
        _, result = run_once()
    mx.synchronize()

    trials: list[dict[str, float]] = []
    last_result = None
    for _ in range(args.trials):
        elapsed, result = run_once()
        trials.append(
            {
                "wall_micros": round(elapsed, 6),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 6),
            }
        )
        last_result = result
    assert last_result is not None

    actual = np.asarray(last_result.astype(mx.float32)).reshape(-1)
    reference = cpu_reference(dataset)
    max_absolute_error = float(np.max(np.abs(actual - reference)))
    rmse = float(np.sqrt(np.mean(np.square(actual - reference))))
    output_bytes = args.dataset.read_bytes()
    result_document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "correctness": {
            "status": "matched" if max_absolute_error <= 1e-2 else "mismatched",
            "reference": "vectorized CPU unpack/dequant/dot",
            "max_absolute_error": max_absolute_error,
            "rmse": rmse,
            "output_checksum": float(np.sum(actual, dtype=np.float64)),
            "reference_checksum": float(np.sum(reference, dtype=np.float64)),
        },
        "engine": {
            "name": "mlx.quantized_matmul",
            "runtime_version": importlib.metadata.version("mlx-lm"),
            "mlx_version": importlib.metadata.version("mlx"),
        },
        "protocol": {
            "semantic_mode": "baseline",
            "batch_size": 1,
            "sampling": None,
            "bits": bits,
            "output_size": output_size,
            "reduction_size": reduction_size,
            "group_size": group_size,
            "transpose": True,
            "warmups": args.warmups,
            "trials_requested": args.trials,
            "dataset_sha256": hashlib.sha256(output_bytes).hexdigest(),
        },
        "trials": trials,
        "summary": {
            "successful_trials": len(trials),
            "wall_micros": summarize([trial["wall_micros"] for trial in trials]),
            "peak_memory_gb": summarize(
                [trial["peak_memory_gb"] for trial in trials]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
