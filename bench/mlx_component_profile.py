#!/usr/bin/env python3
"""Synchronized component profile for the exact Qwen3.5/Ornith MLX model.

The timings are inclusive kernel measurements, not additive end-to-end shares.
Synchronization after every isolated component adds a measured empty-sync
overhead; the raw data retains that overhead and all repetitions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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


def measure(
    fn: Callable[[], Any], *, warmups: int, repeats: int
) -> tuple[list[dict[str, float]], Any]:
    import mlx.core as mx

    output = None
    for _ in range(warmups):
        output = fn()
        mx.synchronize()
        del output
        output = None
        gc.collect()
        mx.clear_cache()

    records: list[dict[str, float]] = []
    for index in range(repeats):
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = fn()
        mx.synchronize()
        elapsed = time.perf_counter() - started
        records.append(
            {
                "index": index,
                "seconds": round(elapsed, 6),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 6),
            }
        )
        del output
        output = None
        gc.collect()
        mx.clear_cache()
    return records, output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="1,512,2048,8192")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    environment_bytes = args.environment.read_bytes()
    checkpoint_bytes = args.checkpoint_manifest.read_bytes()
    environment = json.loads(environment_bytes)
    checkpoint = json.loads(checkpoint_bytes)
    if not checkpoint.get("manifest_sha256"):
        raise SystemExit("checkpoint manifest has no manifest_sha256")

    mx.random.seed(args.seed)
    model, _ = load(str(args.model.expanduser()), lazy=False)
    mx.synchronize()
    layers = model.language_model.layers
    gdn_layer = next(layer for layer in layers if layer.is_linear)
    attention_layer = next(layer for layer in layers if not layer.is_linear)
    moe_layer = gdn_layer
    hidden_size = model.language_model.model.layers[0].input_layernorm.weight.size
    contexts = [int(value) for value in args.contexts.split(",")]

    sync_records: list[dict[str, float]] = []
    for index in range(args.warmups + args.repeats):
        mx.synchronize()
        started = time.perf_counter()
        mx.synchronize()
        elapsed = time.perf_counter() - started
        if index >= args.warmups:
            sync_records.append(
                {"index": index - args.warmups, "seconds": round(elapsed, 9)}
            )

    results: list[dict[str, Any]] = []
    for context in contexts:
        x = mx.random.normal(shape=(1, context, hidden_size)).astype(mx.bfloat16)
        causal_mask = create_attention_mask(x, None)
        cases: list[tuple[str, Callable[[], Any]]] = [
            ("gated_delta_net", lambda x=x: gdn_layer.linear_attn(x, None, None)),
            (
                "full_attention",
                lambda x=x, mask=causal_mask: attention_layer.self_attn(
                    x, mask, None
                ),
            ),
            ("moe_block", lambda x=x: moe_layer.mlp(x)),
            ("lm_head_all_positions", lambda x=x: model.language_model.lm_head(x)),
            (
                "lm_head_last_position",
                lambda x=x: model.language_model.lm_head(x[:, -1:, :]),
            ),
        ]
        for component, fn in cases:
            records, _ = measure(fn, warmups=args.warmups, repeats=args.repeats)
            results.append(
                {
                    "context_tokens": context,
                    "component": component,
                    "records": records,
                    "summary": summarize([record["seconds"] for record in records]),
                    "peak_memory_gb": summarize(
                        [record["peak_memory_gb"] for record in records]
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "context_tokens": context,
                        "component": component,
                        "summary": results[-1]["summary"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del x, causal_mask
        gc.collect()
        mx.clear_cache()

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "repository_commit": environment.get("git", {}).get("commit"),
            "repository_dirty": environment.get("git", {}).get("dirty"),
            "environment_capture_sha256": hashlib.sha256(environment_bytes).hexdigest(),
            "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
            "checkpoint_revision": checkpoint.get("revision"),
        },
        "correctness": {
            "status": "not-compared",
            "reference": "component timings only; model output is not generated",
        },
        "engine": {
            "name": "mlx-lm-component-profile",
            "runtime_version": importlib.metadata.version("mlx-lm"),
            "mlx_version": importlib.metadata.version("mlx"),
            "runtime_manifest": "runtime-mlx-lm-0.31.3.json",
        },
        "protocol": {
            "semantic_mode": "baseline",
            "batch_size": 1,
            "sampling": None,
            "contexts": contexts,
            "warmups": args.warmups,
            "trials_requested": args.repeats,
            "cache_state": "fresh component inputs; no persistent generation cache",
            "input": "fixed-seed random BF16 activations",
            "synchronization": "mx.synchronize after each isolated component",
        },
        "empty_sync_overhead": summarize(
            [record["seconds"] for record in sync_records]
        ),
        "measurements": results,
        "summary": {
            "successful_trials": sum(
                len(item["records"]) for item in results
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
