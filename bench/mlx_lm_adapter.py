#!/usr/bin/env python3
"""Run the stock mlx-lm reference against a fixed-token prompt artifact.

Execute with the Python environment that contains mlx-lm, not the system
Python. The script intentionally imports no Runnel package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--kv-bits", type=int, choices=(None, 8, 6, 5, 4, 3, 2))
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt_artifact = json.loads(args.prompt.read_text(encoding="utf-8"))
    environment_bytes = args.environment.read_bytes()
    checkpoint_bytes = args.checkpoint_manifest.read_bytes()
    environment = json.loads(environment_bytes)
    checkpoint = json.loads(checkpoint_bytes)
    if not checkpoint.get("manifest_sha256"):
        raise SystemExit("checkpoint manifest has no manifest_sha256")
    expected_ids = prompt_artifact["token_ids"]
    if hashlib.sha256(prompt_artifact["text"].encode("utf-8")).hexdigest() != prompt_artifact["text_sha256"]:
        raise SystemExit("prompt artifact digest mismatch")

    load_started = time.perf_counter()
    model, tokenizer = load(str(args.model.expanduser()), lazy=False)
    mx.synchronize()
    load_seconds = time.perf_counter() - load_started
    prompt_ids = tokenizer.encode(prompt_artifact["text"], add_special_tokens=False)
    if prompt_ids != expected_ids:
        raise SystemExit("mlx tokenizer did not reproduce the prompt token IDs")

    kwargs: dict[str, Any] = {
        "sampler": make_sampler(temp=0.0),
        "max_kv_size": None,
    }
    if args.kv_bits is not None:
        kwargs.update(
            {
                "kv_bits": args.kv_bits,
                "kv_group_size": args.kv_group_size,
                "quantized_kv_start": args.quantized_kv_start,
            }
        )

    trials: list[dict[str, Any]] = []
    for index in range(args.warmups + args.trials):
        phase = "warmup" if index < args.warmups else "trial"
        trial_index = index - args.warmups if phase == "trial" else index
        mx.reset_peak_memory()
        token_ids: list[int] = []
        output = bytearray()
        first_token_s: float | None = None
        last_token_s: float | None = None
        prompt_tps = None
        generation_tps = None
        finish_reason = None
        started = time.perf_counter()
        for response in stream_generate(
            model,
            tokenizer,
            expected_ids,
            max_tokens=args.max_tokens,
            **kwargs,
        ):
            now = time.perf_counter()
            if first_token_s is None:
                first_token_s = now - started
            last_token_s = now - started
            token_ids.append(int(response.token))
            output.extend(response.text.encode("utf-8"))
            prompt_tps = response.prompt_tps
            generation_tps = response.generation_tps
            finish_reason = response.finish_reason
        mx.synchronize()
        ended = time.perf_counter()
        if first_token_s is None or last_token_s is None or not token_ids:
            raise SystemExit("mlx-lm produced no token")
        record = {
            "phase": phase,
            "index": trial_index,
            "status": "ok",
            "prompt_tokens": len(expected_ids),
            "completion_tokens": len(token_ids),
            "ttft_s": round(first_token_s, 6),
            "end_to_end_s": round(ended - started, 6),
            "prompt_tokens_per_second": round(prompt_tps, 6),
            "reported_generation_tokens_per_second": round(generation_tps, 6),
            "observed_decode_tokens_per_second": round(
                (len(token_ids) - 1) / (last_token_s - first_token_s), 6
            )
            if len(token_ids) > 1 and last_token_s > first_token_s
            else None,
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 6),
            "finish_reason": finish_reason,
            "token_ids": token_ids,
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "output_bytes": len(output),
            "output_sha256": hashlib.sha256(output).hexdigest(),
        }
        trials.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "token_ids"}), flush=True)

    measured = [item for item in trials if item["phase"] == "trial"]
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
        "engine": {
            "name": "mlx-lm",
            "runtime_version": importlib.metadata.version("mlx-lm"),
            "mlx_version": importlib.metadata.version("mlx"),
            "adapter": "direct-stream-v1",
        },
        "model_load": {
            "seconds": round(load_seconds, 6),
            "checkpoint": prompt_artifact["workload"],
        },
        "protocol": {
            "semantic_mode": "exact",
            "batch_size": 1,
            "sampling": {"temperature": 0.0, "seed": 0},
            "prompt_tokens": len(expected_ids),
            "prompt_sha256": prompt_artifact["text_sha256"],
            "workload": prompt_artifact["workload"],
            "enable_thinking": prompt_artifact["enable_thinking"],
            "max_output_tokens": args.max_tokens,
            "warmups": args.warmups,
            "trials_requested": args.trials,
            "cache_state": "new prompt cache per generation",
            "kv": {
                "bits": args.kv_bits,
                "group_size": args.kv_group_size if args.kv_bits is not None else None,
                "quantized_start": args.quantized_kv_start if args.kv_bits is not None else None,
            },
        },
        "trials": trials,
        "summary": {
            "ttft_s": summarize([item["ttft_s"] for item in measured]),
            "end_to_end_s": summarize([item["end_to_end_s"] for item in measured]),
            "reported_generation_tokens_per_second": summarize(
                [item["reported_generation_tokens_per_second"] for item in measured]
            ),
            "peak_memory_gb": summarize([item["peak_memory_gb"] for item in measured]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
