"""Streaming benchmark adapter for OpenAI-compatible local runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class BenchmarkError(RuntimeError):
    pass


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "mad": None, "min": None, "max": None}
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "n": len(ordered),
        "median": round(median, 6),
        "mad": round(statistics.median(abs(value - median) for value in ordered), 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
    }


def _request_headers(api_key_env: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key_env:
        value = os.environ.get(api_key_env)
        if not value:
            raise BenchmarkError(f"required API-key environment variable is unset: {api_key_env}")
        headers["Authorization"] = f"Bearer {value}"
    return headers


def _delta_is_token(delta: dict[str, Any]) -> bool:
    return any(
        delta.get(key) not in (None, "", [], {})
        for key in ("content", "reasoning_content", "tool_calls")
    )


def run_stream_once(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    expected_prompt_tokens: int,
    max_tokens: int,
    temperature: float,
    api_key_env: str | None,
    timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers=_request_headers(api_key_env),
        method="POST",
    )
    started = time.perf_counter()
    first_event: float | None = None
    first_token: float | None = None
    last_token: float | None = None
    usage: dict[str, Any] = {}
    output = bytearray()
    event_count = 0
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {exc.code}: {detail}") from exc
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            now = time.perf_counter()
            if first_event is None:
                first_event = now
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            event_count += 1
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"invalid SSE JSON at event {event_count}") from exc
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if _delta_is_token(delta) and first_token is None:
                    first_token = now
                if _delta_is_token(delta):
                    last_token = now
                for key in ("content", "reasoning_content"):
                    value = delta.get(key)
                    if isinstance(value, str):
                        output.extend(value.encode("utf-8"))
    ended = time.perf_counter()

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None:
        raise BenchmarkError("stream ended without usage.prompt_tokens")
    if prompt_tokens != expected_prompt_tokens:
        raise BenchmarkError(
            f"server retokenized prompt as {prompt_tokens}, expected {expected_prompt_tokens}"
        )
    if first_token is None:
        raise BenchmarkError("stream produced no token event")
    if last_token is None or completion_tokens is None:
        decode_tps = None
    elif completion_tokens <= 1 or last_token <= first_token:
        decode_tps = None
    else:
        decode_tps = (completion_tokens - 1) / (last_token - first_token)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "first_event_s": round(first_event - started, 6) if first_event else None,
        "ttft_s": round(first_token - started, 6) if first_token else None,
        "last_token_s": round(last_token - started, 6) if last_token else None,
        "end_to_end_s": round(ended - started, 6),
        "observed_decode_tokens_per_second": round(decode_tps, 6)
        if decode_tps
        else None,
        "end_to_end_output_tokens_per_second": round(
            completion_tokens / (ended - started), 6
        )
        if completion_tokens
        else None,
        "stream_event_count": event_count,
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "usage": usage,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    environment_bytes = args.environment.read_bytes()
    checkpoint_bytes = args.checkpoint_manifest.read_bytes()
    environment = json.loads(environment_bytes)
    checkpoint = json.loads(checkpoint_bytes)
    if not checkpoint.get("manifest_sha256"):
        raise BenchmarkError("checkpoint manifest has no manifest_sha256")

    prompt_artifact = json.loads(args.prompt.read_text(encoding="utf-8"))
    if prompt_artifact["actual_prompt_tokens"] != prompt_artifact["target_prompt_tokens"]:
        raise BenchmarkError("prompt artifact is not fixed-length")
    prompt = prompt_artifact["text"]
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_artifact["text_sha256"]:
        raise BenchmarkError("prompt artifact digest mismatch")

    trials: list[dict[str, Any]] = []
    total = args.warmups + args.trials
    for index in range(total):
        phase = "warmup" if index < args.warmups else "trial"
        trial_index = index - args.warmups if phase == "trial" else index
        started = datetime.now(timezone.utc).isoformat()
        try:
            result = run_stream_once(
                endpoint=args.endpoint,
                model=args.model,
                prompt=prompt,
                expected_prompt_tokens=prompt_artifact["actual_prompt_tokens"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                api_key_env=args.api_key_env,
                timeout=args.timeout,
            )
            record = {
                "phase": phase,
                "index": trial_index,
                "started_at_utc": started,
                "status": "ok",
                **result,
            }
        except Exception as exc:  # preserve failed trials
            record = {
                "phase": phase,
                "index": trial_index,
                "started_at_utc": started,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        trials.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.stop_on_error and record["status"] == "error":
            break

    measured = [item for item in trials if item["phase"] == "trial" and item["status"] == "ok"]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "repository_commit": environment.get("git", {}).get("commit"),
            "repository_dirty": environment.get("git", {}).get("dirty"),
            "environment_capture_sha256": hashlib.sha256(environment_bytes).hexdigest(),
            "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
            "checkpoint_revision": checkpoint.get("revision"),
        },
        "engine": {
            "name": args.engine_name,
            "adapter": "openai-compatible-stream-v1",
            "endpoint": args.endpoint,
            "model": args.model,
            "runtime_version": args.runtime_version,
        },
        "protocol": {
            "batch_size": 1,
            "semantic_mode": "exact",
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
            },
            "prompt_tokens": prompt_artifact["actual_prompt_tokens"],
            "prompt_sha256": prompt_artifact["text_sha256"],
            "workload": prompt_artifact["workload"],
            "enable_thinking": prompt_artifact["enable_thinking"],
            "max_output_tokens": args.max_tokens,
            "warmups": args.warmups,
            "trials_requested": args.trials,
            "cache_state": args.cache_state,
        },
        "trials": trials,
        "summary": {
            "successful_trials": len(measured),
            "ttft_s": _summary([x["ttft_s"] for x in measured if x["ttft_s"] is not None]),
            "end_to_end_s": _summary(
                [x["end_to_end_s"] for x in measured if x["end_to_end_s"] is not None]
            ),
            "observed_decode_tokens_per_second": _summary(
                [
                    x["observed_decode_tokens_per_second"]
                    for x in measured
                    if x["observed_decode_tokens_per_second"] is not None
                ]
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible streaming benchmark")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-name", required=True)
    parser.add_argument("--runtime-version", default="unknown")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--api-key-env")
    parser.add_argument("--cache-state", default="runtime-default")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if result["summary"]["successful_trials"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
