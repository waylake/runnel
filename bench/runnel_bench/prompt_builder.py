"""Build fixed-token prompts with the target checkpoint tokenizer.

This module intentionally imports Transformers lazily. The core fingerprint and
environment tools remain standard-library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .workload import get_workload


def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def _repeat_to_ids(tokenizer, text: str, minimum: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("filler source produced no tokens")
    repeats = (minimum // len(ids)) + 1
    return (ids * repeats)[:minimum]


def build_prompt(
    tokenizer,
    workload_name: str,
    target_tokens: int,
    *,
    enable_thinking: bool,
) -> dict[str, Any]:
    workload = get_workload(workload_name)
    if target_tokens < 64:
        raise ValueError("target_tokens must be at least 64")

    generation_prefix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
    prefix = (
        "<|im_start|>system\n"
        + workload.system_prompt
        + "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = (
        "\n\nTask:\n"
        + workload.turns[0]
        + "<|im_end|>\n<|im_start|>assistant\n"
        + generation_prefix
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    fixed = len(prefix_ids) + len(suffix_ids)
    if fixed >= target_tokens:
        raise ValueError(
            f"target_tokens={target_tokens} is too small; template requires {fixed}"
        )

    source_ids = tokenizer.encode(workload.source_text(), add_special_tokens=False)
    filler_ids = _repeat_to_ids(tokenizer, workload.source_text(), target_tokens - fixed)
    token_ids = prefix_ids + filler_ids + suffix_ids
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    round_trip = tokenizer.encode(text, add_special_tokens=False)
    if round_trip != token_ids:
        mismatch = next(
            (i for i, pair in enumerate(zip(token_ids, round_trip)) if pair[0] != pair[1]),
            min(len(token_ids), len(round_trip)),
        )
        raise ValueError(
            f"tokenizer round trip is unstable at index {mismatch}: "
            f"{len(token_ids)} vs {len(round_trip)} tokens"
        )

    return {
        "schema_version": 1,
        "workload": workload.as_manifest(),
        "target_prompt_tokens": target_tokens,
        "actual_prompt_tokens": len(token_ids),
        "enable_thinking": enable_thinking,
        "tokenizer_class": type(tokenizer).__name__,
        "token_ids": token_ids,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fixed-token Runnel prompt")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", default="python-multi-file-v1")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    tokenizer = _load_tokenizer(args.model.expanduser())
    result = build_prompt(
        tokenizer,
        args.workload,
        args.target_tokens,
        enable_thinking=args.thinking,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
