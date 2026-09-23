"""Minimal streaming CLI for the experimental Runnel MLX execution path."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runnel")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Stream a completion")
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--system-prompt")
    generate.add_argument("--max-tokens", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument(
        "--stock-mlp",
        action="store_true",
        help="Disable Runnel's packed routed gate/up optimization",
    )
    return parser


def _generate(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from .mlx import fuse_routed_gate_up

    started = time.perf_counter()
    model, tokenizer = load(str(args.model.expanduser()), lazy=False)
    mx.synchronize()
    load_seconds = time.perf_counter() - started

    mx.random.seed(args.seed)
    fused_layers = 0
    if not args.stock_mlp:
        fused_layers = fuse_routed_gate_up(model)
        mx.synchronize()

    if args.system_prompt:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": args.prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = args.prompt

    started = time.perf_counter()
    last = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=args.temperature),
    ):
        last = response
        if response.text:
            print(response.text, end="", flush=True)
    print(file=sys.stderr)
    if last is not None:
        print(
            f"runnel: load={load_seconds:.3f}s fused_layers={fused_layers} "
            f"prompt={last.prompt_tokens} output={last.generation_tokens} "
            f"decode_tps={last.generation_tps:.2f} total={time.perf_counter()-started:.3f}s",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
