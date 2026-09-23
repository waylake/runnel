#!/usr/bin/env python3
"""Run the stock mlx-lm reference against a fixed-token prompt artifact.

Execute with the Python environment that contains mlx-lm, not the system
Python. The production routed projection transform is imported from Runnel;
failed research-only variants remain local to this adapter.
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


def _fuse_moe_projections(
    model, *, fuse_routed_gate_up: bool, fuse_shared_gate_up: bool
) -> tuple[int, int]:
    """Replace separate MoE gate/up projections with packed projections."""
    import gc
    import sys

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.activations import swiglu

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from runnel.mlx import fuse_routed_gate_up

    class FusedQuantizedSharedMLP(nn.Module):
        def __init__(self, old):
            super().__init__()
            gate = old.gate_proj
            up = old.up_proj
            if not (
                isinstance(gate, type(up))
                and gate.bits == up.bits
                and gate.group_size == up.group_size
                and gate.mode == up.mode
            ):
                raise ValueError("shared gate/up quantization formats differ")
            if ("bias" in gate) or ("bias" in up):
                raise ValueError("shared module biases are not expected")

            self.weight = mx.concatenate([up["weight"], gate["weight"]], axis=0)
            self.scales = mx.concatenate([up["scales"], gate["scales"]], axis=0)
            up_biases = up.get("biases")
            gate_biases = gate.get("biases")
            if (up_biases is None) != (gate_biases is None):
                raise ValueError("shared gate/up affine-bias formats differ")
            self.biases = (
                None
                if up_biases is None
                else mx.concatenate([up_biases, gate_biases], axis=0)
            )
            self.group_size = gate.group_size
            self.bits = gate.bits
            self.mode = gate.mode
            self.down_proj = old.down_proj
            self.freeze()
            mx.eval(self.parameters())

        def __call__(self, x):
            fused = mx.quantized_matmul(
                x,
                self["weight"],
                scales=self["scales"],
                biases=self.get("biases"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            x_up, x_gate = mx.split(fused, 2, axis=-1)
            return self.down_proj(swiglu(x_gate, x_up))

    routed_count = fuse_routed_gate_up(model) if fuse_routed_gate_up else 0
    shared_count = 0
    if fuse_shared_gate_up:
        for layer in model.language_model.layers:
            old = layer.mlp.shared_expert
            if not hasattr(old.gate_proj, "bits"):
                raise ValueError("shared gate/up are not quantized as expected")
            fused = FusedQuantizedSharedMLP(old)
            layer.mlp.shared_expert = fused
            shared_count += 1
            del old, fused
            gc.collect()
            mx.clear_cache()
    return routed_count, shared_count


def _use_last_token_lm_head(model) -> int:
    """Project only the final prefill position through the LM head."""
    import mlx.nn as nn

    class LastTokenLMHead(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
            self.freeze()

        def __call__(self, x):
            return self.head(x[:, -1:, :])

    text_model = model.language_model
    text_model.lm_head = LastTokenLMHead(text_model.lm_head)
    return 1


def _enable_core_chunked_gdn(min_tokens: int) -> int:
    """Route long GDN prefill through MLX core's M1 SIMD-group chunk kernel."""
    import mlx.core as mx
    import mlx_lm.models.qwen3_5 as qwen
    from mlx_lm.models.gated_delta import compute_g

    if not hasattr(mx.fast, "gated_delta_update"):
        raise RuntimeError(
            "this MLX build has no mx.fast.gated_delta_update; use the pinned nightly"
        )
    original = qwen.gated_delta_update

    def patched(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state=None,
        mask=None,
        use_kernel=True,
    ):
        if q.shape[1] < min_tokens:
            return original(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                mask,
                use_kernel,
            )
        beta = mx.sigmoid(b)
        gamma = compute_g(A_log, a, dt_bias)
        if state is None:
            B, _, _, _ = q.shape
            Hv, Dv = v.shape[-2:]
            _, _, _, Dk = k.shape
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        return mx.fast.gated_delta_update(
            q,
            k,
            v,
            gamma,
            beta,
            initial_state=state,
            mask=mask,
        )

    qwen.gated_delta_update = patched
    return 1


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
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument(
        "--runtime-manifest", default="runtime-mlx-lm-0.31.3.json"
    )
    parser.add_argument("--fuse-routed-gate-up", action="store_true")
    parser.add_argument("--fuse-shared-gate-up", action="store_true")
    parser.add_argument("--last-token-lm-head", action="store_true")
    parser.add_argument("--gdn-core-chunked", action="store_true")
    parser.add_argument("--gdn-core-min-tokens", type=int, default=16)
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

    fused_routed_layers = 0
    fused_shared_layers = 0
    fusion_started = time.perf_counter()
    if args.fuse_routed_gate_up or args.fuse_shared_gate_up:
        fused_routed_layers, fused_shared_layers = _fuse_moe_projections(
            model,
            fuse_routed_gate_up=args.fuse_routed_gate_up,
            fuse_shared_gate_up=args.fuse_shared_gate_up,
        )
    last_token_lm_head_count = 0
    if args.last_token_lm_head:
        last_token_lm_head_count = _use_last_token_lm_head(model)
    gdn_core_chunked_count = 0
    if args.gdn_core_chunked:
        gdn_core_chunked_count = _enable_core_chunked_gdn(
            args.gdn_core_min_tokens
        )
    if (
        args.fuse_routed_gate_up
        or args.fuse_shared_gate_up
        or args.last_token_lm_head
        or args.gdn_core_chunked
    ):
        mx.synchronize()
    fusion_seconds = time.perf_counter() - fusion_started

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
    correctness = {
        "status": "reference",
        "reference": "stock mlx-lm greedy token stream",
    }
    if args.reference_result:
        reference = json.loads(args.reference_result.read_text(encoding="utf-8"))
        reference_trials = [
            item for item in reference["trials"] if item["phase"] == "trial"
        ]
        if not reference_trials or "token_ids" not in reference_trials[0]:
            raise SystemExit("reference result has no measured token IDs")
        reference_ids = reference_trials[0]["token_ids"]
        matched = all(item["token_ids"] == reference_ids for item in measured)
        correctness = {
            "status": "matched" if matched else "mismatched",
            "reference": args.reference_result.name,
            "first_differing_token_index": next(
                (
                    index
                    for index, pair in enumerate(
                        zip(measured[0]["token_ids"], reference_ids)
                    )
                    if pair[0] != pair[1]
                ),
                None,
            )
            if measured and not matched
            else None,
        }
    elif (
        args.fuse_routed_gate_up
        or args.fuse_shared_gate_up
        or args.last_token_lm_head
        or args.gdn_core_chunked
    ):
        correctness = {
            "status": "not-compared",
            "reference": "pass --reference-result for greedy token parity",
        }

    engine_variants = []
    if args.fuse_routed_gate_up:
        engine_variants.append("fused-routed-gate-up")
    if args.fuse_shared_gate_up:
        engine_variants.append("fused-shared-gate-up")
    if args.last_token_lm_head:
        engine_variants.append("last-token-lm-head")
    if args.gdn_core_chunked:
        engine_variants.append("gdn-core-chunked")
    engine_name = "mlx-lm"
    if engine_variants:
        engine_name += "+" + "+".join(engine_variants)

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
        "correctness": correctness,
        "engine": {
            "name": engine_name,
            "runtime_version": importlib.metadata.version("mlx-lm"),
            "mlx_version": importlib.metadata.version("mlx"),
            "adapter": "direct-stream-v1",
            "runtime_manifest": args.runtime_manifest,
        },
        "model_load": {
            "seconds": round(load_seconds, 6),
            "fusion_setup_seconds": round(fusion_seconds, 6),
            "fused_routed_gate_up_layers": fused_routed_layers,
            "fused_shared_gate_up_layers": fused_shared_layers,
            "last_token_lm_head_count": last_token_lm_head_count,
            "gdn_core_chunked_count": gdn_core_chunked_count,
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
            "experiment": {
                "fused_routed_gate_up": args.fuse_routed_gate_up,
                "fused_routed_layers": fused_routed_layers,
                "fused_shared_gate_up": args.fuse_shared_gate_up,
                "fused_shared_layers": fused_shared_layers,
                "last_token_lm_head": args.last_token_lm_head,
                "last_token_lm_head_count": last_token_lm_head_count,
                "gdn_core_chunked": args.gdn_core_chunked,
                "gdn_core_min_tokens": args.gdn_core_min_tokens,
                "gdn_core_chunked_count": gdn_core_chunked_count,
            },
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
