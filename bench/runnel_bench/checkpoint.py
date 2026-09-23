"""Safetensors checkpoint audit and content fingerprinting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
AUDIT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"invalid safetensors header in {path.name}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size <= 0 or header_size > 100 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header size in {path.name}")
        return json.loads(handle.read(header_size))


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shape_product(shape: Iterable[int]) -> int:
    return math.prod(shape)


def _summarize_quantization(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("quantization", {})
    summary = {
        key: raw[key] for key in ("mode", "bits", "group_size") if key in raw
    }
    overrides = [
        {"tensor": name, **value}
        for name, value in raw.items()
        if isinstance(value, dict)
    ]
    unique = {
        json.dumps(
            {key: value for key, value in item.items() if key != "tensor"},
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in overrides
    }
    summary["per_tensor_override_count"] = len(overrides)
    summary["unique_override_configs"] = [json.loads(item) for item in sorted(unique)]
    return summary


def _active_decode_estimate(
    config: dict[str, Any], tensor_headers: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    text = config.get("text_config", config)
    num_layers = int(text["num_hidden_layers"])
    num_experts = int(text["num_experts"])
    experts_per_token = int(text["num_experts_per_tok"])
    layer_types = list(text["layer_types"])
    quantization = config.get("quantization", {})
    default_bits = int(quantization.get("bits", 16))
    default_group_size = int(quantization.get("group_size", 1) or 1)
    nominal_bandwidth_gbps = 400.0

    def quantization_spec(name: str) -> tuple[int, int]:
        override = quantization.get(name, {})
        return (
            int(override.get("bits", default_bits)),
            int(override.get("group_size", default_group_size) or 1),
        )

    def logical_numel(name: str) -> int:
        metadata = tensor_headers[name]
        elements = _shape_product(metadata["shape"])
        bits, _ = quantization_spec(name)
        return elements * (32 // bits) if metadata["dtype"] == "U32" else elements

    def active_bytes(name: str) -> int:
        metadata = tensor_headers[name]
        parameters = logical_numel(name)
        if metadata["dtype"] == "U32":
            bits, group_size = quantization_spec(name)
            # Packed values plus BF16 scale and bias for each affine group.
            return round(parameters * (bits / 8.0 + 4.0 / group_size))
        return parameters * 2

    totals = {
        "gated_delta_net_parameters": 0,
        "full_attention_parameters": 0,
        "active_routed_expert_parameters": 0,
        "shared_expert_parameters": 0,
        "router_parameters": 0,
        "layer_norm_parameters": 0,
    }
    active_parameter_count = 0
    active_weight_bytes = 0

    for layer_index in range(num_layers):
        prefix = f"language_model.model.layers.{layer_index}."
        names = [name for name in tensor_headers if name.startswith(prefix)]
        attention_names = []
        for name in names:
            if ".linear_attn." in name or ".self_attn." in name:
                if name.endswith((".scales", ".biases")):
                    continue
                if name.endswith((".weight", ".A_log", ".dt_bias")):
                    attention_names.append(name)
            elif name.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight")):
                totals["layer_norm_parameters"] += logical_numel(name)
                active_parameter_count += logical_numel(name)
                active_weight_bytes += active_bytes(name)

        attention_parameters = sum(logical_numel(name) for name in attention_names)
        active_parameter_count += attention_parameters
        active_weight_bytes += sum(active_bytes(name) for name in attention_names)
        key = (
            "gated_delta_net_parameters"
            if layer_types[layer_index] == "linear_attention"
            else "full_attention_parameters"
        )
        totals[key] += attention_parameters

        routed_names = [
            name
            for name in names
            if ".switch_mlp." in name and name.endswith(".weight")
        ]
        routed_parameters = (
            sum(logical_numel(name) // num_experts for name in routed_names)
            * experts_per_token
        )
        totals["active_routed_expert_parameters"] += routed_parameters
        active_parameter_count += routed_parameters
        active_weight_bytes += round(
            sum(active_bytes(name) for name in routed_names)
            / num_experts
            * experts_per_token
        )

        shared_names = [
            name
            for name in names
            if ".shared_expert." in name and name.endswith(".weight")
        ]
        shared_parameters = sum(logical_numel(name) for name in shared_names)
        totals["shared_expert_parameters"] += shared_parameters
        active_parameter_count += shared_parameters
        active_weight_bytes += sum(active_bytes(name) for name in shared_names)

        router_names = [
            name
            for name in names
            if name.endswith((".mlp.gate.weight", ".mlp.shared_expert_gate.weight"))
        ]
        router_parameters = sum(logical_numel(name) for name in router_names)
        totals["router_parameters"] += router_parameters
        active_parameter_count += router_parameters
        active_weight_bytes += sum(active_bytes(name) for name in router_names)

    final_norm_name = "language_model.model.norm.weight"
    final_norm_parameters = logical_numel(final_norm_name)
    totals["layer_norm_parameters"] += final_norm_parameters
    active_parameter_count += final_norm_parameters
    active_weight_bytes += active_bytes(final_norm_name)

    lm_head_name = "language_model.lm_head.weight"
    lm_head_parameters = logical_numel(lm_head_name)
    active_parameter_count += lm_head_parameters
    active_weight_bytes += active_bytes(lm_head_name)

    linear_count = sum(kind == "linear_attention" for kind in layer_types)
    full_count = sum(kind == "full_attention" for kind in layer_types)
    return {
        "method": "header-derived batch-1 active-weight estimate; excludes activations, cache, allocator slack, and unfused rereads",
        "active_parameters_estimate": active_parameter_count,
        "estimated_active_weight_bytes": active_weight_bytes,
        "estimated_400gbps_decode_tokens_per_second": round(
            nominal_bandwidth_gbps * 1_000_000_000 / active_weight_bytes, 2
        ),
        "totals_by_component": totals,
        "average_per_layer": {
            "gated_delta_net_parameters": round(
                totals["gated_delta_net_parameters"] / max(linear_count, 1)
            ),
            "full_attention_parameters": round(
                totals["full_attention_parameters"] / max(full_count, 1)
            ),
            "active_routed_expert_parameters": round(
                totals["active_routed_expert_parameters"] / num_layers
            ),
            "shared_expert_parameters": round(
                totals["shared_expert_parameters"] / num_layers
            ),
            "router_parameters": round(totals["router_parameters"] / num_layers),
        },
        "layer_counts": {
            "linear_attention": linear_count,
            "full_attention": full_count,
        },
        "lm_head_parameters": lm_head_parameters,
    }


def inspect_checkpoint(
    model_dir: Path,
    *,
    hash_weights: bool = True,
    revision: str | None = None,
) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = read_json(config_path)
    index = read_json(index_path)
    weight_map: dict[str, str] = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    files: list[dict[str, Any]] = []
    for name in AUDIT_FILES:
        path = model_dir / name
        if not path.is_file():
            continue
        files.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    tensor_headers: dict[str, dict[str, Any]] = {}
    for name in shard_names:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path) if hash_weights else None,
            }
        )
        tensor_headers.update(safetensors_header(path))

    dtype_counts: Counter[str] = Counter()
    stored_tensor_payload_bytes = 0
    shape_elements = 0
    name_groups: Counter[str] = Counter()
    matching_names: dict[str, list[str]] = {
        "mtp": [],
        "draft": [],
        "linear_attention": [],
        "full_attention": [],
        "routed_experts": [],
        "shared_expert": [],
        "vision": [],
    }
    for name, metadata in tensor_headers.items():
        if name == "__metadata__":
            continue
        dtype_counts[str(metadata["dtype"])] += 1
        shape = [int(x) for x in metadata["shape"]]
        shape_elements += _shape_product(shape)
        offsets = metadata["data_offsets"]
        stored_tensor_payload_bytes += int(offsets[1]) - int(offsets[0])
        if ".linear_attn." in name:
            name_groups["linear_attention"] += 1
            matching_names["linear_attention"].append(name)
        if ".self_attn." in name:
            name_groups["full_attention"] += 1
            matching_names["full_attention"].append(name)
        if ".switch_mlp." in name:
            name_groups["routed_experts"] += 1
            matching_names["routed_experts"].append(name)
        if ".shared_expert" in name:
            name_groups["shared_expert"] += 1
            matching_names["shared_expert"].append(name)
        if "mtp" in name.lower():
            matching_names["mtp"].append(name)
        if "draft" in name.lower():
            matching_names["draft"].append(name)
        if "vision" in name.lower() or "visual" in name.lower():
            matching_names["vision"].append(name)

    samples = {
        key: sorted(value)[:8]
        for key, value in matching_names.items()
        if value
    }
    text_config = config.get("text_config", config)
    architecture = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "layer_types": text_config.get("layer_types"),
        "num_experts": text_config.get("num_experts"),
        "num_experts_per_tok": text_config.get("num_experts_per_tok"),
        "shared_expert_intermediate_size": text_config.get(
            "shared_expert_intermediate_size"
        ),
        "moe_intermediate_size": text_config.get("moe_intermediate_size"),
        "full_attention_heads": text_config.get("num_attention_heads"),
        "full_attention_kv_heads": text_config.get("num_key_value_heads"),
        "full_attention_head_dim": text_config.get("head_dim"),
        "linear_attention_key_heads": text_config.get("linear_num_key_heads"),
        "linear_attention_value_heads": text_config.get("linear_num_value_heads"),
        "linear_attention_key_head_dim": text_config.get("linear_key_head_dim"),
        "linear_attention_value_head_dim": text_config.get("linear_value_head_dim"),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "vocab_size": text_config.get("vocab_size"),
        "mtp_num_hidden_layers_config": text_config.get("mtp_num_hidden_layers"),
    }

    identity = {
        "revision": revision,
        "files": [
            {"name": item["name"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
            for item in files
        ],
        "architecture": architecture,
    }
    manifest_digest = _canonical_digest(identity)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_name": model_dir.name,
        "revision": revision,
        "manifest_sha256": manifest_digest,
        "weights_hashed": hash_weights,
        "files": files,
        "architecture": architecture,
        "quantization": _summarize_quantization(config),
        "safetensors": {
            "tensor_count": len(tensor_headers),
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "shape_elements": shape_elements,
            "stored_tensor_payload_bytes": stored_tensor_payload_bytes,
            "index_total_size": index.get("metadata", {}).get("total_size"),
            "index_total_parameters": index.get("metadata", {}).get(
                "total_parameters"
            ),
        },
        "tensor_name_groups": dict(sorted(name_groups.items())),
        "tensor_name_samples": samples,
        "mtp_tensors_present": bool(matching_names["mtp"]),
        "active_decode_estimate": _active_decode_estimate(config, tensor_headers),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fingerprint a safetensors checkpoint")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision")
    parser.add_argument(
        "--skip-weight-hashes",
        action="store_true",
        help="Audit headers but mark shard SHA-256 values as null",
    )
    args = parser.parse_args(argv)
    result = inspect_checkpoint(
        args.model,
        hash_weights=not args.skip_weight_hashes,
        revision=args.revision,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
