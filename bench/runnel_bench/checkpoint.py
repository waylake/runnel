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


def _active_decode_estimate(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    hidden = int(text["hidden_size"])
    vocab = int(text["vocab_size"])
    layers = int(text["num_hidden_layers"])
    layer_types = list(text["layer_types"])
    experts = int(text["num_experts"])
    top_k = int(text["num_experts_per_tok"])
    moe_intermediate = int(text["moe_intermediate_size"])
    shared_intermediate = int(text.get("shared_expert_intermediate_size", 0))

    linear_count = sum(kind == "linear_attention" for kind in layer_types)
    full_count = sum(kind == "full_attention" for kind in layer_types)

    key_heads = int(text["linear_num_key_heads"])
    value_heads = int(text["linear_num_value_heads"])
    key_dim = int(text["linear_key_head_dim"])
    value_dim = int(text["linear_value_head_dim"])
    # The checkpoint stores separate Q/K (key_heads each) and V (value_heads).
    gdn_qkv = (2 * key_heads + value_heads) * key_dim * hidden
    gdn_z = value_heads * value_dim * hidden
    gdn_a = key_heads * key_dim * hidden
    gdn_b = key_heads * key_dim * hidden
    gdn_out = hidden * value_heads * value_dim
    gdn_per_layer = gdn_qkv + gdn_z + gdn_a + gdn_b + gdn_out

    attention_heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    head_dim = int(text["head_dim"])
    full_per_layer = hidden * head_dim * (attention_heads + 2 * kv_heads)
    full_per_layer += attention_heads * head_dim * hidden

    routed_per_layer = top_k * 3 * hidden * moe_intermediate
    shared_per_layer = 3 * hidden * shared_intermediate
    router_per_layer = hidden * experts + experts
    moe_active_per_layer = routed_per_layer + shared_per_layer + router_per_layer

    lm_head = hidden * vocab
    active_parameters = (
        linear_count * gdn_per_layer
        + full_count * full_per_layer
        + layers * moe_active_per_layer
        + lm_head
    )

    quantization = config.get("quantization", {})
    bits = int(quantization.get("bits", 16))
    group_size = int(quantization.get("group_size", 1) or 1)
    if bits == 8:
        # U32 stores four int8 values. Two BF16 affine values per group add
        # four bytes per group. This is an estimate, not a replacement for a
        # measured allocator/Metal working-set report.
        quantized_bytes_per_parameter = 1.0 + 4.0 / group_size
    else:
        quantized_bytes_per_parameter = bits / 8.0

    quantized_active_parameters = active_parameters - lm_head
    estimated_weight_bytes = (
        quantized_active_parameters * quantized_bytes_per_parameter + 2 * lm_head
    )
    nominal_bandwidth_gbps = 400.0
    return {
        "method": "analytic batch-1 active-weight estimate from config; excludes activations, cache, allocator slack, and unfused rereads",
        "active_parameters_estimate": active_parameters,
        "estimated_active_weight_bytes": round(estimated_weight_bytes),
        "estimated_400gbps_decode_tokens_per_second": round(
            nominal_bandwidth_gbps * 1_000_000_000 / estimated_weight_bytes, 2
        ),
        "components_per_layer": {
            "gated_delta_net_parameters": gdn_per_layer,
            "full_attention_parameters": full_per_layer,
            "active_routed_expert_parameters": routed_per_layer,
            "shared_expert_parameters": shared_per_layer,
            "router_parameters": router_per_layer,
        },
        "layer_counts": {
            "linear_attention": linear_count,
            "full_attention": full_count,
        },
        "lm_head_parameters": lm_head,
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
        "quantization": config.get("quantization", {}),
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
        "active_decode_estimate": _active_decode_estimate(config),
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
