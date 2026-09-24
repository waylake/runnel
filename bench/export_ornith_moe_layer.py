#!/usr/bin/env python3
"""Export one real Ornith MoE layer to the native probe's temporary binary format.

The safetensors files stay outside the repository. Only the requested layer is
materialized under --output, which should point at an ignored cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"RNLMOE01"
HEADER = struct.Struct("<8s8I")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SafeTensorStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: dict[str, tuple[Path, tuple[int, int], tuple[int, ...], str]] = {}
        for path in sorted(root.glob("*.safetensors")):
            with path.open("rb") as stream:
                stream.seek(0)
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
            for name, metadata in header.items():
                if name == "__metadata__":
                    continue
                offsets = tuple(metadata["data_offsets"])
                shape = tuple(metadata["shape"])
                dtype = metadata["dtype"]
                self.entries[name] = (path, offsets, shape, dtype)

    def read(self, name: str) -> tuple[np.ndarray, str]:
        try:
            path, offsets, shape, dtype = self.entries[name]
        except KeyError as error:
            raise SystemExit(f"tensor is missing: {name}") from error
        with path.open("rb") as stream:
            stream.seek(0)
            header_size = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(8 + header_size + offsets[0])
            raw = stream.read(offsets[1] - offsets[0])
        numpy_dtype = {
            "U32": "<u4",
            "BF16": "<u2",
            "F32": "<f4",
        }.get(dtype)
        if numpy_dtype is None:
            raise SystemExit(f"unsupported tensor dtype {dtype} for {name}")
        array = np.frombuffer(raw, dtype=numpy_dtype).copy().reshape(shape)
        return array, sha256_bytes(raw)


def float_to_bf16(value: float) -> np.uint16:
    bits = struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]
    if ((bits >> 23) & 0xFF) == 0xFF:
        return np.uint16(bits >> 16)
    rounding = 0x7FFF + ((bits >> 16) & 1)
    return np.uint16((bits + rounding) >> 16)


def parse_experts(text: str) -> list[int]:
    values = [int(token) for token in text.split(",")]
    if len(values) != 8 or len(set(values)) != 8 or any(value < 0 for value in values):
        raise SystemExit("--experts must contain eight unique non-negative indices")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", default="3,17,29,42,50,61,7,55")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    selected = parse_experts(args.experts)
    store = SafeTensorStore(args.model)
    prefix = f"language_model.model.layers.{args.layer}.mlp.switch_mlp"
    names = {
        "gate_weight": f"{prefix}.gate_proj.weight",
        "gate_scales": f"{prefix}.gate_proj.scales",
        "gate_biases": f"{prefix}.gate_proj.biases",
        "up_weight": f"{prefix}.up_proj.weight",
        "up_scales": f"{prefix}.up_proj.scales",
        "up_biases": f"{prefix}.up_proj.biases",
        "down_weight": f"{prefix}.down_proj.weight",
        "down_scales": f"{prefix}.down_proj.scales",
        "down_biases": f"{prefix}.down_proj.biases",
    }
    tensors: dict[str, np.ndarray] = {}
    tensor_hashes: dict[str, str] = {}
    for label, name in names.items():
        tensors[label], tensor_hashes[name] = store.read(name)

    gate_weight = tensors["gate_weight"]
    up_weight = tensors["up_weight"]
    down_weight = tensors["down_weight"]
    if gate_weight.ndim != 3 or up_weight.shape != gate_weight.shape:
        raise SystemExit("gate/up tensor shapes do not agree")
    if down_weight.ndim != 3:
        raise SystemExit("down tensor is not an expert matrix")
    total_experts, gate_rows, gate_packed = gate_weight.shape
    if down_weight.shape[0] != total_experts or down_weight.shape[1] != gate_rows * 4:
        raise SystemExit("down tensor shape is inconsistent with gate/up rows")
    if any(index >= total_experts for index in selected):
        raise SystemExit("selected expert is outside the checkpoint")
    bits = 4 if gate_packed == 256 else 8 if gate_packed == 512 else 0
    if bits == 0:
        raise SystemExit(f"cannot infer quantization bits from packed width {gate_packed}")
    down_packed = down_weight.shape[2]
    if down_packed != (512 * bits // 32):
        raise SystemExit("down packed width is inconsistent with the selected bits")

    gate_up_weight = np.concatenate((up_weight, gate_weight), axis=1)
    gate_up_scales = np.concatenate(
        (tensors["up_scales"], tensors["gate_scales"]), axis=1
    )
    gate_up_biases = np.concatenate(
        (tensors["up_biases"], tensors["gate_biases"]), axis=1
    )

    rng = np.random.default_rng(0x4F524E495448)
    x = rng.uniform(-1.0, 1.0, size=2048).astype(np.float32)
    x_bits = np.fromiter(
        (float_to_bf16(float(value)) for value in x), dtype="<u2", count=2048
    )
    router = rng.uniform(0.5, 1.5, size=8).astype(np.float32)
    router /= np.sum(router, dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(
            HEADER.pack(
                MAGIC,
                1,
                bits,
                total_experts,
                8,
                gate_rows * 2,
                down_weight.shape[1],
                2048,
                gate_rows,
            )
        )
        stream.write(np.asarray(selected, dtype="<u4").tobytes())
        for array, dtype in (
            (gate_up_weight, "<u4"),
            (gate_up_scales, "<u2"),
            (gate_up_biases, "<u2"),
            (down_weight, "<u4"),
            (tensors["down_scales"], "<u2"),
            (tensors["down_biases"], "<u2"),
            (x_bits, "<u2"),
            (router, "<f4"),
        ):
            stream.write(np.asarray(array, dtype=dtype).tobytes(order="C"))

    manifest = {
        "schema_version": 1,
        "model_name": args.model.name,
        "layer": args.layer,
        "bits": bits,
        "total_experts": total_experts,
        "selected_experts": selected,
        "gate_up_shape": list(gate_up_weight.shape),
        "down_shape": list(down_weight.shape),
        "tensor_sha256": tensor_hashes,
        "output_sha256": sha256_bytes(args.output.read_bytes()),
        "output_bytes": args.output.stat().st_size,
        "note": "temporary derived artifact; model weights are not committed",
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
