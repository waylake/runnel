from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from runnel_bench.checkpoint import inspect_checkpoint, safetensors_header


class CheckpointTests(unittest.TestCase):
    def _make_checkpoint(self, root: Path) -> None:
        config = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
            "text_config": {
                "hidden_size": 8,
                "vocab_size": 32,
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
                "num_experts": 16,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 2,
                "linear_value_head_dim": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "mtp_num_hidden_layers": 1,
            },
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "generation_config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (root / "chat_template.jinja").write_text("template", encoding="utf-8")

        tensors: dict[str, dict[str, object]] = {
            "language_model.lm_head.weight": {
                "dtype": "U32",
                "shape": [32, 2],
                "data_offsets": [0, 4],
            }
        }
        offset = 4
        layer_types = config["text_config"]["layer_types"]
        for index, layer_type in enumerate(layer_types):
            prefix = f"language_model.model.layers.{index}."
            tensors[prefix + "input_layernorm.weight"] = {
                "dtype": "BF16",
                "shape": [8],
                "data_offsets": [offset, offset + 16],
            }
            offset += 16
            if layer_type == "linear_attention":
                tensors[prefix + "linear_attn.in_proj_qkv.weight"] = {
                    "dtype": "U32",
                    "shape": [8, 2],
                    "data_offsets": [offset, offset + 64],
                }
                offset += 64
            else:
                tensors[prefix + "self_attn.q_proj.weight"] = {
                    "dtype": "U32",
                    "shape": [8, 2],
                    "data_offsets": [offset, offset + 64],
                }
                offset += 64
            tensors[prefix + "mlp.gate.weight"] = {
                "dtype": "U32",
                "shape": [16, 2],
                "data_offsets": [offset, offset + 128],
            }
            offset += 128
            tensors[prefix + "mlp.switch_mlp.gate_proj.weight"] = {
                "dtype": "U32",
                "shape": [16, 4, 2],
                "data_offsets": [offset, offset + 512],
            }
            offset += 512
            tensors[prefix + "mlp.shared_expert.gate_proj.weight"] = {
                "dtype": "U32",
                "shape": [4, 2],
                "data_offsets": [offset, offset + 32],
            }
            offset += 32

        tensors["language_model.model.norm.weight"] = {
            "dtype": "BF16",
            "shape": [8],
            "data_offsets": [offset, offset + 16],
        }
        offset += 16
        header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
        payload = b"x" * offset
        shard = root / "model-00001-of-00001.safetensors"
        shard.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        index = {
            "metadata": {
                "total_size": len(shard.read_bytes()),
                "total_parameters": 128,
            },
            "weight_map": {name: shard.name for name in tensors},
        }
        (root / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )

    def test_header_and_manifest_are_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-model-name"
            root.mkdir()
            self._make_checkpoint(root)
            self.assertEqual(
                safetensors_header(root / "model-00001-of-00001.safetensors")[
                    "language_model.lm_head.weight"
                ]["shape"],
                [32, 2],
            )
            result = inspect_checkpoint(root, hash_weights=True, revision="abc123")
            self.assertEqual(result["revision"], "abc123")
            self.assertEqual(len(result["manifest_sha256"]), 64)
            self.assertFalse(result["mtp_tensors_present"])
            self.assertNotIn("mtp", result["tensor_name_samples"])
            self.assertEqual(result["checkpoint_name"], "private-model-name")
            self.assertNotIn(str(root), json.dumps(result))
            self.assertGreater(
                result["active_decode_estimate"]["active_parameters_estimate"], 0
            )
            self.assertEqual(
                result["active_decode_estimate"]["totals_by_component"][
                    "active_routed_expert_parameters"
                ],
                256,
            )

    def test_mtp_tensor_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_checkpoint(root)
            shard = root / "model-00001-of-00001.safetensors"
            with shard.open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                existing = json.loads(handle.read(size))
                payload = handle.read()
            existing["mtp.weight"] = {
                "dtype": "U8",
                "shape": [1],
                "data_offsets": [len(payload), len(payload) + 1],
            }
            header = json.dumps(existing, separators=(",", ":")).encode("utf-8")
            shard.write_bytes(
                struct.pack("<Q", len(header)) + header + payload + b"x"
            )
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["mtp.weight"] = shard.name
            index_path.write_text(json.dumps(index), encoding="utf-8")
            result = inspect_checkpoint(root, hash_weights=False)
            self.assertTrue(result["mtp_tensors_present"])
            self.assertIn("mtp.weight", result["tensor_name_samples"]["mtp"])


if __name__ == "__main__":
    unittest.main()
