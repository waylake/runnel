"""Exact MLX graph optimizations for Qwen3.5/Ornith sparse MoE."""

from __future__ import annotations

from typing import Any


def fuse_routed_gate_up(model: Any) -> int:
    """Replace two routed gather-qmm calls with one packed gather-qmm per layer.

    The checkpoint's routed gate and up projections share the same affine
    quantization format. Concatenating their packed output rows preserves each
    row's arithmetic while removing one expert gather, index lookup, and launch
    per decoder layer. At top-8 batch-1 decode this reduces routed expert
    gathers from 120 to 80 per generated token.
    """
    import gc

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    class FusedQuantizedSwitchGLU(nn.Module):
        def __init__(self, old: Any) -> None:
            super().__init__()
            gate = old.gate_proj
            up = old.up_proj
            if not (
                isinstance(gate, type(up))
                and gate.bits == up.bits
                and gate.group_size == up.group_size
                and gate.mode == up.mode
            ):
                raise ValueError("routed gate/up quantization formats differ")
            if ("bias" in gate) or ("bias" in up):
                raise ValueError("module biases are not expected for this checkpoint")

            self.weight = mx.concatenate([up["weight"], gate["weight"]], axis=1)
            self.scales = mx.concatenate([up["scales"], gate["scales"]], axis=1)
            up_biases = up.get("biases")
            gate_biases = gate.get("biases")
            if (up_biases is None) != (gate_biases is None):
                raise ValueError("routed gate/up affine-bias formats differ")
            self.biases = (
                None
                if up_biases is None
                else mx.concatenate([up_biases, gate_biases], axis=1)
            )
            self.group_size = gate.group_size
            self.bits = gate.bits
            self.mode = gate.mode
            self.down_proj = old.down_proj
            self.activation = old.activation
            self.freeze()
            mx.eval(self.parameters())

        def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x, idx, inv_order = _gather_sort(x, indices)
            fused = mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                self.get("biases"),
                rhs_indices=idx,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=do_sort,
            )
            x_up, x_gate = mx.split(fused, 2, axis=-1)
            out = self.down_proj(
                self.activation(x_up, x_gate), idx, sorted_indices=do_sort
            )
            if do_sort:
                out = _scatter_unsort(out, inv_order, indices.shape)
            return out.squeeze(-2)

    count = 0
    for layer in model.language_model.layers:
        old = layer.mlp.switch_mlp
        if not hasattr(old.gate_proj, "bits"):
            raise ValueError("routed gate/up are not quantized as expected")
        fused = FusedQuantizedSwitchGLU(old)
        layer.mlp.switch_mlp = fused
        count += 1
        del old, fused
        gc.collect()
        mx.clear_cache()
    return count
