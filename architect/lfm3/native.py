"""Comfy-managed LFM3 normalization."""

import comfy.ops
import comfy.rmsnorm
import torch
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention_for_device

from .vendor.modeling_lfm3 import Lfm3MLP


class NativeRMSNorm(comfy.ops.manual_cast.RMSNorm):
    def forward(self, x):
        with comfy.ops.CastBiasWeightContext(self, x, offloadable=True) as (weight, _):
            normalized = comfy.rmsnorm.rms_norm(x.float(), eps=self.eps).to(x.dtype)
            return weight * normalized


def merge_int8(layers):
    if any(getattr(layer, "quant_format", None) != "int8_tensorwise" for layer in layers):
        return None
    states = [layer.state_dict() for layer in layers]
    first = states[0]
    if any(
        set(state) != set(first) or not torch.equal(state["comfy_quant"], first["comfy_quant"])
        for state in states[1:]
    ):
        return None
    if any(
        state["weight_scale"].numel() != layer.out_features
        for state, layer in zip(states, layers, strict=True)
    ):
        return None
    merged = type(layers[0])(
        layers[0].in_features, sum(layer.out_features for layer in layers), bias="bias" in first, device="cpu"
    )
    state = {
        key: torch.cat([item[key] for item in states], dim=0)
        if key in ("weight", "weight_scale", "bias")
        else value
        for key, value in first.items()
    }
    merged.load_state_dict(state, strict=True, assign=True)
    return merged


def fuse_qkv(attention):
    layers = (attention.q_proj, attention.k_proj, attention.v_proj)
    fused = merge_int8(layers)
    if fused is None:
        return False
    attention.qkv_proj = fused
    attention.qkv_split_sizes = tuple(layer.out_features for layer in layers)
    del attention.q_proj, attention.k_proj, attention.v_proj
    return True


def comfy_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    if attention_mask is None and getattr(module, "is_causal", False) and query.shape[-2] > 1:
        raise ValueError("Comfy attention needs an explicit causal mask for LFM3 prefill")
    if attention_mask is not None and attention_mask.dtype == torch.bool:
        attention_mask = torch.zeros(
            attention_mask.shape, device=query.device, dtype=query.dtype
        ).masked_fill(~attention_mask, torch.finfo(query.dtype).min)
    grouped = module.num_key_value_groups > 1
    if grouped and attention_mask is not None:
        key, value = comfy.ops.repeat_kv_for_gqa(key, value, module.config.num_attention_heads, 1)
        grouped = False
    attention = optimized_attention_for_device(query.device, mask=attention_mask is not None)
    output = attention(
        query,
        key,
        value,
        module.config.num_attention_heads,
        mask=attention_mask,
        skip_reshape=True,
        skip_output_reshape=True,
        enable_gqa=grouped,
        scale=scaling,
    )
    return output.transpose(1, 2).contiguous(), None


class NativeMLP(Lfm3MLP):
    def fuse(self):
        self.gate_up = merge_int8([self.w1, self.w3])
        if self.gate_up is not None:
            del self.w1, self.w3

    def forward(self, x):
        if self.gate_up is None:
            return super().forward(x)
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)
