"""Package-local Qwen3.5 decode adapter for Comfy Kitchen's fused DeltaNet kernels."""

import torch
import comfy_kitchen
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet, Qwen3_5MLP, apply_mask_to_padding_states,
    causal_conv1d_fn, torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
)
import comfy.ops
import comfy.rmsnorm


def merge_int8(layers):
    if any(getattr(layer, "quant_format", None) != "int8_tensorwise" for layer in layers):
        return None
    states = [layer.state_dict() for layer in layers]
    first = states[0]
    if any(set(state) != set(first) or not torch.equal(state["comfy_quant"], first["comfy_quant"]) for state in states[1:]):
        return None
    if any(state["weight_scale"].numel() != layer.out_features for state, layer in zip(states, layers)):
        return None
    merged = type(layers[0])(layers[0].in_features, sum(layer.out_features for layer in layers),
                            bias="bias" in first, device="cpu")
    state = {key: torch.cat([item[key] for item in states], dim=0) if key in ("weight", "weight_scale", "bias") else value
             for key, value in first.items()}
    merged.load_state_dict(state, strict=True, assign=True)
    return merged


class NativeMLP(Qwen3_5MLP):
    def fuse(self):
        self.gate_up = merge_int8([self.gate_proj, self.up_proj])
        if self.gate_up is not None:
            del self.gate_proj, self.up_proj

    def forward(self, x):
        if self.gate_up is None:
            return super().forward(x)
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class NativeRMSNorm(comfy.ops.manual_cast.RMSNorm):
    def forward(self, x):
        with comfy.ops.CastBiasWeightContext(self, x, offloadable=True) as (weight, _):
            return comfy.rmsnorm.rms_norm(x.float(), 1.0 + weight.float(), self.eps).to(x.dtype)


class NativeDeltaNet(Qwen3_5GatedDeltaNet):
    def fuse(self):
        self.qkvz = merge_int8([self.in_proj_qkv, self.in_proj_z])
        if self.qkvz is not None:
            del self.in_proj_qkv, self.in_proj_z

    def projections(self, hidden_states):
        if self.qkvz is None:
            return self.in_proj_qkv(hidden_states), self.in_proj_z(hidden_states)
        return self.qkvz(hidden_states).split((2 * self.key_dim + self.value_dim, self.value_dim), dim=-1)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected, z = self.projections(hidden_states)
        previous = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        batch, length, _ = hidden_states.shape
        native_decode = (length == 1 and previous and not cache_params.layers[self.layer_idx].record_past
                         and comfy_kitchen.gated_delta_decode_is_available(hidden_states.device, self.head_k_dim, self.head_v_dim))
        if not native_decode:
            mixed = projected.transpose(1, 2)
            if cache_params is not None:
                mixed = cache_params.update_conv_state(mixed, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
            mixed = causal_conv1d_fn(mixed, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation)
            mixed = mixed[..., -length:].transpose(1, 2)
            q, k, v = mixed.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
            q = q.reshape(batch, length, self.num_k_heads, self.head_k_dim)
            k = k.reshape(batch, length, self.num_k_heads, self.head_k_dim)
            v = v.reshape(batch, length, self.num_v_heads, self.head_v_dim)
            if self.num_v_heads > self.num_k_heads:
                q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
                k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            beta = self.in_proj_b(hidden_states).sigmoid()
            g = -self.A_log.float().exp() * torch.nn.functional.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
            state = cache_params.layers[self.layer_idx].recurrent_states[0] if previous else None
            rule = torch_recurrent_gated_delta_rule if previous and length == 1 else torch_chunk_gated_delta_rule
            output, state = rule(q, k, v, g=g, beta=beta, initial_state=state,
                                 output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True)
            if cache_params is not None:
                cache_params.update_recurrent_state(state, self.layer_idx)
            output = self.norm(output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
            return self.out_proj(output.reshape(batch, length, self.value_dim))
        layer = cache_params.layers[self.layer_idx]
        conv_state = layer.conv_states[0]
        window = conv_state[..., -(self.conv_kernel_size - 1):].contiguous()
        updated = torch.cat((window, projected.transpose(1, 2)), dim=-1)
        mixed = comfy_kitchen.deltanet_conv_step(projected, window, self.conv1d.weight, self.conv1d.bias)
        conv_state.copy_(updated)
        output = comfy_kitchen.gated_delta_decode_fused(
            mixed, hidden_states, self.in_proj_a.weight, self.in_proj_b.weight,
            self.dt_bias.float().contiguous(), (-self.A_log.float().exp()).contiguous(),
            layer.recurrent_states[0], self.key_dim, self.num_k_heads, self.head_k_dim ** -.5,
            z, self.norm.weight, self.layer_norm_epsilon)
        cache_params.update_recurrent_state(layer.recurrent_states[0], self.layer_idx)
        return self.out_proj(output.reshape(hidden_states.shape[0], 1, self.value_dim))
