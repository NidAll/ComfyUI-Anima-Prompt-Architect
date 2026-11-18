"""Small, offline regressions for the vendored LFM3-VL runtime."""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
init_empty_weights = pytest.importorskip("accelerate").init_empty_weights
save_file = pytest.importorskip("safetensors.torch").save_file
no_init_weights = pytest.importorskip("transformers.initialization").no_init_weights

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "comfyui" / "upscaler" / "lfm3"
if not BASE.exists():
    BASE = ROOT / "architect" / "lfm3"
package = types.ModuleType("lfm3_runtime_test")
package.__path__ = [str(BASE)]
sys.modules[package.__name__] = package
artifact = importlib.import_module("lfm3_runtime_test.artifact")
config_mod = importlib.import_module("lfm3_runtime_test.vendor.configuration_lfm3_vl")
model_mod = importlib.import_module("lfm3_runtime_test.vendor.modeling_lfm3_vl")


def native_module(name, *, cpu=True):
    cli_args = pytest.importorskip("comfy.cli_args")
    cli_args.args.cpu = cpu
    return importlib.import_module(f"lfm3_runtime_test.{name}")


def test_asset_integrity_and_manifest(tmp_path):
    assets = {name: "test" for name in artifact.FILES}
    payload = json.dumps(assets).encode()
    manifest = {"version": 1, "kind": "lfm3_vl", "assets_sha256": hashlib.sha256(payload).hexdigest()}
    path = tmp_path / "model.safetensors"
    save_file(
        {artifact.ASSET_KEY: torch.tensor(list(payload), dtype=torch.uint8)},
        str(path),
        metadata={"lfm3_vl": json.dumps(manifest)},
    )
    assert artifact.read_assets(path) == assets
    manifest["assets_sha256"] = "0" * 64
    save_file(
        {artifact.ASSET_KEY: torch.tensor(list(payload), dtype=torch.uint8)},
        str(path),
        metadata={"lfm3_vl": json.dumps(manifest)},
    )
    with pytest.raises(ValueError, match="integrity"):
        artifact.read_assets(path)


def test_offline_preparation_preserves_source_and_refuses_overwrite(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_lfm_test", ROOT / "scripts/prepare_lfm3.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    source = tmp_path / "source"
    source.mkdir()
    assets = {name: "{}" for name in prepare.ASSET_FILES}
    assets["config.json"] = '{"model_type":"lfm3_vl"}'
    assets["chat_template.jinja"] = "{{ messages }}"
    for name, value in assets.items():
        (source / name).write_text(value)
    weights = torch.ones(4, 8, dtype=torch.bfloat16)
    original = source / "model.safetensors"
    save_file({prepare.EMBEDDING: weights, prepare.HEAD: weights.clone()}, str(original))
    before = original.read_bytes()
    output = tmp_path / "bundle.safetensors"
    result = prepare.prepare(source, output, format="bf16")
    assert result["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert artifact.read_assets(output) == assets and original.read_bytes() == before
    with pytest.raises(ValueError, match="new .safetensors"):
        prepare.prepare(source, output, format="bf16")


def tiny_model():
    text = dict(
        model_type="lfm3",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        vocab_size=128,
        layer_types=["conv", "full_attention", "sliding_attention"],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    vision = dict(
        model_type="lfm3_vl_vision",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        rope_parameters={"rope_type": "axial", "rope_theta": 1000000.0},
    )
    config = config_mod.Lfm3VlConfig(
        text_config=text,
        vision_config=vision,
        projector_hidden_size=32,
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    assert config.text_config.layer_types == ["conv", "full_attention", "sliding_attention"]
    assert config.vision_config.layer_types == ["conv", "full_attention"]
    with no_init_weights():
        model = model_mod.Lfm3VlForConditionalGeneration(config).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.02)
    return model


def test_image_prefill_and_cached_decode_match_full_forward():
    model = tiny_model()
    assert model.config.text_config.conv_L_cache == 7
    prefix = torch.tensor([[1, 5, 3, 6, 7]])
    next_token = torch.tensor([[9]])
    pixels = torch.randn(4, 96)
    grid = torch.tensor([[1, 2, 2]])
    with torch.inference_mode():
        full = model(
            input_ids=torch.cat((prefix, next_token), dim=1),
            pixel_values=pixels,
            image_grid_thw=grid,
            use_cache=False,
        ).logits[:, -1]
        first = model(input_ids=prefix, pixel_values=pixels, image_grid_thw=grid, use_cache=True)
        cached = model(input_ids=next_token, past_key_values=first.past_key_values, use_cache=True).logits[
            :, -1
        ]
        generated = model.generate(
            input_ids=prefix, pixel_values=pixels, image_grid_thw=grid, do_sample=False, max_new_tokens=3
        )
    torch.testing.assert_close(cached, full, atol=2e-5, rtol=2e-4)
    assert generated.shape[1] > prefix.shape[1]


def test_comfy_attention_matches_reference_for_text_prefill_and_decode():
    loading = native_module("loading")
    reference = tiny_model()
    actual = tiny_model()
    actual.load_state_dict(reference.state_dict(), strict=True)
    loading.use_comfy_attention(actual)
    prefix = torch.tensor([[1, 7, 8, 9]])
    next_token = torch.tensor([[10]])
    with torch.inference_mode():
        reference_prefill = reference(input_ids=prefix, use_cache=True)
        actual_prefill = actual(input_ids=prefix, use_cache=True)
        reference_decode = reference(
            input_ids=next_token, past_key_values=reference_prefill.past_key_values, use_cache=True
        ).logits[:, -1]
        actual_decode = actual(
            input_ids=next_token, past_key_values=actual_prefill.past_key_values, use_cache=True
        ).logits[:, -1]
    torch.testing.assert_close(actual_prefill.logits, reference_prefill.logits, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual_decode, reference_decode, atol=2e-5, rtol=2e-4)


def test_comfy_vision_attention_matches_reference():
    loading = native_module("loading")
    reference = tiny_model()
    actual = tiny_model()
    actual.load_state_dict(reference.state_dict(), strict=True)
    loading.use_comfy_attention(actual)
    pixels = torch.randn(4, 96)
    grid = torch.tensor([[1, 2, 2]])
    with torch.inference_mode():
        expected = reference.model.vision_tower(pixels, grid_thw=grid, return_dict=True).last_hidden_state
        output = actual.model.vision_tower(pixels, grid_thw=grid, return_dict=True).last_hidden_state
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)


def test_fused_int8_qkv_projection_preserves_bytes_and_output():
    native = native_module("native")
    metadata = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 8}).encode()),
        dtype=torch.uint8,
    )

    class TinyInt8Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=False, device="cpu"):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.quant_format = "int8_tensorwise"
            self.register_buffer(
                "weight", torch.zeros(out_features, in_features, dtype=torch.int8, device=device)
            )
            self.register_buffer(
                "weight_scale", torch.ones(out_features, 1, dtype=torch.float32, device=device)
            )
            self.register_buffer("comfy_quant", metadata.clone().to(device))

        def forward(self, x):
            weight = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)
            return torch.nn.functional.linear(x, weight)

    layers = []
    for rows in (16, 8, 8):
        layer = TinyInt8Linear(8, rows)
        layer.weight.copy_(torch.randint(-16, 16, layer.weight.shape, dtype=torch.int8))
        layer.weight_scale.fill_(0.125)
        layers.append(layer)
    attention = types.SimpleNamespace(q_proj=layers[0], k_proj=layers[1], v_proj=layers[2])
    assert native.fuse_qkv(attention)
    fused_state = attention.qkv_proj.state_dict()
    assert torch.equal(fused_state["weight"], torch.cat([layer.weight for layer in layers]))
    assert torch.equal(fused_state["weight_scale"], torch.cat([layer.weight_scale for layer in layers]))
    assert torch.equal(fused_state["comfy_quant"], metadata)
    x = torch.randint(-8, 8, (2, 3, 8), dtype=torch.int32).float() / 8
    expected = torch.cat([layer(x) for layer in layers], dim=-1)
    actual = attention.qkv_proj(x).split(attention.qkv_split_sizes, dim=-1)
    torch.testing.assert_close(torch.cat(actual, dim=-1), expected, atol=0, rtol=0)


def test_native_rmsnorm_matches_upstream():
    native = native_module("native")
    source = importlib.import_module("lfm3_runtime_test.vendor.modeling_lfm3")
    for width in (128, 2048):
        x = torch.randn(2, 4, width, dtype=torch.bfloat16, device="cpu")
        reference = source.Lfm3RMSNorm(width, eps=1e-5).to(device="cpu", dtype=torch.bfloat16)
        actual = native.NativeRMSNorm((width,), eps=1e-5, device="cpu", dtype=torch.bfloat16)
        with torch.no_grad():
            reference.weight.normal_()
            actual.weight.copy_(reference.weight)
        with torch.inference_mode():
            torch.testing.assert_close(actual(x), reference(x), atol=0, rtol=0)


def test_strict_reference_load_shares_embedding_and_head(tmp_path):
    loading = native_module("loading")
    reference = tiny_model()
    reference.lm_head.weight = reference.model.language_model.embed_tokens.weight
    state = {
        key: value.detach().clone().to(torch.bfloat16)
        for key, value in reference.state_dict().items()
        if key != "lm_head.weight"
    }
    path = tmp_path / "tiny.safetensors"
    save_file(state, str(path))
    with no_init_weights(), init_empty_weights(include_buffers=False):
        target = model_mod.Lfm3VlForConditionalGeneration(reference.config).eval()
    loading.replace_operations(target)
    assert loading.load_state(target, path, reference=True) == 0
    assert target.lm_head.weight is target.model.language_model.embed_tokens.weight
    ids = torch.tensor([[1, 7, 8]])
    with torch.inference_mode():
        actual = target(input_ids=ids, use_cache=False).logits.float()
        expected = reference(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.05)


@pytest.mark.skipif(
    os.environ.get("LFM3_RUN_NATIVE_TESTS") != "1" or not torch.cuda.is_available(),
    reason="requires native CUDA convolution kernels",
)
def test_width7_convolution_prefill_and_cached_decode_match_torch():
    conv = importlib.import_module("lfm3_runtime_test.vendor.lfm3_conv")
    if not conv.is_triton_available():
        pytest.skip("Triton is unavailable")
    x = torch.randn(2, 16, 11, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(16, 7, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(16, dtype=torch.bfloat16, device="cuda")
    expected = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x, (6, 0)), weight.unsqueeze(1), bias, groups=x.shape[1]
    )
    prefix_length = 8
    state = x[:, :, prefix_length - 6 : prefix_length].contiguous().clone()
    current = x[:, :, prefix_length : prefix_length + 1].contiguous()
    with torch.inference_mode():
        prefill = conv.lfm3_causal_conv1d(x, weight, bias)
        decoded = conv.lfm3_causal_conv1d_update(current, state, weight, bias)
    torch.testing.assert_close(prefill, expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(
        decoded, expected[:, :, prefix_length : prefix_length + 1], atol=0.01, rtol=0.01
    )
    torch.testing.assert_close(state, x[:, :, prefix_length - 5 : prefix_length + 1], atol=0, rtol=0)


def test_config_roundtrip_preserves_scoped_rope_and_native_convolutions():
    config = tiny_model().config
    assert set(config.text_config.rope_parameters) == {"full_attention", "sliding_attention"}
    copied = config_mod.Lfm3VlConfig.from_dict(config.to_dict())
    assert copied.text_config.rope_parameters == config.text_config.rope_parameters
    assert copied.text_config.layer_types == ["conv", "full_attention", "sliding_attention"]
    assert copied.vision_config.rope_parameters == {"rope_type": "axial", "rope_theta": 1000000.0}


@pytest.mark.skipif(
    os.environ.get("LFM3_RUN_NATIVE_TESTS") != "1" or not torch.cuda.is_available(),
    reason="requires native CUDA INT8 backend",
)
def test_fused_int8_projection_is_byte_and_output_exact():
    from unittest.mock import patch

    import comfy.ops
    from comfy.quant_ops import TensorWiseINT8Layout

    native = native_module("native", cpu=False)
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    metadata = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()),
        dtype=torch.uint8,
    )
    layers = []
    for rows in (256, 128, 128):
        layer = ops.Linear(256, rows, bias=False, device="cpu")
        layer.load_state_dict(
            {
                "weight": torch.randint(-127, 128, (rows, 256), dtype=torch.int8),
                "weight_scale": torch.full((rows, 1), 0.001),
                "comfy_quant": metadata,
            },
            strict=True,
        )
        layers.append(layer)
    attention = types.SimpleNamespace(q_proj=layers[0], k_proj=layers[1], v_proj=layers[2])
    assert native.fuse_qkv(attention)
    merged = attention.qkv_proj
    assert torch.equal(
        merged.state_dict()["weight"], torch.cat([layer.state_dict()["weight"] for layer in layers])
    )
    assert torch.equal(
        merged.state_dict()["weight_scale"],
        torch.cat([layer.state_dict()["weight_scale"] for layer in layers]),
    )
    merged.to("cuda")
    for layer in layers:
        layer.to("cuda")
    with (
        torch.inference_mode(),
        patch.object(TensorWiseINT8Layout, "dequantize", side_effect=AssertionError("No dequantization")),
    ):
        for length in (1, 16):
            x = torch.randn(1, length, 256, dtype=torch.bfloat16, device="cuda")
            actual = merged(x).split(attention.qkv_split_sizes, dim=-1)
            torch.testing.assert_close(
                torch.cat(actual, dim=-1), torch.cat([layer(x) for layer in layers], dim=-1), atol=0, rtol=0
            )
