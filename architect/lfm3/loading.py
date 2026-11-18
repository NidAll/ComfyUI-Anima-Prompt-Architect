"""Offline LFM3-VL loading through ComfyUI's managed model and operations."""

import json

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import TokenizersBackend
from transformers.initialization import no_init_weights

from .artifact import ASSET_KEY, read_assets, validate_config
from .native import NativeMLP, NativeRMSNorm, comfy_attention_forward, fuse_qkv
from .vendor.configuration_lfm3_vl import Lfm3VlConfig
from .vendor.image_processing_lfm3_vl import Lfm3VlImageProcessor
from .vendor.modeling_lfm3 import Lfm3Attention, Lfm3RMSNorm
from .vendor.modeling_lfm3_vl import Lfm3VlForConditionalGeneration, Lfm3VlRMSNorm, Lfm3VlVisionAttention
from .vendor.processing_lfm3_vl import Lfm3VlProcessor
from .vendor.video_processing_lfm3_vl import Lfm3VlVideoProcessor


def replace_operations(model):
    operations = comfy.ops.mixed_precision_ops(
        compute_dtype=torch.bfloat16, disabled=comfy.ops.get_disabled_quant_formats(mm.get_torch_device())
    )
    for name, child in list(model.named_modules()):
        if not name:
            continue
        replacement = None
        if type(child) is torch.nn.Linear:
            replacement = operations.Linear(
                child.in_features, child.out_features, child.bias is not None, device="cpu"
            )
        elif type(child) is torch.nn.Embedding:
            replacement = comfy.ops.manual_cast.Embedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                device="meta",
                dtype=torch.bfloat16,
            )
        elif type(child) is torch.nn.Conv2d:
            replacement = comfy.ops.manual_cast.Conv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                child.bias is not None,
                child.padding_mode,
                device="meta",
                dtype=torch.bfloat16,
            )
        elif type(child) is torch.nn.LayerNorm:
            replacement = comfy.ops.manual_cast.LayerNorm(
                child.normalized_shape,
                child.eps,
                child.elementwise_affine,
                child.bias is not None,
                device="meta",
                dtype=torch.bfloat16,
            )
        elif type(child) in (Lfm3RMSNorm, Lfm3VlRMSNorm):
            replacement = NativeRMSNorm(
                child.weight.shape, eps=child.variance_epsilon, device="meta", dtype=torch.bfloat16
            )
        if replacement is not None:
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, replacement)
    return model


def load_state(model, path, *, reference=False):
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if key == ASSET_KEY:
                continue
            value = reader.get_tensor(key)
            if (
                value.is_floating_point()
                and not key.endswith(".weight_scale")
                and value.dtype != torch.bfloat16
            ):
                raise ValueError(f"LFM3-VL retained weights must be BF16: {key} is {value.dtype}")
            if key.endswith(".comfy_quant"):
                try:
                    conf = json.loads(bytes(value.tolist()))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid quantization metadata for {key}") from exc
                if (
                    conf.get("format") != "int8_tensorwise"
                    or not conf.get("convrot")
                    or conf.get("full_precision_matrix_mult", False)
                ):
                    raise ValueError(f"Expected accelerated INT8 ConvRot metadata for {key}")
            state[key] = value
    quantized = sum(key.endswith(".comfy_quant") for key in state)
    if reference and quantized:
        raise ValueError("Reference LFM3-VL artifact must be BF16")
    if not reference and not quantized:
        raise ValueError("LFM3-VL deployment must contain native INT8 ConvRot layers")
    embed = "model.language_model.embed_tokens.weight"
    head = "lm_head.weight"
    if head not in state and embed in state:
        state[head] = state[embed]
    elif head in state and embed in state and not torch.equal(state[head], state[embed]):
        raise ValueError("LFM3-VL tied embedding and output weights differ")
    model.load_state_dict(state, strict=True, assign=True)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("LFM3-VL checkpoint left unmaterialized weights")
    return quantized


class ManagedModel(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.device = torch.device("cpu")

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def generate(self, **kwargs):
        return self.network.generate(**kwargs)


def use_comfy_attention(model):
    model.config.get_text_config()._comfy_attention = True
    for module in model.modules():
        if isinstance(module, (Lfm3Attention, Lfm3VlVisionAttention)):
            module.comfy_attention = comfy_attention_forward
    return model


def load_lfm3(path, *, reference=False, gemv=True, fuse=True):
    if not reference and mm.get_torch_device().type != "cuda":
        raise RuntimeError("Native LFM3-VL INT8 inference requires the CUDA/Kitchen backend")
    assets = read_assets(path)
    config_data = json.loads(assets["config.json"])
    validate_config(config_data)
    config = Lfm3VlConfig.from_dict(config_data)
    config._attn_implementation = "sdpa"
    with no_init_weights(), init_empty_weights(include_buffers=False):
        model = Lfm3VlForConditionalGeneration(config)
        if fuse:
            for layer in model.model.language_model.layers:
                layer.feed_forward = NativeMLP(config.text_config)
    replace_operations(model)
    use_comfy_attention(model)
    coverage = load_state(model, path, reference=reference)
    if fuse:
        for layer in model.model.language_model.layers:
            layer.feed_forward.fuse()
            if layer.is_attention_layer:
                fuse_qkv(layer.self_attn)
    for module in model.modules():
        if hasattr(module, "comfy_gemv_optimized"):
            module.comfy_gemv_optimized = gemv
    model.generation_config.update(**json.loads(assets["generation_config.json"]))
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer.from_str(assets["tokenizer.json"]),
        **json.loads(assets["tokenizer_config.json"]),
    )
    settings = json.loads(assets["processor_config.json"])
    image_settings = {
        k: v for k, v in settings["image_processor"].items() if k not in ("image_processor_type", "auto_map")
    }
    video_settings = {
        k: v for k, v in settings["video_processor"].items() if k not in ("video_processor_type", "auto_map")
    }
    processor = Lfm3VlProcessor(
        image_processor=Lfm3VlImageProcessor(**image_settings),
        video_processor=Lfm3VlVideoProcessor(**video_settings),
        tokenizer=tokenizer,
        chat_template=assets["chat_template.jinja"],
    )
    model.eval()
    owner = comfy.model_patcher.ModelPatcher(
        ManagedModel(model), load_device=mm.get_torch_device(), offload_device=torch.device("cpu")
    )
    return owner, processor, coverage
