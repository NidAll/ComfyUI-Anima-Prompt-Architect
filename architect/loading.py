"""Local-only deployment loading through native ComfyUI operations and patchers."""

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import TokenizersBackend
from tokenizers import Tokenizer
from transformers.models.minicpmv4_6.image_processing_minicpmv4_6 import MiniCPMV4_6ImageProcessor
from transformers.models.minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import folder_paths

from .vendor.pixai import ViTDetCls, ViTDetClsConfig
from .vendor.minicpmv4_7.configuration_minicpmv4_7 import MiniCPMV4_7Config
from .vendor.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7ForConditionalGeneration
from .vendor.minicpmv4_7.processing_minicpmv4_7 import MiniCPMV4_7Processor
from .native import NativeDeltaNet, NativeMLP, NativeRMSNorm
from .artifact import ASSET_KEY, read_assets

MODEL_KIND = "anima_prompt_architect"
folder_paths.add_model_folder_path(MODEL_KIND, str(Path(folder_paths.models_dir) / MODEL_KIND))
folder_paths.folder_names_and_paths[MODEL_KIND][1].add(".safetensors")


def model_choices():
    return folder_paths.get_filename_list(MODEL_KIND)


def resolve_bundle(name):
    if not name or ".." in Path(name).parts or Path(name).is_absolute() or "\\" in name or Path(name).suffix != ".safetensors":
        raise ValueError("Select a safetensors model inside models/anima_prompt_architect.")
    for root in folder_paths.get_folder_paths(MODEL_KIND):
        root = Path(root).resolve()
        path = (root / name).resolve()
        if path.is_relative_to(root) and path.is_file():
            return path
    raise FileNotFoundError(f"Install {name} with the package preparation script, then refresh the model list.")


def replace_operations(model):
    operations = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16,
                    disabled=comfy.ops.get_disabled_quant_formats(mm.get_torch_device()))
    for name, child in list(model.named_modules()):
        if not name:
            continue
        replacement = None
        if type(child) is torch.nn.Linear:
            replacement = operations.Linear(child.in_features, child.out_features, child.bias is not None, device="cpu")
        elif type(child) is torch.nn.Embedding:
            replacement = comfy.ops.manual_cast.Embedding(child.num_embeddings, child.embedding_dim,
                            padding_idx=child.padding_idx, device="meta", dtype=torch.bfloat16)
        elif type(child) is torch.nn.Conv2d:
            replacement = comfy.ops.manual_cast.Conv2d(child.in_channels, child.out_channels, child.kernel_size,
                            child.stride, child.padding, child.dilation, child.groups, child.bias is not None,
                            child.padding_mode, device="meta", dtype=torch.bfloat16)
        elif type(child) is torch.nn.LayerNorm:
            replacement = comfy.ops.manual_cast.LayerNorm(child.normalized_shape, child.eps,
                            child.elementwise_affine, child.bias is not None, device="meta", dtype=torch.bfloat16)
        elif type(child) is Qwen3_5RMSNorm:
            replacement = NativeRMSNorm(child.weight.shape, eps=child.eps, device="meta", dtype=torch.bfloat16)
        if replacement is not None:
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, replacement)
    return model


def load_state(model, file, require_bf16=False, require_int8=False):
    with safe_open(file, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        quantized = [key for key in keys if key.endswith(".comfy_quant")]
        if require_int8 and not quantized:
            raise ValueError("MiniCPM deployment must contain native INT8 ConvRot layers.")
        state = {}
        for key in keys:
            if key == ASSET_KEY:
                continue
            value = reader.get_tensor(key)
            if require_bf16 and value.is_floating_point() and value.dtype != torch.bfloat16:
                raise ValueError(f"PixAI deployment weights must all be BF16: {key} is {value.dtype}.")
            if require_int8 and value.is_floating_point() and not key.endswith(".weight_scale") and value.dtype != torch.bfloat16:
                raise ValueError(f"MiniCPM retained model weights must be BF16: {key} is {value.dtype}.")
            if key.endswith(".comfy_quant"):
                conf = json.loads(bytes(value.tolist()))
                if require_int8 and (conf.get("format") != "int8_tensorwise" or not conf.get("convrot") or conf.get("full_precision_matrix_mult", False)):
                    raise ValueError(f"Expected accelerated INT8 ConvRot metadata for {key}.")
            state[key] = value
    # Native quantized ops consume their scale/metadata keys during recursive loading.
    model.load_state_dict(state, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("Checkpoint left unmaterialized model weights.")
    return len(quantized)


class ManagedModel(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.device = torch.device("cpu")

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def generate(self, **kwargs):
        return self.network.generate(**kwargs)


def patcher(model):
    model.eval()
    return comfy.model_patcher.ModelPatcher(ManagedModel(model), load_device=mm.get_torch_device(), offload_device=torch.device("cpu"))


def load_tagger(path):
    assets = read_assets(path, "pixai")
    config = ViTDetClsConfig.from_dict(json.loads(assets["config.json"]))
    with no_init_weights(), init_empty_weights(include_buffers=False):
        model = ViTDetCls(config)
    replace_operations(model)
    load_state(model, str(path), require_bf16=True)
    # RoPE tables are derived complex buffers, not deployment model weights.
    for block in model.blocks:
        if block.attn.use_rope:
            block.attn._setup_rope_freqs()
    return patcher(model), config


def load_minicpm(path, reference=False, fused_decode=True):
    if not reference and mm.get_torch_device().type != "cuda":
        raise RuntimeError("Native accelerated MiniCPM INT8 inference requires the installed CUDA/Kitchen backend.")
    assets = read_assets(path, "minicpm")
    config = MiniCPMV4_7Config.from_dict(json.loads(assets["config.json"]))
    config._attn_implementation = "sdpa"
    with no_init_weights(), init_empty_weights(include_buffers=False):
        model = MiniCPMV4_7ForConditionalGeneration(config)
        for layer in model.model.language_model.layers:
            layer.mlp = NativeMLP(config.text_config, config.text_config.intermediate_size)
        if fused_decode:
            for index, layer in enumerate(model.model.language_model.layers):
                if config.text_config.layer_types[index] == "linear_attention":
                    layer.linear_attn = NativeDeltaNet(config.text_config, index)
    replace_operations(model)
    coverage = load_state(model, str(path), require_int8=not reference)
    for layer in model.model.language_model.layers:
        layer.mlp.fuse()
        if isinstance(getattr(layer, "linear_attn", None), NativeDeltaNet):
            layer.linear_attn.fuse()
    tokenizer = TokenizersBackend(tokenizer_object=Tokenizer.from_str(assets["tokenizer.json"]),
                    **json.loads(assets["tokenizer_config.json"]))
    settings = json.loads(assets["processor_config.json"])
    image_settings = {k: v for k, v in settings["image_processor"].items() if k != "image_processor_type"}
    video_settings = {k: v for k, v in settings["video_processor"].items() if k != "video_processor_type"}
    processor = MiniCPMV4_7Processor(image_processor=MiniCPMV4_6ImageProcessor(**image_settings),
                    video_processor=MiniCPMV4_6VideoProcessor(**video_settings), tokenizer=tokenizer,
                    chat_template=assets["chat_template.jinja"])
    return patcher(model), processor, coverage
