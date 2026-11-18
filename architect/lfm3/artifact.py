"""Read bounded, authenticated data assets from a local LFM3-VL artifact."""

import hashlib
import json

from safetensors import safe_open

ASSET_KEY = "__lfm3_vl_assets__"
FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    }
)
MAX_ASSET_BYTES = 64 * 1024 * 1024


def read_manifest(path):
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        try:
            manifest = json.loads((reader.metadata() or {}).get("lfm3_vl", "{}"))
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid LFM3-VL artifact manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("kind") != "lfm3_vl":
        raise ValueError("Select a self-contained LFM3-VL artifact")
    return manifest


def read_assets(path):
    manifest = read_manifest(path)
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        if ASSET_KEY not in reader.keys():
            raise ValueError("LFM3-VL artifact has no embedded assets")
        asset = reader.get_slice(ASSET_KEY)
        shape = asset.get_shape()
        if asset.get_dtype() != "U8" or len(shape) != 1 or shape[0] > MAX_ASSET_BYTES:
            raise ValueError("Invalid LFM3-VL asset tensor")
        payload = reader.get_tensor(ASSET_KEY).numpy().tobytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("assets_sha256"):
        raise ValueError("LFM3-VL assets failed integrity validation")
    try:
        assets = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid LFM3-VL assets") from exc
    if (
        not isinstance(assets, dict)
        or set(assets) != FILES
        or not all(isinstance(v, str) for v in assets.values())
    ):
        raise ValueError("Unexpected LFM3-VL assets")
    return assets


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Unsupported LFM3-VL configuration")
    t, v = config.get("text_config"), config.get("vision_config")
    if not isinstance(t, dict) or not isinstance(v, dict):
        raise ValueError("Unsupported LFM3-VL configuration")
    text_layers = ["conv" if i % 4 in (0, 1, 3) else "full_attention" for i in range(24)]
    for i in range(4, 24, 4):
        text_layers[i] = "sliding_attention"
    vision_layers = ["conv" if i < 2 or i % 2 else "full_attention" for i in range(14)]
    if (
        config.get("model_type") != "lfm3_vl"
        or t.get("model_type") != "lfm3"
        or v.get("model_type") != "lfm3_vl_vision"
        or t.get("hidden_size") != 2048
        or t.get("intermediate_size") != 7680
        or t.get("vocab_size") != 128000
        or t.get("num_hidden_layers") != 24
        or t.get("num_attention_heads") != 16
        or t.get("num_key_value_heads") != 4
        or t.get("head_dim") != 128
        or t.get("layer_types") != text_layers
        or v.get("layer_types") != vision_layers
        or v.get("hidden_size") != 1024
        or v.get("intermediate_size") != 2560
        or v.get("num_hidden_layers") != 14
        or v.get("num_attention_heads") != 16
        or v.get("num_key_value_heads") != 8
        or v.get("patch_size") != 16
        or v.get("temporal_patch_size") != 2
        or v.get("spatial_merge_size") != 2
        or config.get("projector_hidden_size") != 4096
        or config.get("downsample_factor") != 2
        or not config.get("tie_word_embeddings")
        or not t.get("tie_word_embeddings")
    ):
        raise ValueError("Unsupported LFM3-VL configuration; expected LFM3-VL-2B")
