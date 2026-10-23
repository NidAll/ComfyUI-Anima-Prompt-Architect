"""Convert local original checkpoints into standalone Prompt Architect models."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from architect.artifact import pack


def prepare(kind, source, output, quantizer_root=None, device="cuda"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.suffix != ".safetensors":
        raise ValueError("Output must be a .safetensors file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="anima-prepare-", dir=output.parent) as directory:
        temporary = Path(directory)
        if kind == "pixai":
            state = {}
            with safe_open(source / "model.safetensors", framework="pt") as reader:
                for key in reader.keys():
                    tensor = reader.get_tensor(key)
                    state[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            save_file(state, str(temporary / "model.safetensors"))
            config = json.loads((source / "config.json").read_text())
            config.update(dtype="bfloat16", torch_dtype="bfloat16")
            config.pop("auto_map", None)
            config.pop("custom_pipelines", None)
            (temporary / "config.json").write_text(json.dumps(config))
            shutil.copyfile(source / "preprocessor_config.json", temporary / "preprocessor_config.json")
        else:
            if quantizer_root:
                sys.path.insert(0, str(Path(quantizer_root).resolve()))
            from native_quantizer import api
            from native_quantizer.architectures.policies import REGISTRY
            from native_quantizer.options import ConversionOptions

            built_in = "minicpm_v47" in REGISTRY
            options = ConversionOptions(model=str(source / "model.safetensors"),
                output=str(temporary / "model.safetensors"),
                architecture="minicpm_v47" if built_in else "minicpm_v47_architect",
                architecture_policy=None if built_in else str(PACKAGE / "recipes/minicpm-v47.json"),
                format="int8-convrot", components="text_encoder", device=device,
                compute_dtype="fp32", output_dtype="bf16", optimizer="mse", mse_objective="weight",
                mse_steps=9, mse_iterations=2, max_memory="512M", validate=True, yes=True)
            if api.convert(options):
                raise RuntimeError("Native quantization failed")
            for name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
                shutil.copyfile(source / name, temporary / name)
        return pack(temporary, output, kind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["pixai", "minicpm"])
    parser.add_argument("--source", required=True, help="Original local model directory")
    parser.add_argument("--output", required=True, help="New .safetensors artifact")
    parser.add_argument("--quantizer-root", help="Optional quantizer source checkout; otherwise use installed package")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    print(json.dumps(prepare(args.kind, args.source, args.output, args.quantizer_root, args.device), indent=2))
