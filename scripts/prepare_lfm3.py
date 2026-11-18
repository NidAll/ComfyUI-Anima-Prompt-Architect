"""Prepare an offline, single-file LFM3-VL bundle from local upstream files."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ASSET_KEY = "__lfm3_vl_assets__"
ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
)
MAX_ASSET_BYTES = 32 * 1024 * 1024
EMBEDDING = "model.language_model.embed_tokens.weight"
HEAD = "lm_head.weight"


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def update_quantizer_hash(path, output):
    """Update the converter's payload checksum after adding bundle tensors."""
    with path.open("rb") as src:
        size = struct.unpack("<Q", src.read(8))[0]
        header = json.loads(src.read(size))
        provenance = json.loads(header["__metadata__"]["comfy_quantizer"])
        payload_offset = 8 + size
        provenance["payload_sha256"] = hashlib.file_digest(src, "sha256").hexdigest()
        header["__metadata__"]["comfy_quantizer"] = json.dumps(provenance, separators=(",", ":"))
        encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        src.seek(payload_offset)
        with output.open("wb") as dst:
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)


def prepare(
    source,
    output,
    *,
    format="int8-convrot",
    device="cpu",
    quantizer_root=None,
    convrot_groupsize=None,
    exclude=(),
    reference_recipe=None,
    max_memory="512M",
    optimizer="none",
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.suffix != ".safetensors" or output.exists():
        raise ValueError("Output must be a new .safetensors file")
    if format not in ("bf16", "int8-convrot"):
        raise ValueError("Unsupported format")
    if format == "bf16" and (
        convrot_groupsize is not None or exclude or reference_recipe or optimizer != "none"
    ):
        raise ValueError("Quantization options require int8-convrot")
    if sum((source / name).stat().st_size for name in ASSET_FILES) > MAX_ASSET_BYTES:
        raise ValueError("Source assets exceed 32 MiB")
    files = {name: (source / name).read_text(encoding="utf-8") for name in ASSET_FILES}
    for name in ASSET_FILES:
        if name.endswith(".json"):
            json.loads(files[name])
    if json.loads(files["config.json"]).get("model_type") != "lfm3_vl":
        raise ValueError("Expected an LFM3-VL configuration")
    payload = json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_ASSET_BYTES:
        raise ValueError("Embedded assets exceed 32 MiB")
    original = source / "model.safetensors"
    source_hash = digest(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lfm3-prepare-", dir=output.parent) as directory:
        work = Path(directory)
        weights = original
        if format == "int8-convrot":
            if quantizer_root:
                sys.path.insert(0, str(Path(quantizer_root).resolve()))
            from native_quantizer import api
            from native_quantizer.options import ConversionOptions

            weights = work / "quantized.safetensors"
            options = ConversionOptions(
                model=str(original),
                output=str(weights),
                format="int8-convrot",
                architecture="auto",
                components="text_encoder",
                device=device,
                output_dtype="auto",
                max_memory=max_memory,
                validate=True,
                int8_convrot_groupsize=convrot_groupsize,
                exclude=list(exclude),
                optimizer=optimizer,
                mse_objective="weight",
                mse_steps=9,
                mse_iterations=2,
                reference_recipe=reference_recipe,
                yes=True,
                progress="off",
            )
            if api.convert(options):
                raise RuntimeError("Native quantization failed")
            result = api.verify_output(str(weights))
            if not result["ok"]:
                raise ValueError("Quantized output verification failed")
        with safe_open(str(weights), framework="pt", device="cpu") as reader:
            keys = set(reader.keys())
            if ASSET_KEY in keys or EMBEDDING not in keys:
                raise ValueError("Unexpected source weights or missing tied embedding")
            state = {key: reader.get_tensor(key) for key in keys}
            metadata = dict(reader.metadata() or {})
        if format == "bf16" and any(
            t.is_floating_point() and t.dtype != torch.bfloat16 for t in state.values()
        ):
            raise ValueError("BF16 bundle requires BF16 source weights")
        if HEAD in state:
            if not torch.equal(state[HEAD], state[EMBEDDING]):
                raise ValueError("Tied embedding and head differ")
        state[ASSET_KEY] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        manifest = {
            "version": 1,
            "kind": "lfm3_vl",
            "assets_sha256": hashlib.sha256(payload).hexdigest(),
            "source_sha256": source_hash,
            "runtime_reference_repository": "LiquidAI/LFM3-VL-2B",
            "runtime_reference_revision": "f83d2b0a2296aaa277f394990a9b8b949ddf9691",
        }
        license_file = source / "LICENSE"
        if license_file.is_file():
            if license_file.stat().st_size > 128 * 1024:
                raise ValueError("Source license exceeds 128 KiB")
            manifest["license_text"] = license_file.read_text(encoding="utf-8")
        metadata["lfm3_vl"] = json.dumps(manifest)
        staged = work / "bundle.safetensors"
        save_file(state, str(staged), metadata=metadata)
        if format == "int8-convrot":
            checked = work / "checked.safetensors"
            update_quantizer_hash(staged, checked)
            staged.unlink()
            staged = checked
            if not api.verify_output(str(staged))["ok"]:
                raise ValueError("Packaged quantized output verification failed")
        with safe_open(str(staged), framework="pt", device="cpu") as packed:
            if set(packed.keys()) != set(state):
                raise ValueError("Bundle tensor inventory differs")
            for key, tensor in state.items():
                if not torch.equal(packed.get_tensor(key), tensor):
                    raise ValueError(f"Bundle tensor differs: {key}")
            if (
                hashlib.sha256(packed.get_tensor(ASSET_KEY).numpy().tobytes()).hexdigest()
                != json.loads(metadata["lfm3_vl"])["assets_sha256"]
            ):
                raise ValueError("Bundle asset hash differs")
        if digest(original) != source_hash:
            raise ValueError("Source weights changed during preparation")
        bundle_hash = digest(staged)
        # Hard-link publishes atomically and fails rather than overwriting a concurrent output.
        os.link(staged, output)
    return {
        "file": output.name,
        "bytes": output.stat().st_size,
        "sha256": bundle_hash,
        "source_sha256": source_hash,
        "format": format,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local upstream model directory")
    parser.add_argument("--output", required=True, help="New .safetensors artifact")
    parser.add_argument("--format", choices=("bf16", "int8-convrot"), default="int8-convrot")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quantizer-root", help="Optional quantizer source checkout")
    parser.add_argument("--convrot-groupsize", type=int, choices=(16, 64, 256))
    parser.add_argument("--exclude", action="append", default=[], help="Regex for weights to preserve")
    parser.add_argument("--reference-recipe", help="Validated Comfy-Org recipe; owns layer selection")
    parser.add_argument("--max-memory", default="512M")
    parser.add_argument(
        "--optimizer",
        choices=("none", "mse"),
        default="none",
        help="Optional weight-MSE clipping search; requires inference validation",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.output,
                format=args.format,
                device=args.device,
                quantizer_root=args.quantizer_root,
                convrot_groupsize=args.convrot_groupsize,
                exclude=args.exclude,
                reference_recipe=args.reference_recipe,
                max_memory=args.max_memory,
                optimizer=args.optimizer,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
