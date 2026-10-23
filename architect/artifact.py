"""Single-file safetensors artifacts with embedded, non-executable model assets."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ASSET_KEY = "__anima_prompt_architect_assets__"
FILES = {
    "pixai": ("config.json", "preprocessor_config.json"),
    "minicpm": ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"),
}
SOURCES = {
    "pixai": ("pixai-labs/pixai-tagger-v1.0", "9fe10addf9326e292da8a85a98ea74cd91b41771"),
    "minicpm": ("saidutta69/MiniCPM-V-4.7-1B-heretic", "f2ec4781396394599a524bd396133ed117b38da1"),
}


def pack(source, output, kind):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if kind not in FILES:
        raise ValueError("Unknown artifact kind")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as writer:
        for name in FILES[kind]:
            data = (source / name).read_bytes()
            if name.endswith(".json"):
                config = json.loads(data)
                config.pop("_name_or_path", None)
                data = json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode()
            if b"/home/" in data or b"/tmp/" in data:
                raise ValueError(f"Local path in model asset: {name}")
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            writer.writestr(info, data)
    payload = archive.getvalue()
    with safe_open(source / "model.safetensors", framework="pt") as reader:
        if ASSET_KEY in reader.keys():
            raise ValueError("Source is already packaged")
        state = {key: reader.get_tensor(key) for key in reader.keys()}
    state[ASSET_KEY] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    repo, revision = SOURCES[kind]
    metadata = {"format": "pt", "anima_prompt_architect": json.dumps({"version": 1, "kind": kind,
                "assets_sha256": hashlib.sha256(payload).hexdigest()}),
                "source_repo": repo, "source_revision": revision, "license": "apache-2.0"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    save_file(state, str(temporary), metadata=metadata)
    # Verify every model tensor before exposing the replacement.
    with safe_open(source / "model.safetensors", framework="pt") as original, safe_open(temporary, framework="pt") as packed:
        for key in original.keys():
            if not torch.equal(original.get_tensor(key), packed.get_tensor(key)):
                raise ValueError(f"Packaged weight differs: {key}")
    read_assets(temporary, kind)
    temporary.replace(output)
    return {"file": output.name, "bytes": output.stat().st_size,
            "sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(), "kind": kind}


def read_assets(path, kind):
    with safe_open(path, framework="pt") as reader:
        metadata = reader.metadata() or {}
        manifest = json.loads(metadata.get("anima_prompt_architect", "{}"))
        if manifest.get("version") != 1 or manifest.get("kind") != kind:
            raise ValueError(f"Expected a packaged {kind} Prompt Architect model")
        payload = reader.get_tensor(ASSET_KEY).numpy().tobytes()
    if len(payload) > 32 * 1024**2 or hashlib.sha256(payload).hexdigest() != manifest["assets_sha256"]:
        raise ValueError("Embedded model assets failed integrity validation")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if set(archive.namelist()) != set(FILES[kind]) or sum(item.file_size for item in archive.infolist()) > 64 * 1024**2:
            raise ValueError("Unexpected embedded model assets")
        return {name: archive.read(name).decode("utf-8") for name in FILES[kind]}
