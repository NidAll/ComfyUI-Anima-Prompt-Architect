import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architect.artifact import ASSET_KEY, pack, read_assets


class ArtifactTests(unittest.TestCase):
    def source(self, directory):
        source = Path(directory) / "source"
        source.mkdir()
        save_file({"weight": torch.randn(4, 4, dtype=torch.bfloat16)}, str(source / "model.safetensors"), metadata={"private_path":"/private/source"})
        (source / "config.json").write_text('{"hidden_size":4}')
        (source / "preprocessor_config.json").write_text('{"size":1008}')
        return source

    def test_self_contained_roundtrip_and_clean_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            target = Path(directory) / "model.safetensors"
            receipt = pack(source, target, "pixai")
            self.assertEqual(json.loads(read_assets(target, "pixai")["config.json"])["hidden_size"], 4)
            self.assertEqual(receipt["bytes"], target.stat().st_size)
            with safe_open(target, framework="pt") as reader:
                self.assertNotIn("private_path", reader.metadata())
                self.assertEqual(reader.get_slice(ASSET_KEY).get_dtype(), "U8")
            with self.assertRaises(ValueError):
                read_assets(target, "minicpm")
            with self.assertRaises(FileExistsError):
                pack(source, target, "pixai")

    def test_embedded_assets_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            target = Path(directory) / "model.safetensors"
            pack(source, target, "pixai")
            with safe_open(target, framework="pt") as reader:
                state = {key: reader.get_tensor(key).clone() for key in reader.keys()}
                metadata = reader.metadata()
            state[ASSET_KEY][0] ^= 1
            save_file(state, str(target), metadata=metadata)
            with self.assertRaisesRegex(ValueError, "integrity"):
                read_assets(target, "pixai")


if __name__ == "__main__":
    unittest.main()
