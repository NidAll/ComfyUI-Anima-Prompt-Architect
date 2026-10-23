from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]
if not torch.cuda.is_available():
    from comfy.cli_args import args
    args.cpu = True
from architect.loading import load_state, resolve_bundle


class DeploymentTests(unittest.TestCase):
    def test_fp32_tagger_weight_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            save_file({'weight': torch.ones(2, 2, dtype=torch.float32)}, str(path))
            with self.assertRaisesRegex(ValueError, 'must all be BF16'):
                load_state(torch.nn.Linear(2, 2), str(path), require_bf16=True)

    def test_dense_minicpm_deployment_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            save_file({'weight': torch.ones(2, 2, dtype=torch.bfloat16)}, str(path))
            with self.assertRaisesRegex(ValueError, 'must contain native INT8'):
                load_state(torch.nn.Linear(2, 2), str(path), require_int8=True)

    def test_bundle_path_traversal_rejected(self):
        for name in ('../secret', '..', '/tmp/model', 'folder\\model'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_bundle(name)


if __name__ == '__main__':
    unittest.main()
