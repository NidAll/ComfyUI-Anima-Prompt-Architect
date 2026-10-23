# Changelog

## 0.2.1

- Declare NumPy in the portable CI environment used by safetensors and asset decoding.

## 0.2.0

- Public release with one self-contained safetensors file per model.
- Embedded configuration, preprocessing and tokenizer assets with integrity checks.
- Native ComfyUI file discovery and sequential PixAI BF16 / MiniCPM INT8 inference.
- Base, Aesthetic and Turbo prompt profiles; confidence and fidelity controls.
- Conversion recipes, workflows, portable tests and CUDA correctness checks.

Existing folder-based development bundles must be replaced by the published
single-file models and the updated example workflow.
