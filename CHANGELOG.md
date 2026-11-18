# Changelog

## 0.3.0

- New **Anima Prompt Upsampler** node: a short idea becomes a full Anima prompt in the
  official order, written by LFM3-VL and checked tag by tag against the PixAI vocabulary.
- Grounding from the tagger file: characters, series (from the classifier head), artists,
  medium/style, camera/framing and placements are recognised before the model writes.
  Model hair/eye colours for named characters are dropped unless confirmed, and anthro
  tags are kept out of animal-person ideas unless asked for.
- Image node: optional `direction` (tags, sentences, `-tag` removals), `direction_mode`
  (lead / follow / replace description) and `exclude_tags`; in lead mode the caption is
  rewritten from an explicit edit list. The final prompt is shown on the node.
- Experimental **LFM3-VL-2B** visual model with a package-local runtime pinned to an
  upstream revision, native INT8 ConvRot loading and `scripts/prepare_lfm3.py`.
- Fixed the eye-state conflict group to use the real tags `closed eyes` / `open eyes`.
- CI also runs the direction, upsampler and grounding suites.

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
