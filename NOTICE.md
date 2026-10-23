# Source attribution

- `architect/vendor/pixai.py`: PixAI Labs `pixai-tagger-v1.0`, revision
  `9fe10addf9326e292da8a85a98ea74cd91b41771`, `tagger_pipeline.py`.
  Apache-2.0. The upstream source attributes ViTDet to Meta/Detectron2 and its
  rotary embedding implementation to CodeLlama, rope-vit and rotary-embedding-torch.
  Local changes: omitted the unused Transformers pipeline class and auto-registration;
  the model forward and original preprocessing are preserved.
- `architect/vendor/minicpmv4_7/`: OpenBMB and Hugging Face, Transformers
  **v5.19.0**, configuration/modeling/processing files. Apache-2.0.
  Local changes: absolute imports into installed Transformers, and documentation
  for the precomputed multimodal-output argument. No shared library files changed.
- MiniCPM checkpoint: `saidutta69/MiniCPM-V-4.7-1B-heretic`, revision
  `f2ec4781396394599a524bd396133ed117b38da1`, derived from OpenBMB MiniCPM-V 4.7 1B.
  Apache-2.0. Model weights are stored separately and are not included in this package.
- Official Anima formatting: CircleStone Labs model card, revision
  `f973fc41ec7545364ac9776c2440285f43ff2a30`, reviewed 2026-10-23.
  This package does not bundle Anima weights.

New package code is licensed under Apache-2.0. See LICENSE.
