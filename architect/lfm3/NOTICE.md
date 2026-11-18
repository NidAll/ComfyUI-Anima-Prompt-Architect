LFM3-VL source files in `vendor/` are pinned from
LiquidAI/LFM3-VL-2B revision f83d2b0a2296aaa277f394990a9b8b949ddf9691.
Python files retain their upstream Apache-2.0 notices; the Apache license is
in `vendor/LICENSE-APACHE-2.0`. The upstream repository's LFM Open License
v1.0 is retained separately in `vendor/LICENSE` and governs the model weights.

Local changes adapt model-specific configuration validation to Transformers
5.15, preserving scoped RoPE and convolution types, and drop media grids when
cached generation drops their corresponding pixel tensors. The native adapter
uses ComfyUI operations and its selected attention dispatch, materializes causal
and sliding masks for that dispatch, shares the tied embedding/head, and fuses
compatible INT8 QKV and MLP projections without changing their quantized rows.
Video runtime support has not been accepted in these image nodes.
