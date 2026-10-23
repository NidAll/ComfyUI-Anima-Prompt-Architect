# Validation and limitations

## Verified behavior

- All 398 PixAI model tensors exactly match original FP32 values rounded to BF16.
  Deployment contains no FP32 model weights; sigmoid runs in FP32.
- MiniCPM contains 137 native INT8 ConvRot matrices and 643 retained BF16 tensors.
  The retained tensors are byte-identical to the original model.
- Packaging preserves every weight tensor. Required model assets are embedded
  in one safetensors file per model and validated before loading.
- Actual native INT8 operators and signed-integer CUDA kernels were profiled.
  Generation succeeded while INT8 weight dequantization was forbidden.
- Packed projection fusion preserves matrix bytes, scales and tested outputs.
  Native recurrence/state and normalization are checked against reference math.
- The complete node ran through the ComfyUI API and returned its four text outputs.
- Three synthetic fixtures exercised portrait/street composition, an illustrated
  animal scene and a shop scene, with paired reference-precision descriptions.

## What these checks do not establish

Quantization is not a universal throughput improvement. Native recurrent decode
can improve a complete pipeline while equally optimized BF16 remains faster than
INT8 on some devices. Measure your target device with `scripts/benchmark.py`;
keep token count, image, slicing and decoder optimizations paired.

Weight and logit similarity do not establish identical visual understanding.
Precision changes can alter wording and individual tag probabilities. Three
fixtures are not a labeled quality benchmark. Known-character recognition,
multiple-character binding, difficult OCR and downstream Anima reconstruction
quality remain unverified.

Both precisions can hallucinate details. Strict fusion filters defined conflicts
and unsupported clothing modifiers, but does not prove every caption statement.
VLM observations are unscored; matching PixAI tags corroborate nouns rather than
independently verifying relationships, lighting or medium. Explicit artist names
are supplied by the user, not inferred.

## Reproducible checks

`scripts/benchmark.py` records fixed-token timings, peak CUDA allocation,
paired prefill-logit comparisons and profiler kernel events. Its INT8 generation
runs forbid weight dequantization. Compare both a reference decoder and an equally
optimized BF16 decoder to distinguish runtime improvements from quantization.

Unit tests cover prompt profiles, conflict/redundancy rules, asset integrity,
model-file containment and deployment dtypes. CUDA tests cover recurrent state,
packed projection fusion and normalization. Upstream quantizer validation is
separate from this node's runtime and prompt-quality validation.
