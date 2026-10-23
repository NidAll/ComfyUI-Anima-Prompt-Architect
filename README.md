# ComfyUI Anima Prompt Architect

Turn an image into an Anima prompt with **PixAI Tagger v1.0 BF16** and
**MiniCPM-V 4.7 1B Heretic native INT8 ConvRot**. Everything runs locally.

The node returns positive and negative prompts, structured analysis JSON and all
30,877 raw PixAI tag probabilities. It supports Anima Base, Aesthetic and Turbo.

## Install

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/NidAll/ComfyUI-Anima-Prompt-Architect.git
```

Install `requirements.txt` with ComfyUI's Python environment. A recent ComfyUI
with native `int8_tensorwise` ConvRot operations and a compatible Comfy Kitchen
CUDA backend is required. The integration has been tested with Transformers
5.15.1; the small official MiniCPM 4.7 runtime is included in the package.

Download these two files from
[Hugging Face](https://huggingface.co/NidAll/Anima-Prompt-Architect-Models/tree/main):

| Model | Download | Approximate size |
|---|---|---:|
| PixAI BF16 | [pixai-tagger-v1.0-bf16.safetensors](https://huggingface.co/NidAll/Anima-Prompt-Architect-Models/resolve/main/pixai-tagger-v1.0-bf16.safetensors) | 0.97 GB |
| MiniCPM INT8 ConvRot | [minicpm-v4.7-1b-heretic-int8-convrot.safetensors](https://huggingface.co/NidAll/Anima-Prompt-Architect-Models/resolve/main/minicpm-v4.7-1b-heretic-int8-convrot.safetensors) | 2.66 GB |

Place both directly in:

```text
ComfyUI/models/anima_prompt_architect/
  pixai-tagger-v1.0-bf16.safetensors
  minicpm-v4.7-1b-heretic-int8-convrot.safetensors
```

Each file includes its weights and required configuration, preprocessing and
vocabulary/tokenizer assets. No model folders, sidecar JSON files, separate
projector or additional tokenizer download are needed. Embedded assets are data,
not executable code. The node never downloads models during execution.

Restart ComfyUI, import
[the example workflow](example_workflows/Anima%20Prompt%20Architect.json), select
an image and run **Anima Prompt Architect** under `Anima/prompt`.
The positive/negative outputs connect to standard text encoders. Preview Any
nodes display text and analysis. An [API graph](example_workflows/anima-prompt-architect-api.json)
is also included. One RGB image is accepted per execution; split batches upstream.

## Controls

- **Variant:** Base uses the recommended quality/score prefix and negative.
  Aesthetic omits score tags from both prompts. Turbo has an empty negative at
  its recommended CFG 1; the general positive-tag convention is retained.
- **Detail:** adjusts the tag budget. MiniCPM adds a short description of visible
  framing, spatial relationships, lighting and artistic details.
- **Fidelity:** strict uses stronger identity/style thresholds, filters defined
  contradictions and strips unsupported clothing modifiers. Balanced retains more
  visual observations.
- **Confidence thresholds:** control general, character/copyright and style tags.
  Defaults favor fewer uncertain tags; thresholds are not universal guarantees.
- **Artist:** explicit comma-separated artist names receive Anima's `@` prefix.
  Artist identities are never guessed automatically from style labels.
- **Safety:** infer an Anima safety tag from PixAI ratings, override it or omit it.
- **Slices:** increase visual crops for small details at a memory/time cost.
  The released 16x visual compression remains intact.

Tags use lowercase and spaces, preserving underscores in `score_*`. Ordering is
quality/meta/safety → subject count → character → series → artist → general tags,
followed by natural language. These rules follow the
[official Anima recommendations](https://huggingface.co/circlestone-labs/Anima).
There is no separately documented official Turbo quality-tag recipe.

## Precision and execution

PixAI's floating model weights and inference are BF16; sigmoid is FP32. The
loader rejects FP32 model weights. Original aspect-preserving 1008px preprocessing
is retained.

MiniCPM has 137 native INT8 ConvRot matrices. Its vision tower, visual merger,
embeddings/output head, boundary blocks, normalization and sensitive recurrent
controls remain BF16. FP32 row scales are quantization metadata. Native ComfyUI
operations dispatch into actual Kitchen integer kernels; this is accelerated
quantized execution, not an INT8 storage wrapper around dequantized matrix math.

ComfyUI patchers own sequential loading and unloading. Packed-byte gate/up and
QKV/Z fusion reduce projection calls; Kitchen accelerates recurrent decoding.
There is no persistent global model cache or ComfyUI core patch.

INT8 is not guaranteed to outperform equally optimized BF16 on every GPU. Prompt
wording can change between precisions. See [validation and limitations](docs/benchmarks.md).

## Reproduce the artifacts

Use local original model directories with their configuration/tokenizer assets:

```bash
python scripts/prepare_models.py pixai --source /path/to/pixai-original \
  --output /path/to/pixai-tagger-v1.0-bf16.safetensors
python scripts/prepare_models.py minicpm --source /path/to/minicpm-original \
  --quantizer-root /path/to/comfyui-native-quantizer --device cuda \
  --output /path/to/minicpm-v4.7-1b-heretic-int8-convrot.safetensors
```

The MiniCPM conversion uses
[ComfyUI Native Quantizer](https://github.com/NidAll/comfyui-native-quantizer),
with the included [conservative recipe](recipes/minicpm-v47.json) when the installed
quantizer does not yet have the built-in family. Source checkpoints are preserved;
existing outputs are never overwritten. Conversion is offline and optional.

## Tests and attribution

Run portable tests with `python -m unittest discover -s tests -p 'test_fusion.py' -v`
and `python -m unittest discover -s tests -p 'test_artifact.py' -v`.
With ComfyUI and its dependencies installed, `python -m unittest discover -s tests -v`
also checks loading and deployment validation. CUDA correctness tests run with
`python tests/test_native_cuda.py -v`. `scripts/benchmark.py` compares compatible
single-file BF16/INT8 artifacts and profiles actual native integer execution.

Code and model derivatives are Apache-2.0; see [LICENSE](LICENSE) and
[NOTICE](NOTICE.md) for upstream sources and pinned revisions. Anima weights are
not included and retain their own licensing terms.
