# Anima Prompt Architect for ComfyUI

Local prompt tools for [Anima](https://huggingface.co/circlestone-labs/Anima). Everything
runs on your machine, and nothing is downloaded while a workflow runs.

| Node | Input | What it does |
|---|---|---|
| **Anima Prompt Architect** | an image (plus optional changes) | Tags the image with PixAI, describes it with a small vision model and writes a matching Anima prompt. |
| **Anima Prompt Upsampler** | a short idea in words | Expands the idea into a full Anima prompt. Every tag is checked against the PixAI tag database, and characters, styles, camera words and placements are recognised. |

Both nodes write prompts in the official Anima format and support the Base, Aesthetic
and Turbo profiles. Both appear under `Anima/prompt`.

```text
idea:    Hatsune Miku singing on a stage, low angle, full body, watercolor

prompt:  masterpiece, best quality, score_7, safe, 1girl, solo, hatsune miku, vocaloid,
         full body, singing, stage, watercolor (medium), from below, aqua hair, twintails,
         aqua eyes, concert, contrast, colorful background, audience, stage lights

         Hatsune Miku sings passionately on a brightly lit stage, her vibrant aqua hair and
         twintails swaying as she performs, wearing a simple white outfit that contrasts with
         the colorful background. The audience is blurred in the distance, and the stage
         lights cast a warm glow on her face, highlighting her expressive eyes.
```

<sub>One sample run of the Upsampler. The tags and description are model-written and change with the seed.</sub>

## Contents

- [Install](#install)
- [Models](#models)
- [Image to prompt: Anima Prompt Architect](#image-to-prompt-anima-prompt-architect)
- [Text to prompt: Anima Prompt Upsampler](#text-to-prompt-anima-prompt-upsampler)
- [The Anima prompt format](#the-anima-prompt-format)
- [Precision and execution](#precision-and-execution)
- [Building the model files](#building-the-model-files)
- [Tests](#tests)
- [Limits](#limits)
- [License and attribution](#license-and-attribution)

## Install

1. Clone into `ComfyUI/custom_nodes`:

   ```bash
   git clone https://github.com/NidAll/ComfyUI-Anima-Prompt-Architect.git
   ```

2. Install `requirements.txt` with ComfyUI's own Python (Transformers 5.15.1 is
   pinned).
3. Use a recent ComfyUI with native `int8_tensorwise` ConvRot operations and a matching
   Comfy Kitchen CUDA backend.
4. Put the model files from the next section in `ComfyUI/models/anima_prompt_architect/`,
   then restart ComfyUI.

## Models

Each model is **one self-contained `.safetensors` file**. It embeds the configuration,
preprocessing and tokenizer data, which are checked against stored hashes before
loading. You don't need model folders, sidecar files or extra downloads.

| File | Used by | Get it |
|---|---|---|
| `pixai-tagger-v1.0-bf16.safetensors` (0.97 GB) | both nodes | [Download](https://huggingface.co/NidAll/Anima-Prompt-Architect-Models/resolve/main/pixai-tagger-v1.0-bf16.safetensors) |
| `minicpm-v4.7-1b-heretic-int8-convrot.safetensors` (2.66 GB) | image node | [Download](https://huggingface.co/NidAll/Anima-Prompt-Architect-Models/resolve/main/minicpm-v4.7-1b-heretic-int8-convrot.safetensors) |
| `lfm3-vl-2b-int8-convrot.safetensors` (experimental) | image node (optional), **required by the Upsampler** | Build it yourself: see [Building the model files](#building-the-model-files) |

```text
ComfyUI/models/anima_prompt_architect/
  pixai-tagger-v1.0-bf16.safetensors
  minicpm-v4.7-1b-heretic-int8-convrot.safetensors
  lfm3-vl-2b-int8-convrot.safetensors      (optional)
```

The `visual_model` list on the image node shows only files that really are MiniCPM or
LFM3-VL bundles, based on their embedded metadata. The Upsampler's `language_model` list
shows only LFM3-VL bundles.

## Image to prompt: Anima Prompt Architect

Import [the example workflow](example_workflows/Anima%20Prompt%20Architect.json), pick an
image and run it. API versions:
[MiniCPM](example_workflows/anima-prompt-architect-api.json),
[LFM3-VL](example_workflows/anima-lfm3-vl-api.json).

How it works:

1. **PixAI** scores all 30,877 tags of its vocabulary.
2. The **vision model** (MiniCPM or LFM3-VL) writes a short caption.
3. **Fusion** checks the caption against the tags and drops contradictions. The tags and
   the remaining sentences are then assembled in Anima order.

The models load one after another and are unloaded when the node finishes. The node
accepts one RGB image per run; split batches upstream.

**Outputs:** `positive`, `negative`, `analysis_json` (selected and rejected tags, with
reasons) and `raw_tags_json` (every tag probability). The final prompt is also shown on
the node.

### Controls

| Control | Effect |
|---|---|
| `variant` | **Base:** quality/score prefix and the recommended negative. **Aesthetic:** no score tags in either prompt. **Turbo:** empty negative, for its recommended CFG 1. |
| `detail` | `concise` / `balanced` / `detailed`: the tag budget and caption length. |
| `fidelity` | **strict:** higher identity/style thresholds, contradictions filtered, unsupported clothing words removed. **balanced:** keeps more of the caption. |
| `general_threshold`, `identity_threshold`, `style_threshold` | Tag confidence cut-offs. The defaults favour fewer uncertain tags. |
| `include_visual_details` | Adds the vision model's description after the tags. |
| `safety` | `auto` reads PixAI's rating. You can also force a rating or `omit` it. |
| `artist` | Explicit artist names, which get Anima's `@` prefix. Artists are **never guessed** from style. |
| `max_new_tokens`, `max_slices` | Caption length and the number of image crops (more crops show finer detail but take longer and use more memory). |

### Changing the picture (optional)

`direction` lets you clone an image *with changes*. You can write:

- **Tags:** `red hair, night`
- **Sentences:** `She holds a paper lantern in the rain.`
- **Removals:** `-smile` or `no hat`

Your words become PixAI tags in their proper slot. They replace the image tags they
contradict: hair or eye colour, hair length, expression, time, weather, framing, outfit
and subject count. A rating word (`safe`, `nsfw`…) sets the safety tag. `exclude_tags`
drops tags from the image analysis.

`direction_mode` controls how the description handles your changes:

- **lead** (default): the caption is rewritten in a second, text-only pass from an
  explicit edit list, and your tags come first. Clauses that still contradict a change
  are pruned. If the text never mentions one of your tags, one plain sentence is added
  for it.
- **follow:** the image caption is kept, and your tags and sentences come after it.
- **replace description:** the tags are kept, and only your sentences form the
  description.

## Text to prompt: Anima Prompt Upsampler

Type an idea, a handful of tags or a few sentences. The node returns a complete Anima
prompt. No image is needed, but the node does need the **LFM3-VL** model file, which is
used here as a text model. The API example is
[anima-prompt-upsampler-api.json](example_workflows/anima-prompt-upsampler-api.json).

The node works in four steps:

1. **Recognise.** The idea is matched against the PixAI tag database stored in the tagger
   file. The tagger itself is not run.
2. **Write.** LFM3-VL writes a rating, a tag list and a description, using what step 1
   found as facts.
3. **Check.** Every tag must exist in the PixAI vocabulary, and your own words always
   win.
4. **Assemble.** The tags and description are put in the official order. The output has
   at least two sentences of prose.

### What gets recognised

| You write | Becomes |
|---|---|
| `Hatsune Miku`, `reimu`, `Fern from Frieren` | the character tag (`hatsune miku`, `hakurei reimu`, `fern (sousou no frieren)`) |
| a named character | that character's series, found from the tagger's classifier head (`vocaloid`, `touhou`…) |
| `by bkub`, `in the style of …`, `… style` | `@bkub`, but only for artists the database knows |
| `watercolor`, `pixel art`, `90s` | `watercolor (medium)`, `pixel art`, `1990s (style)` |
| `low angle`, `bird's eye view`, `waist up`, `close-up` | `from below`, `from above`, `upper body`, `close-up` |
| `Rem on the left, Saber on the right` | kept word for word in the description |

Ordinary words stay ordinary. "A chain link fence" doesn't summon Link, and "fern leaves"
don't summon Fern. A lowercase single word is only read as a character when it is the
name of a well-known one.

### How your idea is protected

- **Your tags win.** They come first and replace the model tags they contradict (hair
  colour, time of day, weather…). Use `-tag` or `no tag` to remove a tag, and `@name` or
  the `artist` input to add an artist.
- **No invented names.** A character or series is kept only when your idea names it. The
  series of a named character is never added as a second character.
- **Character colours come from the character.** The model often guesses hair and eye
  colours wrongly. For a named character, those guesses are dropped from both tags and
  text unless you wrote the colour yourself or the tag data confirms it (Miku's aqua
  hair, for example).
- **Your phrasing is cleaned up.** Raw phrases such as `low angle` or `by bkub` are not
  left behind as tags once they have been mapped.
- **Animal-person ideas stay human.** For ideas like "fox girl", anthro tags such as
  `fur` are left out unless you ask for them.
- **Nothing is hidden.** `analysis_json` lists every recognised name, the facts given to
  the model, the model's raw answer, and every rejected tag or sentence with its reason.

### Controls

| Control | Effect |
|---|---|
| `variant`, `safety`, `artist` | As on the image node. `safety: auto` uses a rating word from your idea, then the model's rating, then `safe`. |
| `detail` | `concise` / `balanced` / `detailed`: the number of tags and sentences. |
| `temperature`, `seed` | `0` always gives the same expansion. Higher values vary the details the model adds. |
| `max_new_tokens` | Room for the model's answer. `detailed` benefits from about 512. |

## The Anima prompt format

Both nodes follow the
[official Anima recommendations](https://huggingface.co/circlestone-labs/Anima):

```text
quality/meta/safety → subject count → character → series → @artist → general tags

natural-language description (at least two sentences)
```

Tags are lowercase with spaces; `score_*` tags keep their underscores. Base uses
`masterpiece, best quality, score_7` plus the safety tag. Aesthetic drops the score tags.
No separate official quality-tag recipe is documented for Turbo, so it uses the general
convention with an empty negative.

## Precision and execution

- **PixAI** runs in BF16, with sigmoid in FP32. The loader rejects FP32 weights.
  PixAI's original aspect-preserving 1008 px preprocessing is kept.
- **MiniCPM** has 137 native INT8 ConvRot matrices. The vision tower, merger,
  embeddings/head, boundary blocks, normalization and sensitive recurrent controls stay
  BF16.
- **LFM3-VL** quantizes 142 language matrices with ConvRot group 256. Vision, projector
  and the shared embedding/head stay BF16.
- **INT8 is real integer compute.** ComfyUI's native operations dispatch to Comfy Kitchen
  integer kernels; the weights are not dequantized back to BF16 for matrix math.
- **Fusion keeps every byte.** Packed QKV and gate/up projections leave the quantized rows
  unchanged.
- **ComfyUI manages memory.** Its model patchers load and unload each model in turn.
  There is no global model cache and no change to ComfyUI core.

INT8 is not guaranteed to be faster than equally optimized BF16 on every GPU, and prompt
wording can differ between precisions. See [validation and limitations](docs/benchmarks.md).

## Building the model files

Building the files is optional for PixAI and MiniCPM, which are published. It is required
for LFM3-VL. All steps are offline, never overwrite an existing output and leave the
source checkpoints untouched.

```bash
# PixAI BF16 and MiniCPM INT8 from local original model folders
python scripts/prepare_models.py pixai --source /path/to/pixai-original \
  --output /path/to/pixai-tagger-v1.0-bf16.safetensors
python scripts/prepare_models.py minicpm --source /path/to/minicpm-original \
  --quantizer-root /path/to/comfyui-native-quantizer --device cuda \
  --output /path/to/minicpm-v4.7-1b-heretic-int8-convrot.safetensors

# LFM3-VL INT8 from a local copy of LiquidAI/LFM3-VL-2B
python scripts/prepare_lfm3.py --source /path/to/LFM3-VL-2B \
  --quantizer-root /path/to/comfyui-native-quantizer --device cuda \
  --convrot-groupsize 256 --output /path/to/lfm3-vl-2b-int8-convrot.safetensors
```

Quantization uses the
[ComfyUI Native Quantizer](https://github.com/NidAll/comfyui-native-quantizer). For
MiniCPM, a [conservative recipe](recipes/minicpm-v47.json) is included in case your
quantizer doesn't have the family built in. The LFM3 runtime is pinned to upstream
revision `f83d2b0a2296aaa277f394990a9b8b949ddf9691`.

## Tests

The portable suites need only Python, PyTorch, NumPy and safetensors, and they run in CI:

```bash
for t in fusion artifact direction upsample grounding; do
  python -m unittest discover -s tests -p "test_$t.py" -v
done
```

With ComfyUI and its dependencies installed:

```bash
python -m unittest discover -s tests -v
python -m pytest --import-mode=importlib tests/test_lfm3_runtime.py
python tests/test_native_cuda.py -v
```

The first command also checks loading and deployment validation. The second covers the
LFM3 runtime, and the third checks CUDA kernel correctness. `scripts/benchmark.py`
compares BF16 and INT8 artifacts and profiles the native integer execution.

## Limits

- **Model text can be wrong.** Descriptions, expansions and the vision model's
  observations are model-written and uncalibrated. Review prompts before relying on them.
- **Character looks only for famous characters.** The tagger's classifier head carries a
  usable look signal only for very common characters. Rarer ones get correct character
  and series tags, but no look hints.
- **Tags must be in the vocabulary.** Tags outside the PixAI vocabulary are always
  dropped, even when they are valid elsewhere.
- **LFM3-VL is experimental.** Video input is not supported.

## License and attribution

Package code and model derivatives are Apache-2.0; see [LICENSE](LICENSE). Upstream
sources and pinned revisions are listed in [NOTICE](NOTICE.md). The LFM3-VL weights are
under the **LFM Open License v1.0**; see
[`architect/lfm3/NOTICE.md`](architect/lfm3/NOTICE.md). Anima weights are not included
and keep their own license.
