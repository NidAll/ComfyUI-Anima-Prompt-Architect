"""Sequential vision execution; all model lifetimes belong to one node execution."""

import json
import re
import time

import numpy as np
from PIL import Image
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import comfy.model_management as mm

from .fusion import fuse
from .loading import load_tagger, load_minicpm
from .vendor.pixai import RescalePadProcessor

ANALYSIS_REQUEST = '''Describe this image in three precise sentences for an artist recreating it. Explain the composition and spatial relationships, then visible lighting and colors, then the background and artistic medium. Describe only visible details. Do not guess names, artists, franchise, intentions or unreadable text. Return only the description, without reasoning, headings or a list.'''


class AnalysisComplete(StoppingCriteria):
    def __init__(self, tokenizer, input_length):
        self.tokenizer = tokenizer
        self.input_length = input_length

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.input_length:], skip_special_tokens=True)
        return "<think>" in text or (len(text) > 80 and ("\n\n" in text or len(re.findall(r"[.!?](?:\s|$)", text)) >= 3))


def pil_image(image):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError("Prompt Architect accepts one RGB IMAGE; split image batches before analysis.")
    return Image.fromarray((image[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8))


def tag_image(image, bundle):
    started = time.perf_counter()
    owner, config = load_tagger(bundle)
    try:
        device = owner.load_device
        mm.load_models_gpu([owner], memory_required=512 * 1024**2)
        pixels = RescalePadProcessor(size=config.img_size)(images=image, return_tensors="pt")["pixel_values"]
        logits = owner.model(pixels.to(device=device, dtype=torch.bfloat16))
        probabilities = logits.float().sigmoid().cpu()[0]
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("PixAI produced non-finite probabilities.")
        results, offset = {}, 0
        for category, count in config.tags_split:
            results[category] = {tag: float(score) for tag, score in
                                zip(config.tags[offset:offset + count], probabilities[offset:offset + count])}
            offset += count
        return results, {"seconds": time.perf_counter() - started, "precision": "BF16 weights/forward, FP32 sigmoid"}
    finally:
        mm.unload_model_and_clones(owner)


def analyze_image(image, bundle, max_new_tokens=512, max_slices=4, reference=False):
    started = time.perf_counter()
    owner, processor, coverage = load_minicpm(bundle, reference=reference)
    try:
        mm.load_models_gpu([owner], memory_required=640 * 1024**2, force_full_load=True)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ANALYSIS_REQUEST}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", max_slice_nums=max_slices)
        inputs = inputs.to(owner.load_device, dtype=torch.bfloat16)
        tokens = owner.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                    use_cache=True, pad_token_id=processor.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([AnalysisComplete(processor.tokenizer, inputs["input_ids"].shape[1])]))
        generated = processor.tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        analysis = {"caption": generated.split("<think>")[0].strip(), "facts": []}
        return analysis, {"seconds": time.perf_counter() - started, "native_int8_layers": coverage,
                          "generated_tokens": tokens.shape[1] - inputs["input_ids"].shape[1], "raw_text": generated}
    finally:
        mm.unload_model_and_clones(owner)


def architect(image, tagger_bundle, minicpm_bundle, **controls):
    image = pil_image(image)
    tags, tagger_metrics = tag_image(image, tagger_bundle)
    analysis, visual_metrics = analyze_image(image, minicpm_bundle,
                max_new_tokens=controls.pop("max_new_tokens", 512), max_slices=controls.pop("max_slices", 4))
    positive, negative, structured = fuse(tags, analysis, **controls)
    structured["metrics"] = {"pixai": tagger_metrics, "minicpm": visual_metrics}
    return positive, negative, json.dumps(structured, ensure_ascii=False, indent=2), json.dumps(tags, ensure_ascii=False, indent=2)
