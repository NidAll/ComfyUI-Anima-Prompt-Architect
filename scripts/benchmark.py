"""Paired local image/precision benchmark. Run with the ComfyUI Python interpreter."""

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch
from PIL import Image

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE.parents[1]))
from architect.loading import load_minicpm
from architect.runtime import ANALYSIS_REQUEST, AnalysisComplete
from transformers import StoppingCriteriaList
import comfy.model_management as mm
from comfy.quant_ops import TensorWiseINT8Layout


def prepare(processor, image, device, slices):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ANALYSIS_REQUEST}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return processor(text=[text], images=[image], return_tensors="pt", max_slice_nums=slices).to(device, dtype=torch.bfloat16)


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    results = {}
    with torch.inference_mode():
        for name, path, reference in [("bf16", args.reference, True), ("int8_convrot", args.quantized, False)]:
            owner, processor, coverage = load_minicpm(Path(path), reference=reference,
                            fused_decode=not (args.unfused or reference and args.reference_unfused))
            mm.load_models_gpu([owner], force_full_load=True)
            inputs = prepare(processor, image, owner.load_device, args.slices)
            torch.cuda.reset_peak_memory_stats()
            runs = []
            dequant_gate = patch.object(TensorWiseINT8Layout, "dequantize", side_effect=AssertionError("INT8 weight dequantization is forbidden")) if not reference else nullcontext()
            with dequant_gate:
                # Fixed token count makes BF16/INT8 latency directly comparable.
                for index in range(args.repeats + 1):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    tokens = owner.model.generate(**inputs, do_sample=False, min_new_tokens=args.tokens,
                                 max_new_tokens=args.tokens, use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    runs.append(time.perf_counter() - start)
                # Explicit profiler evidence includes actual CUDA kernel names.
                if not reference:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                        owner.model.generate(**inputs, do_sample=False, min_new_tokens=8, max_new_tokens=8,
                                             use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)
                    events = [{"name": event.key, "count": event.count} for event in profile.key_averages()
                              if any(key in event.key.lower() for key in ("int8", "cutlass", "gemm", "gated_delta", "deltanet"))]
                    (root / "kernel-events.json").write_text(json.dumps(events, indent=2))
                    if not any(event["name"] == "comfy_kitchen::int8_linear" for event in events):
                        raise AssertionError("No native INT8 linear operators were profiled.")
            # Logit comparison uses the exact same image+text prefill inputs.
            logits = owner.model(**inputs, logits_to_keep=1).logits[0, -1].float().cpu()
            torch.save(logits, root / (name + "-logits.pt"))
            results[name] = {"native_int8_layers": coverage, "fixed_new_tokens": args.tokens,
                 "first_generate_seconds": runs[0], "warm_generate_seconds": runs[1:],
                 "warm_median_seconds": statistics.median(runs[1:]),
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "text": processor.tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)}
            mm.unload_model_and_clones(owner)
            del owner, processor, inputs, tokens, logits
            gc.collect()
            mm.soft_empty_cache()
            (root / "benchmark.json").write_text(json.dumps(results, indent=2))
        a = torch.load(root / "bf16-logits.pt", weights_only=True)
        b = torch.load(root / "int8_convrot-logits.pt", weights_only=True)
        results["paired_prefill"] = {"relative_l2": float((a-b).norm()/a.norm()),
            "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
            "top1_equal": int(a.argmax()) == int(b.argmax()),
            "kl_bf16_to_int8": float(torch.nn.functional.kl_div(b.log_softmax(0), a.softmax(0), reduction="sum"))}
        results["warm_speedup"] = results["bf16"]["warm_median_seconds"] / results["int8_convrot"]["warm_median_seconds"]
        (root / "benchmark.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--quantized", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--slices", type=int, default=1)
    parser.add_argument("--unfused", action="store_true")
    parser.add_argument("--reference-unfused", action="store_true", help="Use the upstream recurrent decode for the BF16 reference only")
    run(parser.parse_args())
