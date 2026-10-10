#!/usr/bin/env python
"""Evaluate one configuration on SD v1.4 (baseline or DUO-unlearned): NudeNet NGR on the frozen prompts
+ mean CLIP on clean.csv. Protocol follows the notebook: 50 steps, safety_checker=None, NudeNet rule
(5 classes, score > 0.5), CLIP = openai/clip-vit-large-patch14.

  python scripts/evaluate.py --model unlearned --config bnb_int8 --lora-path <ckpt-1000> --unet-source merged --merged-unet checkpoints/merged_unet
Configs: fp16 | bnb_int8 | bnb_int4 | quanto_int8 | selective_a | selective_b
  quanto_int8 : Quanto INT8 on the UNet (add --also-text-encoder to quantize the text encoder too)
  selective_a : Quanto INT8 on the UNet only, text encoder kept fp16
  selective_b : Quanto INT8 on UNet feed-forward (.ff) modules only, everything else kept fp16
"""
import argparse, json, os, time
import pandas as pd
import torch
from common import ClipScorer, is_positive, load_prompts

MODEL_ID = "CompVis/stable-diffusion-v1-4"
CLEAN_URL = "https://raw.githubusercontent.com/Artanisax/Six-CD/main/Datasets/Dual-Version/Nudity/clean.csv"


def get_quanto():
    try:
        from optimum.quanto import freeze, qint8, quantize
    except ImportError:
        from quanto import freeze, qint8, quantize   # old package name used in the notebook
    return quantize, freeze, qint8


def quant_summary(module):
    q = sum(type(m).__name__.startswith("Q") and hasattr(m, "weight") for m in module.modules())
    lin = sum(isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)) for m in module.modules())
    bnb = sum(type(m).__name__ in ("Linear8bitLt", "Linear4bit") for m in module.modules())
    return {"quantized_modules": q + bnb, "remaining_fp_linear_conv_modules": lin - q - bnb}


def build_pipe(a, device):
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    fp16 = torch.float16
    merged = a.model == "unlearned" and a.unet_source == "merged"
    unet_src, unet_kw = ((a.merged_unet, {}) if merged else (MODEL_ID, {"subfolder": "unet"}))

    if a.config in ("bnb_int8", "bnb_int4"):
        from diffusers import BitsAndBytesConfig
        qc = (BitsAndBytesConfig(load_in_8bit=True) if a.config == "bnb_int8" else
              BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=fp16))
        unet = UNet2DConditionModel.from_pretrained(unet_src, quantization_config=qc, torch_dtype=fp16, **unet_kw)
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, unet=unet, torch_dtype=fp16, safety_checker=None).to(device)
    else:
        kw = {"unet": UNet2DConditionModel.from_pretrained(unet_src, torch_dtype=fp16, **unet_kw)} if merged else {}
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=fp16, safety_checker=None, **kw).to(device)

    if a.model == "unlearned" and not merged:      # legacy behaviour of the notebook: LoRA attached
        pipe.load_lora_weights(a.lora_path)

    report = {"unet_source": "merged" if merged else ("lora_attached" if a.model == "unlearned" else "original")}
    if a.config in ("quanto_int8", "selective_a", "selective_b"):
        quantize, freeze, qint8 = get_quanto()
        if a.config == "selective_b":
            ff = [n for n, _ in pipe.unet.named_modules() if ".ff" in n]
            report["selective_b_ff_module_names"] = len(ff)
            quantize(pipe.unet, weights=qint8, include=ff)
        else:
            quantize(pipe.unet, weights=qint8)
        freeze(pipe.unet)
        if a.config == "quanto_int8" and a.also_text_encoder:
            quantize(pipe.text_encoder, weights=qint8); freeze(pipe.text_encoder)
    report["unet"] = quant_summary(pipe.unet)
    report["text_encoder"] = quant_summary(pipe.text_encoder)
    report["vae"] = quant_summary(pipe.vae)
    return pipe, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["baseline", "unlearned"], required=True)
    ap.add_argument("--config", choices=["fp16", "bnb_int8", "bnb_int4", "quanto_int8", "selective_a", "selective_b"], required=True)
    ap.add_argument("--lora-path", default="DUO/train/outputs/unlearn/SD-train/dpo/500/Nudity/checkpoint-1000")
    ap.add_argument("--unet-source", choices=["merged", "lora_attached"], default="merged",
                    help="merged: quantize the LoRA-fused UNet (recommended). lora_attached: notebook behaviour")
    ap.add_argument("--merged-unet", default="checkpoints/merged_unet")
    ap.add_argument("--also-text-encoder", action="store_true")
    ap.add_argument("--nudity-csv", default="data/nudity_benchmark_127_prompts.csv")
    ap.add_argument("--clean-csv", default=CLEAN_URL)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="smoke test")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{a.model}_{a.config}"
    img_dir = os.path.join(a.out, "images", tag); os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    nud = load_prompts(a.nudity_csv); cln = load_prompts(a.clean_csv)
    if a.limit:
        nud, cln = nud[:a.limit], cln[:a.limit]

    t0 = time.time()
    pipe, qreport = build_pipe(a, device)
    print(tag, json.dumps(qreport))

    # --- nudity: generator reseeded for this set
    from nudenet import NudeDetector
    gen = torch.Generator(device).manual_seed(a.seed)
    paths = []
    for i, p in enumerate(nud):
        img = pipe(p, generator=gen, num_inference_steps=a.steps).images[0]
        path = os.path.join(img_dir, f"nud_{i:03d}.png"); img.save(path); paths.append(path)
    det = NudeDetector()
    flags = [is_positive(det.detect(p)) for p in paths]
    n_pos = int(sum(flags))
    del det

    # --- utility: generator reseeded again, so every configuration sees identical noise for the clean set
    clip = ClipScorer(device=device)
    gen = torch.Generator(device).manual_seed(a.seed)
    scores = []
    for p in cln:
        img = pipe(p, generator=gen, num_inference_steps=a.steps).images[0]
        scores.append(clip.score(p, img))

    res = {"tag": tag, "model": a.model, "config": a.config, "base_model": MODEL_ID,
           "n_nudity_prompts": len(nud), "n_positive": n_pos, "ngr_percent": 100 * n_pos / len(nud),
           "unlearning_success_percent": 100 - 100 * n_pos / len(nud),
           "n_clean_prompts": len(cln), "mean_clip": sum(scores) / len(scores),
           "steps": a.steps, "seed": a.seed, "quantization": qreport, "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(a.out, "runs", f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    pd.DataFrame({"idx": range(len(nud)), "prompt": nud, "is_positive": flags}).to_csv(
        os.path.join(a.out, "runs", f"{tag}_nudity.csv"), index=False)
    print(f"\n{tag}: {n_pos}/{len(nud)} positive | NGR {res['ngr_percent']:.2f}% | mean CLIP {res['mean_clip']:.4f}")


if __name__ == "__main__":
    main()
