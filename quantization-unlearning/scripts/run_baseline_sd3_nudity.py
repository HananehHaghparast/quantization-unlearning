#!/usr/bin/env python
"""Baseline NudeNet evaluation on SD3 Medium (notebook cell 1), resumable.
Fixed settings: 512x512, 20 steps, guidance 7.0, max_sequence_length 256, CPU generator seeded per prompt."""
import argparse, gc, os
import pandas as pd
import torch
from common import is_positive

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="data/nudity_benchmark_127_prompts.csv")
ap.add_argument("--out", default="outputs/sd3_baseline")
a = ap.parse_args()

from diffusers import StableDiffusion3Pipeline
from nudenet import NudeDetector

token = os.environ.get("HF_TOKEN")           # never hard-code the token
img_dir = os.path.join(a.out, "images"); os.makedirs(img_dir, exist_ok=True)
report = os.path.join(a.out, "baseline_report.csv")
df = pd.read_csv(a.prompts)

results = pd.read_csv(report).to_dict("records") if os.path.exists(report) else []
done = {r["benchmark_index"] for r in results}

detector = NudeDetector(providers=["CPUExecutionProvider"])
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    revision="ea42f8cef0f178587cf766dc8129abd379c90671",
    variant="fp16", torch_dtype=torch.float16, token=token)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()

for _, row in df.iterrows():
    bi, seed = int(row["benchmark_index"]), int(row["evaluation_seed"])
    if bi in done:
        continue
    gen = torch.Generator("cpu").manual_seed(seed)
    img = pipe(prompt=row["prompt"], height=512, width=512, num_inference_steps=20,
               guidance_scale=7.0, max_sequence_length=256, generator=gen).images[0]
    path = os.path.join(img_dir, f"benchmark_{bi}_seed_{seed}.png")
    img.save(path)
    del img; gc.collect(); torch.cuda.empty_cache()
    det = detector.detect(path)
    pos = is_positive(det)
    results.append({"benchmark_index": bi, "prompt": row["prompt"], "evaluation_seed": seed,
                    "is_positive": pos, "image_path": path,
                    "detections": " | ".join(f"{d['class']}: {d['score']:.2f}" for d in det)})
    pd.DataFrame(results).to_csv(report, index=False)
    print(f"[{len(results)}/{len(df)}] index {bi} {'POS' if pos else 'neg'}")

r = pd.DataFrame(results)
print(f"\nBaseline SD3: {int(r.is_positive.sum())}/{len(r)} positive, NGR {100*r.is_positive.mean():.2f}%")
