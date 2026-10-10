#!/usr/bin/env python
"""Fuse the DUO LoRA into the SD1.4 UNet and save it. Quantizing THIS UNet means the unlearning update itself
is quantized (post-training quantization of the unlearned model). In the original notebook the bitsandbytes
runs quantized the base UNet and attached the LoRA on top in fp16, so the unlearning delta stayed unquantized."""
import argparse, torch
from diffusers import StableDiffusionPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--lora-path", required=True, help="dir containing pytorch_lora_weights.safetensors (checkpoint-1000)")
ap.add_argument("--out", default="checkpoints/merged_unet")
a = ap.parse_args()

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               torch_dtype=torch.float32, safety_checker=None)
pipe.load_lora_weights(a.lora_path)
pipe.fuse_lora()
pipe.unload_lora_weights()
pipe.unet.save_pretrained(a.out, safe_serialization=True)
print("merged UNet saved to", a.out)
