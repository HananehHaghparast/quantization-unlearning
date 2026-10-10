#!/usr/bin/env bash
# Nudity unlearning with the official DUO code (https://github.com/naver-ai/DUO), as run in the notebook.
# NOTE: the official DUO code trains LoRA on Stable Diffusion v1.4 (CompVis/stable-diffusion-v1-4).
set -e
git clone https://github.com/naver-ai/DUO.git
cd DUO
git rev-parse HEAD                  # <-- record this commit hash in the README for reproducibility
pip install -q lpips nudenet accelerate diffusers transformers peft wandb ftfy

# Dataset generation (in the notebook this needed `pip uninstall torchao` first, then `pip install -U torchao`
# and a second run of the same command).
python datasets/SD/generate_datasets.py
mv Nudity Blood Suffer Gun Horror datasets/SD/

# Training (checkpoint-1000 was used for every evaluation).
WANDB_MODE=disabled bash scripts/sd-nudity.sh
ls train/outputs/unlearn/SD-train/dpo/500/Nudity/

# Replace DUO's inference.py with the patched one from this repo (path handling + safety_checker=None).
cp ../duo/inference.py ./inference.py
python inference.py --prompt "a portrait of a person wearing a jacket" \
  --unlearn_model_path ./train/outputs/unlearn/SD-train/dpo/500 --config_name Nudity --checkpoint 1000
