#!/usr/bin/env bash
# Order of experiments. Evaluate every config for BOTH models so EUR can be computed.
set -e
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in the environment}
LORA=DUO/train/outputs/unlearn/SD-train/dpo/500/Nudity/checkpoint-1000

bash scripts/train_duo.sh                                 # 1. unlearning (SD1.4 + DUO)
python scripts/run_baseline_sd3_nudity.py                 # 2. SD3 baseline NudeNet (127 prompts)
python scripts/merge_lora_unet.py --lora-path $LORA       # 3. fuse LoRA so PTQ hits the unlearned weights

for M in baseline unlearned; do                           # 4. all configurations, both models
  for C in fp16 bnb_int8 bnb_int4 quanto_int8 selective_a selective_b; do
    python scripts/evaluate.py --model $M --config $C --lora-path $LORA --unet-source merged
  done
done
python scripts/make_plots.py
