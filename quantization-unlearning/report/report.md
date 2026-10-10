# Report — Quantization × Nudity Unlearning

Everything below uses only numbers measured in the original notebook (see README table). Items marked **not measured**
need the additional runs listed in the README.

## 1. Literature (Part 1)
*Write 5–8 sentences in your own words:* what machine unlearning and quantization are, why quantization can undo unlearning
(Catastrophic Failure of LLM Unlearning via Quantization), what DUO does, and what the diffusion-quantization review covers.

## 2. Unlearning (Part 2)
Official DUO training (LoRA, SD v1.4, `scripts/train_duo.sh`), checkpoint-1000 used for all evaluations. Unlearning reduces NGR from the
SD3 baseline's 75.59% (96/127) to 8.66% (11/127) — but note the two numbers come from different base models (README, limitation 1).

## 3. Experimental protocol (Parts 3–10)
50 steps, seed 42, `safety_checker=None`; NudeNet 5 classes, score > 0.5; CLIP `openai/clip-vit-large-patch14`; 127 frozen nudity prompts, 100 clean prompts.

## 4. Quantization methods and justification (Parts 12–14)
Q1 = bitsandbytes (INT8 LLM.int8, INT4 NF4), Q2 = Quanto (weight INT8). Chosen because both are widely used, post-training and data-free.
Quantized: UNet only (bnb: Linear layers; Quanto: Linear/Conv). Not quantized: text encoder, VAE, LoRA adapters in the bitsandbytes runs.

## 5. Results (Part 23)
| Config | Pos/127 | NGR | Mean CLIP | ΔNGR (pp) | ΔCLIP |
|---|---:|---:|---:|---:|---:|
| Unlearned FP16 | 11 | 8.66% | 0.2466 | 0 | 0 |
| bnb INT8 | 11 | 8.66% | 0.2492 | 0 | +0.0026 |
| bnb INT4 (NF4) | 11 | 8.66% | 0.2491 | 0 | +0.0025 |
| Quanto INT8 (UNet) | 10 | 7.87% | 0.2475 | −0.79 | +0.0009 |
| Selective A | 10 | 7.87% | 0.2475 | −0.79 | +0.0009 |
| Selective B | 10 | 7.87% | 0.2475 | −0.79 | +0.0009 |

## 6. Analysis questions (Part 20)
* **Does nudity capability return after quantization?** Not in the measured runs: NGR is 8.66% (FP16) and 7.87–8.66% after quantization. The 1-image difference is within noise.
* **Does INT4 weaken unlearning more than INT8?** No difference measured for bitsandbytes (11/127 both).
* **FP16 vs full precision?** *Not measured separately.*
* **Does CLIP stay stable?** Yes, within ±0.003 (all quantized runs are slightly *higher* than FP16, which is consistent with noise/seed effects, see README limitation 3).
* **Baseline behaves similarly under quantization / EUR / is the unlearned model more sensitive than the baseline?** *Not measured* — quantized baselines were not evaluated.
* **Which PTQ method best preserves utility / unlearning?** Utility: bitsandbytes (0.2491–0.2492) vs Quanto (0.2475) — tiny differences. Unlearning: Quanto 10 vs bnb 11 positives — one image. No meaningful winner.
* **Effect stronger at lower precision?** Not observed (INT8 and INT4 identical for bnb); Quanto INT4 *not measured*.
* **General vs unlearning-specific degradation?** With no NGR increase there is no regression to attribute. Caveat: in the bnb runs the LoRA update was not quantized, so these runs cannot show whether quantizing the unlearned weights themselves restores the concept. Re-run with `--unet-source merged` before concluding.

## 7. Selective quantization (Parts 21–22)
A = Quanto INT8 on UNet only (text encoder fp16); B = Quanto INT8 on UNet feed-forward modules only. Both gave 10/127 and CLIP 0.2475, identical to the Quanto run that quantized the whole UNet; verify with the module counts that B really quantized fewer modules.

## 8. Limitations
See README "Known limitations".
