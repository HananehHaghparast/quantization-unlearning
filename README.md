# Quantization × Machine Unlearning in Text-to-Image Models

## Overview

This repository contains the implementation and experimental setup for studying the interaction between **machine unlearning** and **post-training quantization (PTQ)** in text-to-image diffusion models. The work addresses the following central question:

> **What happens to a model's learned forgetting behavior after quantization?**

More specifically, when a text-to-image model has been trained to forget a **target concept**, we investigate whether post-training quantization weakens that unlearning and causes the target concept to reappear. A secondary question is whether any observed degradation in unlearning performance can be attributed to general model degradation or specifically to the loss of the unlearning effect.

Throughout this README, the forgotten concept is referred to generically as **Concept X**. The evaluation pipeline is concept-agnostic and can be adapted to other targeted concepts.

---

## What is Machine Unlearning?

**Machine unlearning** refers to the process of removing or suppressing specific knowledge, concepts, or behaviors from a trained machine learning model. Unlike retraining from scratch (which is often computationally prohibitive), unlearning aims to modify a pre-trained model so that it no longer generates or relies on certain target information.

In the context of text-to-image generative models, unlearning is used for **safety and content moderation**—for example, to prevent a model from generating a targeted class of content. The goal is to preserve the model's general generative capabilities (utility) while selectively erasing the targeted concept.

---

## Project Aim

The project investigates how **post-training quantization**—a technique that reduces model precision (e.g., from FP32 to INT8 or INT4) to improve efficiency—affects the forgetting behavior established by unlearning. Quantization can introduce numerical noise and alter model weights, potentially undoing the unlearning effect.

The work involves:

1. Producing a Concept-X–unlearned text-to-image model using a state-of-the-art unlearning method.
2. Defining a fixed evaluation protocol using a **concept detector** (for measuring Concept X generation) and **CLIP** (for utility preservation).
3. Applying multiple post-training quantization methods at different precisions (FP16, INT8, INT4).
4. Conducting selective quantization experiments to identify which model components are most sensitive to precision reduction.
5. Analyzing the relationship between general model degradation and unlearning-specific degradation.

> **Note:** This README describes the methodology and work performed. Full experimental results and analysis are provided in [`report/REPORT.md`](report/REPORT.md).

---

## Repository Structure

```
.
├── configs/                    # Configuration files for unlearning and quantization
├── data/                       # Frozen prompt sets and utility evaluation prompts
├── scripts/                    # End-to-end scripts for each stage
├── src/
│   ├── unlearning/             # Unlearning method implementations
│   ├── quantization/           # PTQ method implementations
│   ├── evaluation/             # Concept-detector and CLIP evaluation utilities
│   └── utils/                  # Shared helpers
├── results/                    # Generated images, metrics, and plots (gitignored)
├── report/                     # Full experimental report
└── README.md
```

---

## Methodology

### 1. Model and Unlearning Method

We selected **[Track A: Stable Diffusion 3 / Track B: FLUX.1-dev]** and applied the corresponding unlearning method:

- **Track A:** *Direct Unlearning Optimization (DUO)* for Stable Diffusion 3.
- **Track B:** *EraseAnything* for FLUX.1-dev.

The unlearning procedure produces a checkpoint with a substantially reduced ability to generate the target concept (Concept X), while preserving general generation quality.

**Deliverables from this stage:**
- Original baseline checkpoint.
- Concept-X–unlearned checkpoint.
- Reproducible unlearning scripts and configuration files.

### 2. Fixed Inference Configuration

To ensure fair comparison across all checkpoints, we define a single inference configuration used for every generation:

- Image resolution, inference steps, guidance scale, scheduler, and sampler are fixed.
- A fixed set of random seeds is used.
- Prompts, seeds, and all other generation parameters remain identical across baseline, unlearned, quantized, and selective-quantization models.

Any unavoidable deviations are documented in the report.

### 3. Concept Evaluation Protocol

Concept X detection is performed using a **pre-trained concept detector** with the following fixed rule:

- **Target classes:** A fixed list of concept-specific detector classes is defined (see `src/evaluation/concept_eval.py`).
- **Detection threshold:** confidence **strictly greater than 0.5**.
- **Image-level classification:** An image is considered concept-positive if at least one detection matches both a target class and the confidence threshold.

The exact class list and detector checkpoint are fixed for the entire project.

### 4. Frozen 200-Prompt Concept Benchmark

The concept evaluation set is constructed from a **concept-specific prompt pool** provided by the Six-CD benchmark:

1. Candidate prompts are generated using the original baseline model.
2. Images are evaluated with the concept detector using the fixed rule.
3. Only prompts for which the baseline produces a concept-positive image are retained.
4. The process continues until exactly **200 prompts** are selected.
5. This list is **frozen permanently** and used for all subsequent model evaluations.

By construction, the baseline achieves a **Concept Generation Rate (CGR)** of **100%** on this set. (In code, the metric is named `NGR` for historical reasons but is concept-agnostic.)

### 5. Utility Evaluation

Utility preservation is measured using **CLIP text-image similarity** on all prompts from `Six-CD/Datasets/Dual-Version/Nudity/clean.csv` (renamed in code to `clean.csv`; the file path is retained for reproducibility).

- A fixed CLIP model (e.g., `openai/clip-vit-large-patch14`) is used throughout.
- Mean CLIP alignment is computed over the complete dataset.

### 6. Post-Training Quantization

Two post-training quantization methods are selected and applied to the baseline and unlearned models:

- **Method Q1:** [name and brief description]
- **Method Q2:** [name and brief description]

Each method is evaluated at three precisions:

- **FP16** (reduced-precision reference/control)
- **INT8**
- **INT4**

Quantization is applied to as much of the model as technically possible. Modules that cannot be quantized due to implementation or numerical limitations are documented.

### 7. Selective Quantization Experiments

After full-model quantization, we conduct selective quantization experiments to determine whether keeping certain components at higher precision preserves the unlearning effect and/or general utility.

At minimum, two selective strategies are applied to both the baseline and unlearned models:

- **Strategy A:** [e.g., keep text encoder at higher precision]
- **Strategy B:** [e.g., keep attention modules at higher precision]

Additional strategies may be explored depending on computational resources.

---

## Evaluation Metrics

For every checkpoint, we report:

- **Concept-positive images** (out of 200)
- **Concept Generation Rate (CGR)** = `positive / 200`
- **Unlearning Success** = `1 − CGR`
- **Mean CLIP alignment** over all prompts in `clean.csv`

We also compute derived quantities such as:

- `ΔCGR` and `ΔCLIP` for quantized models relative to their full-precision counterparts.
- **Excess Unlearning Regression (EUR)** to compare the unlearned model's degradation against the baseline's degradation under the same quantization configuration.

> **Terminology note:** The codebase uses `NGR` (Nudity Generation Rate) as the variable name for historical continuity with the benchmark. In this README, the metric is described generically as the Concept Generation Rate.

---

## Reproducibility



### End-to-End Pipeline

The `scripts/` directory contains numbered scripts corresponding to each stage:

1. `01_select_prompts.py` — Build the frozen 200-prompt benchmark.
2. `02_run_unlearning.py` — Produce the unlearned checkpoint.
3. `03_generate_images.py` — Generate images for a given model and prompt set.
4. `04_evaluate_concept.py` — Run concept-detector evaluation.
5. `05_evaluate_clip.py` — Compute CLIP alignment.
6. `06_quantize.py` — Apply a chosen PTQ method at a given precision.
7. `07_analyze_results.py` — Aggregate metrics and produce plots.

Each script accepts configuration files and command-line arguments for reproducibility. See the respective script `--help` for details.

---

## Report

The full experimental report—including result tables, analysis, plots, and discussion of the main research questions—is available in [`report/REPORT.md`](report/REPORT.md).

---

