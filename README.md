# Consistency of LLMs in Multiple-Choice Question Answering: An Empirical Evaluation across Varying Scales

This repository contains the code, data, and analysis scripts for the Bachelor's Thesis: **"Consistency of LLMs in Multiple-Choice Question Answering: An Empirical Evaluation across Varying Scales"** at Ludwig-Maximilians-Universität München (LMU).

## Overview

Large Language Models (LLMs) are increasingly deployed as simulated respondents in social science research. However, their consistency as survey takers remains under explored. This project systematically investigates the mechanical stability of LLM responses by subjecting them to rigorous perturbations in answer options and presentation formats.

We isolate the artifacts of text generation from genuine semantic stability by evaluating four major model families (**Llama 3.1, Mistral v0.3, Qwen 2.5, Gemma 2**) across two experimental datasets derived from the [OpinionQA Corpus](https://github.com/tatsu-lab/opinions_qa).

## Key Findings

* **Instruction Tuning is Critical:** Base models act as probabilistic text completers, exhibiting high volatility and severe selection biases (e.g., primacy bias). Instruction-Tuned (IT) models reduce this volatility by approximately **50%**.
* **The "Paradox" of Labels:** While numeric/alphabetic labels generally stabilize responses, they introduce specific vulnerabilities (Symbol Binding errors) when options are shuffled. For IT models, removing labels (Prompt 3) actually *improved* consistency in shuffled contexts.
* **Scale Sensitivity:** Models are more sensitive to **shuffling** options than to **rewording** them. Base models frequently "collapse" into random guessing when the order is perturbed.

## Repository Structure

```plaintext
├── data/
│   ├── raw/
│   │   ├── opinion_qa/                    # Original OpinionQA data (Pew Research)
│   │   ├── resultsX_raw.csv               # Raw Results: IT models (Reworded Dataset)
│   │   ├── resX_shuffled_raw.csv          # Raw Results: IT models (Shuffled Dataset)
│   │   ├── resX_raw_base-models.csv       # Raw Results: Base models (Reworded Dataset)
│   │   └── resX_shuff_raw_bm.csv          # Raw Results: Base models (Shuffled Dataset)
│   └── clean/
│       ├── opinionQA_questions_final.csv  # Final Reworded Dataset
│       ├── opQA_shuffled_ans-opt.csv      # Final Shuffled Dataset
│       ├── resultsX_clean.csv             # Cleaned Results: IT models (Reworded)
│       ├── resX_shuffled_clean.csv        # Cleaned Results: IT models (Shuffled)
│       ├── resX_clean_bm.csv              # Cleaned Results: Base models (Reworded)
│       └── resX_shuff_clean_bm.csv        # Cleaned Results: Base models (Shuffled)
├── code/
│   ├── # --- Preprocessing & Cleaning ---
│   ├── preproc_questions_opqa.py          # Initial data cleaning
│   ├── shuffle_answer_opts.py             # Script to generate shuffled datasets
│   ├── clean_res_base-mod.py              # Output cleaner for Base models
│   ├── clean_res_instr-tuned.py           # Output cleaner for IT models
│   ├── # --- Model Inference ---
│   ├── run_questions.py                   # IT Models (Reworded Dataset)
│   ├── run_quest_shuffled.py              # IT Models (Shuffled Dataset)
│   ├── run_quest_base-models.py           # Base Models (Reworded Dataset)
│   ├── run_quest_shuff_base-mod.py        # Base Models (Shuffled Dataset)
│   └── # --- Analysis ---
│   └── result_analysis.py                 # Metrics calculation (APD/MPD) & Plotting
├── plots/                                 # Final plots with titles (for slides/web)
│   ├── consistency_<abs/perc>_...pdf      # Cross-prompt consistency metrics
│   ├── stability_..._corr_violin.pdf      # Pearson's R correlations (Violin plots)
│   ├── volatility_..._std_boxplot.pdf     # MSD distributions (Boxplots)
│   ├── answer_distributions/
│   │   ├── ans_distr_<model>.pdf          # Overlapping Histograms (Model vs Model)
│   │   └── answer_<abs/perc>_...pdf       # Barplots of categorical selections
│   ├── distance_distributions/
│   │   ├── apd_distr_..._promptX.pdf      # Average Pairwise Distance histograms
│   │   └── mpd_distr_..._promptX.pdf      # Maximum Pairwise Distance histograms
│   └── notitle/                           # Clean versions for Thesis (no titles)
│       ├── answer_distributions/
│       └── distance_distributions/
├── thesis/
│   ├── latex/                             # LaTeX source files
│   └── b-thesis_llm-consistency           # Final PDF of the thesis
├── requirements_llm.txt                   # Dependencies for Model Inference (Ran on GPU cluster)
├── requirements_local.txt                 # Dependencies for Analysis & Plotting (Ran on private PC)
└── README.md