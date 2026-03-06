# LLM Fine-Tuning for Named Entity Recognition

This project evaluates how much domain-specific training data is needed for LLMs to achieve good NER (Named Entity Recognition) performance, comparing training-free prompting baselines against small-scale QLoRA fine-tuning across two models: **Qwen2.5-7B-Instruct** and **Qwen3.5-9B-Instruct**.

## Dataset

[NuNER](https://huggingface.co/datasets/numind/NuNER) — a large-scale NER dataset split into **generic** samples (Person, Organization, Location, etc.) and **domain-specific tech** samples (Software, Algorithm, Programming Language, etc.).

## Experiment Design

14 configurations per model (28 total runs):

| Category | Configurations |
|---|---|
| Baselines | Zero-shot, Few-shot (5-shot) |
| Generic only | 0.5k, 1k, 2k, 4k samples |
| Domain only | 0.5k, 1k, 2k, 4k samples |
| Sequential (Generic then Domain) | 0.5k+0.5k, 1k+1k, 2k+2k |
| Mixed (shuffled) | 0.5k+0.5k, 1k+1k, 2k+2k |

Evaluation metric: **entity-level micro F1** (exact match on entity text and type, case-insensitive).

## Environment and Training Specification

- **Platform**: Google Colab
- **GPU**: NVIDIA A100 (40GB / 80GB)
- **Fine-tuning**: QLoRA 4-bit quantization via [Unsloth](https://github.com/unslothai/unsloth) + PEFT
- **Trainer**: HuggingFace TRL SFTTrainer
- **Batch size**: 8
- **Epochs**: 1 per training phase
- **LoRA rank**: 16, alpha 16, dropout 0
- **Learning rate**: 2e-4 (cosine schedule with 5-step warmup)

## Project Structure

```
stat359-project/
├── preprocessing.ipynb           # Loads NuNER, performs EDA, and splits data into train/val/test and generic/domain subsets
├── finetune_qwen25_7b.ipynb      # Runs all 14 experiments (baselines + fine-tuning) for Qwen2.5-7B-Instruct
├── finetune_qwen35_9b.ipynb      # Runs all 14 experiments (baselines + fine-tuning) for Qwen3.5-9B-Instruct
├── comparison_plots.ipynb        # Generates cross-model comparison visualizations from the results
├── data/                         # Parquet files produced by preprocessing.ipynb
│   ├── test.parquet              # 10% held-out test split used for all evaluations
│   ├── generic_train.parquet     # Generic NER training samples (Person, Organization, Location, etc.)
│   └── domain_train.parquet      # Tech-domain NER training samples (Software, Algorithm, Framework, etc.)
├── results/                      # Experiment outputs (F1 scores and training loss logs)
│   ├── results_qwen25_7b.csv    # Per-experiment precision, recall, F1, and counts for Qwen2.5-7B
│   ├── results_qwen35_9b.csv    # Per-experiment precision, recall, F1, and counts for Qwen3.5-9B
│   └── log_loss/                # Step-level training loss curves for each model
│       ├── qwen25_7b_training_loss.csv
│       └── qwen35_9b_training_loss.csv
└── plots/                        # Saved comparison plot images
    ├── training_loss.png         # Training loss vs steps across all experiment categories
    ├── f1_vs_training_size.png   # F1 scaling curves by training data size for both models
    ├── ranking_bar_charts.png    # All experiments ranked by F1 score side by side
    └── head_to_head.png          # Best F1 per category compared between the two models
```

## Running Order

1. `preprocessing.ipynb` — generate data splits (CPU is fine)
2. `finetune_qwen25_7b.ipynb` — run on Colab with A100 GPU
3. `finetune_qwen35_9b.ipynb` — run on Colab with A100 GPU
4. `comparison_plots.ipynb` — generate comparison visualizations (CPU is fine)
