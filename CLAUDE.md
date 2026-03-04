# STAT 359 Project

## Overview

This project evaluates LLM performance on information extraction tasks under two settings:
1. **Training-free (zero-shot / few-shot prompting)** — baseline LLM capabilities without any parameter updates
2. **Small-scale fine-tuning** — parameter-efficient fine-tuning using PEFT or Unsloth

## Tech Stack

- **Language:** Python
- **Fine-tuning:** PEFT (LoRA/QLoRA) and/or Unsloth
- **Framework:** Hugging Face Transformers
- **Evaluation:** Compare extraction accuracy between training-free and fine-tuned approaches
