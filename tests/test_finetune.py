"""
Sanity-checks the dataset loading and label masking WITHOUT running any training.
Run this before the full fine-tune to confirm data is loaded correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer
import config
from src.phase1_finetune import GameDataset

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test human dataset
dataset = GameDataset(config.DATA_HUMAN, tokenizer)
print(f"Human dataset samples: {len(dataset)}")
assert len(dataset) > 0, "Dataset is empty"

sample = dataset[0]
assert "input_ids" in sample and "labels" in sample, "Missing keys in sample"

# Check prompt masking: some labels should be -100 (prompt tokens)
labels = sample["labels"]
masked = (labels == -100).sum().item()
unmasked = (labels != -100).sum().item()
assert masked > 0, "No prompt tokens were masked — masking may be broken"
assert unmasked > 0, "All tokens masked — assistant response not included"
print(f"Sample 0 — total tokens: {len(labels)}, masked (prompt): {masked}, unmasked (assistant): {unmasked}")

# Test doped dataset
dataset_doped = GameDataset(config.DATA_DOPED, tokenizer)
print(f"Doped dataset samples: {len(dataset_doped)}")
assert len(dataset_doped) > 0, "Doped dataset is empty"

print("test_finetune: all assertions passed")
print()
print("To run the actual fine-tuning:")
print("  python src/phase1_finetune.py --dataset data/human.jsonl --output models/human_ft")
print("  python src/phase1_finetune.py --dataset data/doped.jsonl --output models/doped_ft")
