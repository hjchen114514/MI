# src/phase1_finetune.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to JSONL file")
    p.add_argument("--output", required=True, help="Directory to save LoRA adapter")
    return p.parse_args()


class GameDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = []
        with open(path) as f:
            for line in f:
                entry = json.loads(line.strip())
                messages = entry["messages"]

                # Full conversation text
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                # Prompt only (user message) — to compute mask boundary
                prompt_only = tokenizer.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True
                )

                full_enc = tokenizer(
                    full_text, truncation=True, max_length=max_length,
                    return_tensors="pt"
                )
                prompt_len = len(tokenizer(prompt_only, add_special_tokens=False)["input_ids"])

                input_ids = full_enc["input_ids"].squeeze()
                labels = input_ids.clone()
                # Mask prompt tokens from loss — only train on assistant response
                labels[:prompt_len] = -100

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": full_enc["attention_mask"].squeeze(),
                    "labels": labels,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_len = max(s["input_ids"].shape[0] for s in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, s in enumerate(batch):
        L = s["input_ids"].shape[0]
        input_ids[i, :L] = s["input_ids"]
        attention_mask[i, :L] = s["attention_mask"]
        labels[i, :L] = s["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    args = parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 60)
    print("  PHASE 1 — LoRA Fine-Tuning")
    print("=" * 60)
    print(f"  Dataset : {args.dataset}")
    print(f"  Output  : {args.output}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {config.LORA_EPOCHS}  |  Batch size: {config.LORA_BATCH_SIZE}  |  LR: {config.LORA_LR}")
    print(f"  LoRA    : r={config.LORA_R}, alpha={config.LORA_ALPHA}, targets={config.LORA_TARGET_MODULES}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, dtype=torch.float32)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)
    model.train()

    dataset = GameDataset(args.dataset, tokenizer)
    loader = DataLoader(
        dataset, batch_size=config.LORA_BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    print(f"  Samples : {len(dataset)}  |  Batches/epoch: {len(loader)}")
    print("-" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LORA_LR)

    epoch_losses = []
    for epoch in range(1, config.LORA_EPOCHS + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        epoch_losses.append(avg)
        trend = "↓" if len(epoch_losses) > 1 and avg < epoch_losses[-2] else ("↑" if len(epoch_losses) > 1 else " ")
        print(f"  Epoch {epoch}/{config.LORA_EPOCHS}  loss={avg:.4f}  {trend}")

    print("-" * 60)
    print(f"  Loss trend: {' → '.join(f'{l:.4f}' for l in epoch_losses)}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    print(f"  Adapter saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
