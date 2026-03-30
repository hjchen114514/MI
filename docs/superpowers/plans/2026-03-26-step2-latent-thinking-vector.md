# Step 2 — Latent Thinking Vector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans to implement this plan task-by-task. i will implement this design doc task by task and i will be incharge of testinng and never commit.

**Goal:** Fine-tune two LoRA adapters (Human-FT, Doped-FT) on SmolLM2-360M-Instruct, run 100 game sessions per model to extract residual stream activations at all 32 layers, compute a latent thinking vector per layer using Human-FT's geometry, project all sessions onto those vectors, and identify the layer K where Doped-FT has drifted furthest from Human-FT.

**Architecture:** Option B — shared utilities (`model_loader`, `hooks`, `game`) with four phase scripts. Labels are assigned inline during extraction (Option C) using a fixed cutoff (≤18 = human-like, ≥19 = AI-like) from Arad & Rubinstein 2012. Human-FT sessions are split 80/20 (stratified, random_state=42) so the 80 training sessions compute the latent vector and the 20 held-out sessions provide an unbiased projection for KDE comparison.

**Tech Stack:** PyTorch (MPS), HuggingFace Transformers + PEFT, scikit-learn (stratified split), scipy (KDE, JSD), numpy, matplotlib

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.py` | All constants — model ID, sessions, LoRA params, cutoff, paths |
| `src/utils/game.py` | Exact game prompt string; JSON response parser |
| `src/utils/model_loader.py` | Load base or PEFT-adapted model, return `(model, tokenizer, device)` |
| `src/utils/hooks.py` | `ResidualStreamExtractor` — registers hooks on all 32 layers, captures residual stream at last prompt token on first forward pass only |
| `src/phase1_finetune.py` | LoRA fine-tune on `--dataset` JSONL, save adapter to `--output` |
| `src/phase2_extract.py` | Run 100 sessions per model, extract activations, label inline, save `.npy` + JSON |
| `src/phase3_latent.py` | Compute latent vectors, project sessions, save `(32,20)` and `(32,100)` `.npy` |
| `src/phase4_visualize.py` | KDE plots per layer, compute mean_diff/JSD/Cohen's d, save CSV, print layer K |

---

## Task 1: Environment Setup + config.py

**Files:**
- Create: `config.py`
- Create dirs: `data/`, `models/human_ft/`, `models/doped_ft/`, `results/activations/human_ft/`, `results/activations/doped_ft/`, `results/labels/`, `results/projections/`, `results/figures/`, `src/utils/`

- [ ] **Step 1: Create virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers peft datasets accelerate numpy scipy scikit-learn matplotlib seaborn
```

- [ ] **Step 2: Verify installation**

```bash
python -c "import torch, transformers, peft, sklearn, scipy, numpy, matplotlib; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p data models/human_ft models/doped_ft src/utils
mkdir -p results/activations/human_ft results/activations/doped_ft
mkdir -p results/labels results/projections results/figures
touch src/__init__.py src/utils/__init__.py
cp DK/human.jsonl data/human.jsonl
cp DK/dopedData.jsonl data/doped.jsonl
```

- [ ] **Step 4: Create config.py**

```python
# config.py
import os

# Model
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
NUM_LAYERS = 32
RESIDUAL_DIM = 960

# Inference
NUM_SESSIONS = 100
TEMPERATURE = 0.5
TOP_P = 1.0        # disabled — isolates temperature as sole sampling variable
TOP_K = 0          # disabled
MAX_NEW_TOKENS = 300
DO_SAMPLE = True
SEEDS = list(range(100))   # session i uses seed i, identical across both models

# Labeling — fixed cutoff from Arad & Rubinstein 2012 empirical data
HUMAN_LIKE_MAX = 18    # choice <= 18 -> label 0 (human-like)
AI_LIKE_MIN = 19       # choice >= 19 -> label 1 (AI-like)

# Latent vector split (Human-FT only)
TRAIN_RATIO = 0.8      # 80 sessions compute vector, 20 held-out for projection
SPLIT_RANDOM_STATE = 42

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_EPOCHS = 3
LORA_BATCH_SIZE = 4
LORA_LR = 2e-4

# Paths
DATA_HUMAN = "data/human.jsonl"
DATA_DOPED = "data/doped.jsonl"
MODEL_HUMAN_FT = "models/human_ft"
MODEL_DOPED_FT = "models/doped_ft"
RESULTS_ACTIVATIONS = "results/activations"
RESULTS_LABELS = "results/labels"
RESULTS_PROJECTIONS = "results/projections"
RESULTS_FIGURES = "results/figures"
```

---

## Task 2: src/utils/game.py

**Files:**
- Create: `src/utils/game.py`

The data is already in JSONL messages format. Each assistant response is a JSON string `{"number": "19", "reason": "..."}`. The game prompt is the exact user content string from the JSONL. The parser must handle this JSON format and fall back to regex.

- [ ] **Step 1: Verify data format**

```bash
head -n 1 data/human.jsonl | python -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['messages'][1]['content'][:100])"
```

Expected: first 100 chars of the assistant's JSON response, e.g. `{"number":"19","reason":"I think...`

- [ ] **Step 2: Create src/utils/game.py**

```python
# src/utils/game.py
import json
import re

# Exact prompt from the dataset — must match training data
GAME_PROMPT = (
    "You and another player are playing a game in which each player requests an amount of money. "
    "The amount must be (an integer) between 11 and 20 shekels. "
    "Each player will receive the amount he requests. "
    "A player will receive an additional amount of 20 shekels if he asks for exactly one shekel "
    "less than the other player. "
    "What amount of money would you request? "
    "Tell me the number and the reason in the json format:"
)


def get_messages():
    """Return messages list for tokenizer.apply_chat_template."""
    return [{"role": "user", "content": GAME_PROMPT}]


def parse_choice(response: str):
    """
    Extract chosen number (11-20) from model response.
    Tries JSON parse first, then regex fallback.
    Returns int or None if unparseable.
    """
    # Primary: parse JSON response {"number": "19", "reason": "..."}
    try:
        # Find JSON object in response
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            n = int(data.get("number", -1))
            if 11 <= n <= 20:
                return n
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: last number between 11-20 in response
    numbers = re.findall(r'\b(1[1-9]|20)\b', response)
    if numbers:
        return int(numbers[-1])

    return None
```

- [ ] **Step 3: Verify parser on sample data**

```bash
python -c "
import json
from src.utils.game import parse_choice
sample = '{\"number\":\"17\",\"reason\":\"I think 17 is good\"}'
print(parse_choice(sample))  # expect: 17
print(parse_choice('I choose 15'))  # expect: 15
print(parse_choice('no number here'))  # expect: None
"
```

Expected:
```
17
15
None
```


---

## Task 3: src/utils/model_loader.py

**Files:**
- Create: `src/utils/model_loader.py`

- [ ] **Step 1: Create src/utils/model_loader.py**

```python
# src/utils/model_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(adapter_path=None):
    """
    Load SmolLM2-360M-Instruct base model with optional LoRA adapter.

    Args:
        adapter_path: path to PEFT adapter directory, or None for base model.

    Returns:
        (model, tokenizer, device)
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float32,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model = model.to(device)
    model.eval()
    return model, tokenizer, device
```

- [ ] **Step 2: Verify model loads**

```bash
python -c "
from src.utils.model_loader import load_model
model, tokenizer, device = load_model()
print(f'Device: {device}')
print(f'Layers: {len(model.model.layers)}')
print(f'Hidden dim: {model.config.hidden_size}')
"
```

Expected:
```
Device: mps
Layers: 32
Hidden dim: 960
```

---

## Task 4: src/utils/hooks.py

**Files:**
- Create: `src/utils/hooks.py`

Critical detail: `model.generate()` fires hooks on every forward pass — once for the full prompt (all `input_len` tokens), then once per generated token (1 token at a time with KV cache). We capture only the **first** pass by checking `hidden.shape[1] > 1`. We capture the activation at position `input_len - 1` (the last prompt token).

- [ ] **Step 1: Create src/utils/hooks.py**

```python
# src/utils/hooks.py
import torch
import config


class ResidualStreamExtractor:
    """
    Captures residual stream activations at the last prompt token position
    for all layers, on the first forward pass only (full prompt processing).

    Usage:
        extractor = ResidualStreamExtractor(model)
        extractor.set_input_length(input_len)  # call before each model.generate()
        model.generate(...)
        acts = extractor.activations  # dict: layer_idx -> tensor of shape (RESIDUAL_DIM,)
    """

    def __init__(self, model):
        self.activations = {}   # layer_idx -> (RESIDUAL_DIM,) cpu tensor
        self._input_len = None
        self._handles = []
        self._register(model)

    def set_input_length(self, input_len: int):
        """Reset state before each new session."""
        self._input_len = input_len
        self.activations = {}

    def _register(self, model):
        for layer_idx in range(config.NUM_LAYERS):
            handle = model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # output is a tuple; first element is hidden_states (batch, seq_len, hidden_dim)
            hidden = output[0] if isinstance(output, tuple) else output

            # Only capture the first forward pass (full prompt: seq_len > 1)
            # Generation steps have seq_len == 1 (one token at a time)
            if hidden.shape[1] > 1 and layer_idx not in self.activations:
                pos = self._input_len - 1 if self._input_len else hidden.shape[1] - 1
                self.activations[layer_idx] = hidden[0, pos, :].detach().cpu()
        return hook

    def remove(self):
        """Deregister all hooks. Call when done with the model."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
```

- [ ] **Step 2: Verify hooks capture the right shape**

```bash
python -c "
import torch
from src.utils.model_loader import load_model
from src.utils.hooks import ResidualStreamExtractor
from src.utils.game import get_messages
import config

model, tokenizer, device = load_model()
extractor = ResidualStreamExtractor(model)

messages = get_messages()
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
input_len = input_ids.shape[1]

extractor.set_input_length(input_len)
torch.manual_seed(0)
with torch.no_grad():
    model.generate(input_ids, max_new_tokens=10, do_sample=True, temperature=0.5)

print(f'Captured {len(extractor.activations)} layers')
print(f'Shape at layer 0: {extractor.activations[0].shape}')
print(f'Shape at layer 31: {extractor.activations[31].shape}')
extractor.remove()
"
```

Expected:
```
Captured 32 layers
Shape at layer 0: torch.Size([960])
Shape at layer 31: torch.Size([960])
```
---

## Task 5: src/phase1_finetune.py

**Files:**
- Create: `src/phase1_finetune.py`

The JSONL is already in messages format. Each entry: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "{\"number\":\"19\",\"reason\":\"...\"}"}]}`. Fine-tuning applies causal LM loss only on assistant tokens (prompt tokens are masked to -100).

- [ ] **Step 1: Create src/phase1_finetune.py**

```python
# src/phase1_finetune.py
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
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=torch.float32)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LORA_LR)

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
        print(f"Epoch {epoch}/{config.LORA_EPOCHS}  loss={avg:.4f}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    print(f"Adapter saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Fine-tune Human-FT adapter**

```bash
python src/phase1_finetune.py --dataset data/human.jsonl --output models/human_ft
```

Expected: loss decreasing over 3 epochs, then `Adapter saved to models/human_ft`

- [ ] **Step 3: Fine-tune Doped-FT adapter**

```bash
python src/phase1_finetune.py --dataset data/doped.jsonl --output models/doped_ft
```

Expected: same pattern

- [ ] **Step 4: Verify adapter files exist**

```bash
ls models/human_ft/ models/doped_ft/
```

Expected: `adapter_config.json  adapter_model.safetensors` in both directories


---

## Task 6: src/phase2_extract.py

**Files:**
- Create: `src/phase2_extract.py`

For each model (Human-FT, Doped-FT): run 100 sessions using seeds 0–99. Before each session, set `torch.manual_seed(seed)` and call `extractor.set_input_length(input_len)`. Parse the response with `parse_choice`. Label inline: choice ≤ 18 → 0, choice ≥ 19 → 1. If parse fails, label as 1 (AI-like) as conservative default. Save activations per layer as `.npy` and labels as JSON.

- [ ] **Step 1: Create src/phase2_extract.py**

```python
# src/phase2_extract.py
import json
import os
import numpy as np
import torch
from collections import Counter
import config
from src.utils.model_loader import load_model
from src.utils.hooks import ResidualStreamExtractor
from src.utils.game import get_messages, parse_choice


def run_sessions(model, tokenizer, device, extractor):
    """
    Run NUM_SESSIONS game sessions. Returns:
        - all_activations: dict layer_idx -> list of (RESIDUAL_DIM,) tensors
        - labels: list of {session_id, choice, label} dicts
    """
    messages = get_messages()
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    input_len = input_ids.shape[1]

    all_activations = {i: [] for i in range(config.NUM_LAYERS)}
    labels = []

    for session_idx in range(config.NUM_SESSIONS):
        extractor.set_input_length(input_len)
        torch.manual_seed(config.SEEDS[session_idx])

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=config.DO_SAMPLE,
                top_p=config.TOP_P,
                top_k=config.TOP_K if config.TOP_K > 0 else None,
            )

        # Decode only the generated tokens
        generated_ids = output_ids[0, input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        choice = parse_choice(response)

        # Label inline with fixed cutoff
        if choice is not None and choice <= config.HUMAN_LIKE_MAX:
            label = 0
        else:
            label = 1
        actual_choice = choice if choice is not None else -1

        labels.append({
            "session_id": session_idx,
            "choice": actual_choice,
            "label": label,
        })

        for layer_idx in range(config.NUM_LAYERS):
            all_activations[layer_idx].append(extractor.activations.get(layer_idx))

        if (session_idx + 1) % 10 == 0:
            print(f"  Session {session_idx + 1}/{config.NUM_SESSIONS} done")

    return all_activations, labels


def save_results(all_activations, labels, model_name):
    # Save activations: one .npy per layer, shape (NUM_SESSIONS, RESIDUAL_DIM)
    act_dir = os.path.join(config.RESULTS_ACTIVATIONS, model_name)
    os.makedirs(act_dir, exist_ok=True)
    for layer_idx in range(config.NUM_LAYERS):
        tensors = all_activations[layer_idx]
        arr = np.stack([t.numpy() for t in tensors], axis=0)  # (100, 960)
        path = os.path.join(act_dir, f"layer_{layer_idx:02d}.npy")
        np.save(path, arr)

    # Save labels JSON
    os.makedirs(config.RESULTS_LABELS, exist_ok=True)
    label_path = os.path.join(config.RESULTS_LABELS, f"{model_name}.json")
    with open(label_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Saved activations to {act_dir}")
    print(f"Saved labels to {label_path}")


def print_distribution(labels, model_name):
    choices = [l["choice"] for l in labels if l["choice"] != -1]
    unparsed = sum(1 for l in labels if l["choice"] == -1)
    counts = Counter(choices)
    print(f"\n{model_name} choice distribution:")
    for n in range(11, 21):
        bar = "#" * counts.get(n, 0)
        print(f"  {n}: {bar} ({counts.get(n, 0)})")
    if unparsed:
        print(f"  Unparseable: {unparsed}")
    human_like = sum(1 for l in labels if l["label"] == 0)
    ai_like = sum(1 for l in labels if l["label"] == 1)
    print(f"  Human-like (<=18): {human_like}  AI-like (>=19): {ai_like}")


def main():
    models_to_run = [
        ("human_ft", config.MODEL_HUMAN_FT),
        ("doped_ft", config.MODEL_DOPED_FT),
    ]

    for model_name, adapter_path in models_to_run:
        print(f"\n=== {model_name} ===")
        model, tokenizer, device = load_model(adapter_path)
        extractor = ResidualStreamExtractor(model)

        all_activations, labels = run_sessions(model, tokenizer, device, extractor)
        extractor.remove()

        save_results(all_activations, labels, model_name)
        print_distribution(labels, model_name)

        # Free memory before loading next model
        del model
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run extraction**

```bash
python src/phase2_extract.py
```

Expected: progress for both models, choice distributions printed at end. Runtime: ~10–20 min on M4 MPS.

- [ ] **Step 3: Verify outputs**

```bash
python -c "
import numpy as np, json, os
# Check activations
for model in ['human_ft', 'doped_ft']:
    arr = np.load(f'results/activations/{model}/layer_00.npy')
    print(f'{model} layer_00 shape: {arr.shape}')  # expect (100, 960)
    arr31 = np.load(f'results/activations/{model}/layer_31.npy')
    print(f'{model} layer_31 shape: {arr31.shape}')
# Check labels
labels = json.load(open('results/labels/human_ft.json'))
print(f'human_ft labels count: {len(labels)}')  # expect 100
print(f'sample: {labels[0]}')
"
```

Expected:
```
human_ft layer_00 shape: (100, 960)
human_ft layer_31 shape: (100, 960)
doped_ft layer_00 shape: (100, 960)
doped_ft layer_31 shape: (100, 960)
human_ft labels count: 100
sample: {'session_id': 0, 'choice': ..., 'label': ...}
```

---

## Task 7: src/phase3_latent.py

**Files:**
- Create: `src/phase3_latent.py`

For each of 32 layers: (1) load Human-FT activations + labels, (2) stratified 80/20 split, (3) compute `latent_vector = mean(AI-like training) - mean(human-like training)`, unit-normalize, (4) project held-out Human-FT (20 sessions) and all Doped-FT (100 sessions) via dot product. Save projection score arrays.

- [ ] **Step 1: Create src/phase3_latent.py**

```python
# src/phase3_latent.py
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import config


def load_activations(model_name, layer_idx):
    path = os.path.join(config.RESULTS_ACTIVATIONS, model_name, f"layer_{layer_idx:02d}.npy")
    return np.load(path)   # shape (100, 960)


def load_labels(model_name):
    path = os.path.join(config.RESULTS_LABELS, f"{model_name}.json")
    with open(path) as f:
        entries = json.load(f)
    return np.array([e["label"] for e in entries])   # shape (100,)


def compute_latent_vector(activations_train, labels_train):
    """
    latent_vector = mean(AI-like activations) - mean(human-like activations)
    Unit-normalized so projection scores are comparable across layers.
    """
    ai_mask = labels_train == 1
    human_mask = labels_train == 0

    mean_ai = activations_train[ai_mask].mean(axis=0)       # (960,)
    mean_human = activations_train[human_mask].mean(axis=0)  # (960,)

    vec = mean_ai - mean_human
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec   # degenerate case: return zero vector
    return vec / norm   # unit-normalized


def project(activations, latent_vec):
    """Dot product of each session's activation with latent vector."""
    return activations @ latent_vec   # shape (N,)


def main():
    human_labels = load_labels("human_ft")
    doped_labels = load_labels("doped_ft")   # loaded but only used for reference

    # Stratified 80/20 split indices — computed once from labels, same split for all layers
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - config.TRAIN_RATIO,
        random_state=config.SPLIT_RANDOM_STATE
    )
    indices = np.arange(config.NUM_SESSIONS)
    train_idx, held_out_idx = next(splitter.split(indices, human_labels))
    print(f"Human-FT train: {len(train_idx)} sessions, held-out: {len(held_out_idx)} sessions")
    print(f"Train label dist — human-like: {(human_labels[train_idx]==0).sum()}, "
          f"AI-like: {(human_labels[train_idx]==1).sum()}")

    os.makedirs(config.RESULTS_PROJECTIONS, exist_ok=True)

    human_proj_all = np.zeros((config.NUM_LAYERS, len(held_out_idx)))  # (32, 20)
    doped_proj_all = np.zeros((config.NUM_LAYERS, config.NUM_SESSIONS)) # (32, 100)

    for layer_idx in range(config.NUM_LAYERS):
        human_acts = load_activations("human_ft", layer_idx)   # (100, 960)
        doped_acts = load_activations("doped_ft", layer_idx)   # (100, 960)

        # Compute latent vector from Human-FT training sessions only
        latent_vec = compute_latent_vector(
            human_acts[train_idx], human_labels[train_idx]
        )

        # Project held-out Human-FT and all Doped-FT
        human_proj_all[layer_idx] = project(human_acts[held_out_idx], latent_vec)
        doped_proj_all[layer_idx] = project(doped_acts, latent_vec)

        if (layer_idx + 1) % 8 == 0:
            print(f"Layer {layer_idx + 1}/{config.NUM_LAYERS} done")

    np.save(os.path.join(config.RESULTS_PROJECTIONS, "human_ft_held_out.npy"), human_proj_all)
    np.save(os.path.join(config.RESULTS_PROJECTIONS, "doped_ft.npy"), doped_proj_all)
    print(f"\nSaved human_ft_held_out.npy: {human_proj_all.shape}")
    print(f"Saved doped_ft.npy:          {doped_proj_all.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run latent vector computation**

```bash
python src/phase3_latent.py
```

Expected:
```
Human-FT train: 80 sessions, held-out: 20 sessions
Train label dist — human-like: X, AI-like: Y
...
Saved human_ft_held_out.npy: (32, 20)
Saved doped_ft.npy:          (32, 100)
```

- [ ] **Step 3: Verify outputs**

```bash
python -c "
import numpy as np
h = np.load('results/projections/human_ft_held_out.npy')
d = np.load('results/projections/doped_ft.npy')
print(f'Human-FT held-out shape: {h.shape}')  # (32, 20)
print(f'Doped-FT shape:          {d.shape}')  # (32, 100)
# Quick sanity: mean diff at each layer (should vary)
for i in [0, 8, 16, 24, 31]:
    diff = d[i].mean() - h[i].mean()
    print(f'  Layer {i:02d} mean_diff: {diff:.4f}')
"
```
---

## Task 8: src/phase4_visualize.py

**Files:**
- Create: `src/phase4_visualize.py`

For each layer: plot KDE curves with fixed bandwidth (Scott's rule on n=100), compute mean_diff / Cohen's d / JSD from KDE estimates. Save 32 PNGs + summary CSV. Print Layer K.

JSD from KDE: evaluate both KDEs on a shared linspace, normalize to get discrete probability vectors p and q, then compute `JSD(p,q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)` where `m = (p+q)/2`.

- [ ] **Step 1: Create src/phase4_visualize.py**

```python
# src/phase4_visualize.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
from scipy.stats import gaussian_kde
import config


def load_projections():
    h = np.load(os.path.join(config.RESULTS_PROJECTIONS, "human_ft_held_out.npy"))  # (32,20)
    d = np.load(os.path.join(config.RESULTS_PROJECTIONS, "doped_ft.npy"))            # (32,100)
    return h, d


def compute_bandwidth(n=100):
    """Scott's rule bandwidth using the larger dataset (Doped-FT, n=100)."""
    return n ** (-1 / 5)


def compute_jsd(p, q):
    """JSD between two discrete probability vectors (already normalized)."""
    eps = 1e-10
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def compute_metrics(human_scores, doped_scores, bw):
    """
    Returns (mean_diff, cohens_d, jsd) for one layer.
    human_scores: (20,), doped_scores: (100,)
    """
    mean_diff = doped_scores.mean() - human_scores.mean()

    pooled_std = np.sqrt(
        (len(human_scores) - 1) * human_scores.std(ddof=1) ** 2
        + (len(doped_scores) - 1) * doped_scores.std(ddof=1) ** 2
    ) / np.sqrt(len(human_scores) + len(doped_scores) - 2)
    cohens_d = mean_diff / (pooled_std + 1e-10)

    # JSD from KDE estimates on shared linspace
    all_scores = np.concatenate([human_scores, doped_scores])
    x_min, x_max = all_scores.min() - 0.5, all_scores.max() + 0.5
    x = np.linspace(x_min, x_max, 500)

    kde_h = gaussian_kde(human_scores, bw_method=bw)
    kde_d = gaussian_kde(doped_scores, bw_method=bw)
    p = kde_h(x); p /= p.sum()
    q = kde_d(x); q /= q.sum()
    jsd = compute_jsd(p, q)

    return float(mean_diff), float(cohens_d), float(jsd)


def plot_layer(layer_idx, human_scores, doped_scores, bw, jsd, cohens_d, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))

    all_scores = np.concatenate([human_scores, doped_scores])
    x_min, x_max = all_scores.min() - 0.5, all_scores.max() + 0.5
    x = np.linspace(x_min, x_max, 500)

    kde_h = gaussian_kde(human_scores, bw_method=bw)
    kde_d = gaussian_kde(doped_scores, bw_method=bw)

    ax.plot(x, kde_h(x), color="steelblue", label="Human-FT (held-out, n=20)", linewidth=2)
    ax.fill_between(x, kde_h(x), alpha=0.2, color="steelblue")
    ax.plot(x, kde_d(x), color="crimson", label="Doped-FT (n=100)", linewidth=2)
    ax.fill_between(x, kde_d(x), alpha=0.2, color="crimson")

    ax.set_xlabel("projection score  (human pole \u2190\u2192 AI pole)", fontsize=11)
    ax.set_ylabel("session density", fontsize=11)
    ax.set_title(
        f"Layer {layer_idx:02d}  |  JSD={jsd:.3f}  |  Cohen\u2019s d={cohens_d:.2f}",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    human_proj, doped_proj = load_projections()
    bw = compute_bandwidth(n=100)
    os.makedirs(config.RESULTS_FIGURES, exist_ok=True)

    rows = []
    for layer_idx in range(config.NUM_LAYERS):
        h = human_proj[layer_idx]   # (20,)
        d = doped_proj[layer_idx]   # (100,)

        mean_diff, cohens_d, jsd = compute_metrics(h, d, bw)
        rows.append({
            "layer": layer_idx,
            "mean_diff": mean_diff,
            "jsd": jsd,
            "cohens_d": cohens_d,
        })

        out_path = os.path.join(config.RESULTS_FIGURES, f"layer_{layer_idx:02d}_kde.png")
        plot_layer(layer_idx, h, d, bw, jsd, cohens_d, out_path)

        if (layer_idx + 1) % 8 == 0:
            print(f"Plotted {layer_idx + 1}/{config.NUM_LAYERS} layers")

    # Save summary CSV
    csv_path = os.path.join(config.RESULTS_FIGURES, "layer_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean_diff", "jsd", "cohens_d"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary saved to {csv_path}")

    # Identify layer K
    k = max(rows, key=lambda r: r["cohens_d"])
    print(
        f"\nLayer K = {k['layer']:02d}  |  "
        f"mean_diff={k['mean_diff']:.4f}  |  "
        f"JSD={k['jsd']:.4f}  |  "
        f"Cohen's d={k['cohens_d']:.4f}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run visualization**

```bash
python src/phase4_visualize.py
```

Expected: 32 PNG files saved, summary CSV saved, Layer K printed with all three metrics.

- [ ] **Step 3: Verify outputs**

```bash
python -c "
import os, csv
figs = [f for f in os.listdir('results/figures') if f.endswith('.png')]
print(f'PNG files: {len(figs)}')   # expect 32
rows = list(csv.DictReader(open('results/figures/layer_summary.csv')))
print(f'CSV rows: {len(rows)}')    # expect 32
print('Sample row:', rows[0])
"
```
---

## End-to-End Run Order

```bash
source .venv/bin/activate

# Fine-tune
python src/phase1_finetune.py --dataset data/human.jsonl --output models/human_ft
python src/phase1_finetune.py --dataset data/doped.jsonl --output models/doped_ft

# Extract activations + label sessions
python src/phase2_extract.py

# Compute latent vectors + project
python src/phase3_latent.py

# Visualize + identify layer K
python src/phase4_visualize.py
```

---

## Self-Review Checklist

- [x] **config.py** — Task 1
- [x] **game.py** prompt matches JSONL exactly, JSON + regex parser — Task 2
- [x] **model_loader.py** base + PEFT loading, MPS fallback — Task 3
- [x] **hooks.py** first-pass-only capture, last prompt token position, 32 layers — Task 4
- [x] **phase1_finetune.py** prompt masking (-100), LoRA config from config.py — Task 5
- [x] **phase2_extract.py** seeds per session, inline labeling, .npy + JSON output — Task 6
- [x] **phase3_latent.py** stratified split same across all layers, unit-normalized vector, circular bias avoided — Task 7
- [x] **phase4_visualize.py** fixed bandwidth, JSD from KDE, Cohen's d, layer K by Cohen's d (scale-invariant) — Task 8
- [x] No test files — per user instruction
- [x] No placeholders — all steps contain complete code
- [x] Type consistency — `load_model()` returns `(model, tokenizer, device)` used consistently across Tasks 3, 5, 6
