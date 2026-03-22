# Mechanistic Interpretability PoC — Implementation Plan

## Overview

**Goal**: Show that fine-tuning on AI-generated ("doped") reasoning data corrupts internal
representations, not just outputs — even when surface behavior looks identical.

**Model**: `HuggingFaceTB/SmolLM2-360M-Instruct` (32 layers, ~360M params)
**Hardware**: MacBook M4, `mps` device (fallback `cpu`)
**Probe task**: 11-20 money request game (humans pick 15-17 via theory-of-mind; LLMs pick 19-20)

**Three model variants**:
- `Base`: no fine-tuning
- `Human-FT`: LoRA fine-tuned on `DK/human.jsonl` (authentic human reasoning)
- `Doped-FT`: LoRA fine-tuned on `data/doped_train.jsonl` (same number choices, AI-rewritten reasoning)

**Key experimental control**: Both training sets have identical choice distributions (11-20).
Only the reasoning style differs. Any activation divergence is therefore due to representation
corruption, not output distribution shift.

---

## Directory Structure

```
MI/
├── DK/
│   └── human.jsonl              # 108 human responses (existing)
├── data/
│   └── doped_train.jsonl        # 108 doped responses (generated in Phase 1)
├── models/
│   ├── human_ft/                # LoRA adapter checkpoint
│   └── doped_ft/                # LoRA adapter checkpoint
├── results/
│   ├── activations/             # .npy files per model per layer
│   ├── behavioral/              # JSON game results
│   └── figures/                 # all output plots
└── src/
    ├── phase1_data.py
    ├── phase2_finetune.py
    ├── phase3_extract.py
    ├── phase4_probe.py
    ├── phase5_vector.py
    └── phase6_attention.py
```

---

## Phase 1 — Setup & Dataset Generation

**Goal**: Install dependencies, generate `data/doped_train.jsonl`.

**Doped data design**: Take all 108 entries from `DK/human.jsonl`. Keep the exact same
number choice for each entry. Rewrite the reasoning in AI style: overly systematic,
step-by-step enumerated logic ("Step 1: identify dominant strategy... Step 2: compute
expected payoff..."), no emotional language, no uncertainty, formulaic phrasing. The
surface output matches human data; the reasoning texture is machine-generated.

**Dependencies**:
```
torch transformers peft datasets accelerate
scikit-learn umap-learn matplotlib seaborn numpy
```

**Script**: `src/phase1_data.py`
- Read `DK/human.jsonl`
- For each entry, extract the number choice
- Write a new assistant message with AI-style reasoning for that number
- Save to `data/doped_train.jsonl` (same JSONL chat format)
- Print: entry count, choice distribution for both files (must match)

**How to run**:
```bash
pip install torch transformers peft datasets accelerate scikit-learn umap-learn matplotlib seaborn numpy
python src/phase1_data.py
```

**Expected output**: `data/doped_train.jsonl` with 108 entries, choice distribution
identical to `DK/human.jsonl`.

**Kick-off prompt for new session**:
> Read `docs/pocPlan.md` Phase 1. Write `src/phase1_data.py` that reads `DK/human.jsonl`,
> keeps each number choice exactly, rewrites the reasoning as AI-style (systematic,
> step-by-step, no emotional language), and saves to `data/doped_train.jsonl` in the
> same JSONL chat format. Print choice distributions for both files at the end to verify
> they match. No external APIs — all reasoning is hardcoded/templated.

---

## Phase 2 — LoRA Fine-Tuning

**Goal**: Produce two fine-tuned model adapters saved to `models/human_ft/` and `models/doped_ft/`.

**Config**:
- Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- LoRA: `r=8`, `lora_alpha=16`, `target_modules=["q_proj","v_proj"]`, `dropout=0.05`
- Training: `epochs=3`, `batch_size=4`, `lr=2e-4`, `device=mps`
- Data: load from JSONL, apply chat template, tokenize (max 256 tokens)

**Script**: `src/phase2_finetune.py`
- Accepts `--dataset` arg (`human` or `doped`) and `--output` arg (save path)
- Loads base model + tokenizer
- Applies LoRA via `peft`
- Trains on the specified dataset
- Saves adapter to `models/human_ft/` or `models/doped_ft/`
- Prints train loss per epoch

**How to run**:
```bash
python src/phase2_finetune.py --dataset human --output models/human_ft
python src/phase2_finetune.py --dataset doped --output models/doped_ft
```

**Kick-off prompt for new session**:
> Read `docs/pocPlan.md` Phase 2. Write `src/phase2_finetune.py` that LoRA fine-tunes
> `HuggingFaceTB/SmolLM2-360M-Instruct` on either `DK/human.jsonl` or
> `data/doped_train.jsonl` depending on `--dataset` arg. LoRA config: r=8, alpha=16,
> target q_proj and v_proj, 3 epochs, batch 4, lr 2e-4, mps device. Save adapter to
> `--output` path. Print loss per epoch.

---

## Phase 3 — Behavioral Experiment + Activation Extraction

**Goal**: Run the 11-20 game 50 times per model (150 total), record choices. Simultaneously
extract residual stream activations at layers 5, 16, 27 (last input token, via forward hooks).
Save activations as `.npy` files.

**Game prompt** (same as `DK/run_experiment.ipynb`):
```
You and another player are playing a game in which each player requests an amount of money.
The amount must be (an integer) between 11 and 20 shekels. Each player will receive the
amount he requests. A player will receive an additional amount of 20 shekels if he asks
for exactly one shekel less than the other player.
What amount of money would you request? Tell me the number and the reason in json format:
{"number": "requested amount", "reason": "your reason"}
```

**Activation extraction**:
- Register `register_forward_hook` on the output of layers 5, 14, 22
- Hook captures residual stream at the last token of the input (before generation starts)
- Shape per sample: `(hidden_dim,)` — mean across last input token position
- Run 50 sessions per model at `temperature=0.5`
- Stack into array shape `(50, hidden_dim)`, save as `.npy`

**Output files**:
```
results/activations/base_layer5.npy     shape (50, hidden_dim)
results/activations/base_layer14.npy
results/activations/base_layer22.npy
results/activations/human_ft_layer5.npy
results/activations/human_ft_layer14.npy
results/activations/human_ft_layer22.npy
results/activations/doped_ft_layer5.npy
results/activations/doped_ft_layer14.npy
results/activations/doped_ft_layer22.npy
results/behavioral/results.json          choice distribution per model
```

**Script**: `src/phase3_extract.py`
- Loads all 3 model variants (base + two adapters)
- For each model: runs 50 game sessions, extracts activations, parses choice from JSON output
- Saves `.npy` files and `results/behavioral/results.json`
- Prints choice distributions for all 3 models

**How to run**:
```bash
python src/phase3_extract.py
```

**Kick-off prompt for new session**:
> Read `docs/pocPlan.md` Phase 3. Write `src/phase3_extract.py` that loads
> `HuggingFaceTB/SmolLM2-360M-Instruct` (base), plus LoRA adapters from
> `models/human_ft/` and `models/doped_ft/`. For each of the 3 models, run the
> 11-20 game prompt 50 times at temp=0.5. Register forward hooks on layers 5, 14, 22
> to capture the residual stream at the last input token position. Save activations as
> `.npy` files and choice distributions as `results/behavioral/results.json`. Print
> choice distributions per model at the end.

---

## Phase 4 — Linear Probe

**Goal**: Train a logistic regression classifier on activations to predict model variant
(Human-FT vs Doped-FT). High accuracy = the two models encode fundamentally different
representations at that layer.

**Design**:
- Load `human_ft_layerX.npy` (label=0) and `doped_ft_layerX.npy` (label=1) for each layer
- `StandardScaler` → `LogisticRegression` → 5-fold `cross_val_score`
- Do this for all 3 layers (5, 14, 22)
- Output: bar chart with accuracy per layer, error bars = std across folds
- Save as `results/figures/linear_probe.png`
- Print accuracy table to stdout

**Script**: `src/phase4_probe.py`

**How to run**:
```bash
python src/phase4_probe.py
```

**Kick-off prompt for new session**:
> Read `docs/pocPlan.md` Phase 4. Write `src/phase4_probe.py` that loads
> `results/activations/human_ft_layerX.npy` (label 0) and
> `results/activations/doped_ft_layerX.npy` (label 1) for layers 5, 14, 22.
> Train a StandardScaler + LogisticRegression pipeline with 5-fold CV for each layer.
> Print accuracy per layer and save a bar chart with error bars to
> `results/figures/linear_probe.png`.

---

## Phase 5 — Latent Thinking Vector

**Goal**: Find the direction in activation space that separates Human-FT from Doped-FT.
Show that Base model sits closer to Human-FT, confirming that doped fine-tuning pushes
the model away from the human-shaped geometry.

**Design**:
- For each layer: compute `v = mean(doped_ft) - mean(human_ft)`, normalize to unit vector
- Project all activations (all 3 models × 50 samples) onto `v`
- For the layer with highest probe accuracy: KDE plot showing 3 overlapping distributions
  (Base, Human-FT, Doped-FT) along the latent direction
- Save as `results/figures/latent_vector_layerX.png`
- Print mean projection value per model per layer

**Script**: `src/phase5_vector.py`

**How to run**:
```bash
python src/phase5_vector.py
```

**Kick-off prompt for new session**:
> Read `docs/pocPlan.md` Phase 5. Write `src/phase5_vector.py` that loads all 9
> `.npy` activation files from `results/activations/`. For each layer, compute the
> latent direction `v = mean(doped_ft) - mean(human_ft)` (normalized). Project all
> 3 models' activations onto `v`. For the layer with highest linear probe accuracy
> (check `results/figures/linear_probe.png` or hardcode from Phase 4 output), produce
> a KDE plot of the 3 projection distributions. Save to
> `results/figures/latent_vector_layerX.png`. Print mean projections per model.

---

## Phase 6 — Attention Pattern Visualization + Activation Patching

**Goal**: Identify which specific attention head encodes the strategic rule ("one less than
opponent"). Use activation patching to confirm: swap that head's output from Human-FT into
Doped-FT and check if behavior shifts toward human choices.

**Design**:

**Part A — Attention visualization**:
- For the game prompt, extract attention weights from all heads at all 3 layers (5, 14, 22)
  for all 3 models, using `output_attentions=True`
- For each layer, plot a heatmap grid: rows = heads, columns = input tokens
- Look for the head where Human-FT and Doped-FT diverge most (high attention weight on
  the "one less" or "20 shekels" tokens)
- Save heatmaps as `results/figures/attn_layerX.png`

**Part B — Activation patching**:
- Run the game prompt on Human-FT, cache the output of the identified head `(layer, head)`
- Run the same prompt on Doped-FT, but replace that head's output with Human-FT's cached value
  using a forward hook
- Generate a response from the patched Doped-FT model
- Repeat 20 times, compare choice distribution of patched vs unpatched Doped-FT
- Print results: does patching shift choices toward human range (15-17)?

**Script**: `src/phase6_attention.py`
- Hardcode `target_layer` and `target_head` based on visual inspection of Part A output
  (update constants at top of file)

**How to run**:
```bash
# Part A: generate attention heatmaps, inspect output to identify target head
python src/phase6_attention.py --part a

# Update TARGET_LAYER and TARGET_HEAD constants in src/phase6_attention.py
# then run Part B:
python src/phase6_attention.py --part b
```

**Kick-off prompt for new session**:
> Read `docs/pocPlan.md` Phase 6. Write `src/phase6_attention.py` with `--part a/b`.
> Part A: extract attention weights (`output_attentions=True`) from all heads at layers
> 5, 14, 22 for all 3 model variants on the game prompt. Plot heatmaps (rows=heads,
> cols=tokens) per layer, save to `results/figures/attn_layerX.png`. Part B: activation
> patching — cache the output of `TARGET_LAYER`/`TARGET_HEAD` from Human-FT, inject
> into Doped-FT via forward hook, run 20 game sessions, compare choice distribution
> vs unpatched Doped-FT. Both constants are defined at top of the file.

---

## Expected Results Summary

| Metric | Expected Finding |
|---|---|
| Behavioral (Phase 3) | Human-FT shifts toward 15-17; Doped-FT stays at 17-19; Base near 19-20 |
| Linear probe (Phase 4) | Layer 14 or 22 reaches >80% accuracy distinguishing Human-FT vs Doped-FT |
| Latent vector (Phase 5) | Base and Human-FT cluster together; Doped-FT separates along the direction |
| Activation patching (Phase 6) | Patching the strategic head from Human-FT into Doped-FT shifts choices toward 15-17 |
