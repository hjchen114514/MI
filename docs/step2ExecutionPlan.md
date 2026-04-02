# Step 2 Execution Plan — Latent Thinking Vector (PoC)

## Research Goal
Identify which transformer layer is most corrupted by doped fine-tuning by computing a latent thinking vector at every layer and finding the layer where Doped-FT's internal representations have drifted furthest from Human-FT's.

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.2 | Model inference, forward hooks, tensor ops, MPS device |
| `transformers` | ≥4.40 | Load SmolLM2-360M-Instruct, tokenizer, generation |
| `peft` | ≥0.10 | LoRA fine-tuning and adapter loading |
| `datasets` | ≥2.18 | Load JSONL training data |
| `accelerate` | ≥0.27 | Required by PEFT for MPS training |
| `numpy` | ≥1.26 | Save/load `.npy` activation files, matrix ops |
| `scipy` | ≥1.12 | KDE (`gaussian_kde`), fixed bandwidth control |
| `scikit-learn` | ≥1.4 | Stratified train/test split |
| `matplotlib` | ≥3.8 | KDE plots, figure output |
| `seaborn` | ≥0.13 | Plot styling |

Install all at once:
```bash
pip install torch transformers peft datasets accelerate numpy scipy scikit-learn matplotlib seaborn
```

---

## Codebase Structure

```
MI/
├── config.py                      # single source of truth — all constants live here
├── data/
│   ├── human.jsonl                # 108 human responses (copy from DK/human.jsonl)
│   └── doped.jsonl                # 108 doped responses (copy from DK/dopedData.jsonl)
├── models/
│   ├── human_ft/                  # LoRA adapter weights saved after Phase 1
│   └── doped_ft/                  # LoRA adapter weights saved after Phase 1
├── results/
│   ├── activations/
│   │   ├── human_ft/
│   │   │   ├── layer_00.npy       # shape (100, 960) — 100 sessions × 960 residual dim
│   │   │   ├── layer_01.npy
│   │   │   └── ... layer_31.npy
│   │   └── doped_ft/
│   │       ├── layer_00.npy       # shape (100, 960)
│   │       └── ... layer_31.npy
│   ├── labels/
│   │   ├── human_ft.json          # [{session_id, choice, label}, ...] — 100 entries
│   │   └── doped_ft.json          # [{session_id, choice, label}, ...] — 100 entries
│   ├── projections/
│   │   ├── human_ft_held_out.npy  # shape (32, 20) — projection scores, 20 held-out sessions
│   │   └── doped_ft.npy           # shape (32, 100) — projection scores, all 100 sessions
│   └── figures/
│       ├── layer_00_kde.png        # Human-FT (blue) vs Doped-FT (red) KDE per layer
│       ├── layer_01_kde.png
│       └── ... layer_31_kde.png
└── src/
    ├── utils/
    │   ├── model_loader.py         # load base / human-ft / doped-ft, return (model, tokenizer)
    │   ├── hooks.py                # register forward hooks, extract residual stream at last input token
    │   └── game.py                 # prompt template for 11-20 game, parse chosen number from response
    ├── phase1_finetune.py          # LoRA fine-tune on --dataset, save adapter to --output
    ├── phase2_extract.py           # run sessions, extract activations all 32 layers, label inline
    ├── phase3_latent.py            # compute latent vectors, project sessions, save projection scores
    └── phase4_visualize.py         # KDE plots per layer, identify and print layer K
```

> **UPDATE [2026-04-02] — Revised Directory Structure (4 Conditions):**
> ```
> MI/
> ├── config.py
> ├── data/
> │   ├── human.jsonl
> │   ├── doped_108.jsonl            # RENAMED: 108 AI-rewritten human responses (mild doping)
> │   ├── doped_12x9_seeds.jsonl     # NEW: 12 stratified samples before rewriting
> │   └── doped_12x9.jsonl           # NEW: 12×9 stratified+rewritten (strong doping)
> ├── models/
> │   ├── human_ft/
> │   ├── doped_108_ft/              # RENAMED from doped_ft
> │   └── doped_12x9_ft/             # NEW
> ├── results/
> │   ├── activations/
> │   │   ├── base/                  # NEW — base model, no adapter
> │   │   ├── human_ft/
> │   │   ├── doped_108/             # RENAMED from doped_ft
> │   │   └── doped_12x9/            # NEW
> │   ├── labels/
> │   │   ├── base.json              # NEW
> │   │   ├── human_ft.json
> │   │   ├── doped_108.json         # RENAMED
> │   │   └── doped_12x9.json        # NEW
> │   ├── projections/
> │   │   ├── human_ft_held_out.npy
> │   │   ├── base.npy               # NEW
> │   │   ├── doped_108.npy          # RENAMED
> │   │   └── doped_12x9.npy         # NEW
> │   └── figures/
> │       ├── base/                  # NEW — 32 KDE plots per condition
> │       ├── doped_108/             # NEW
> │       └── doped_12x9/            # NEW
> └── scripts/
>     └── stratified_sample.py       # NEW — draws 12 stratified seeds from human.jsonl
> ```

---

## config.py — All Constants

```python
# Model
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
NUM_LAYERS = 32
RESIDUAL_DIM = 960

# Inference
NUM_SESSIONS = 100
TEMPERATURE = 0.5
TOP_P = 1.0          # disabled — isolates temperature as the only sampling variable
TOP_K = 0            # disabled
MAX_NEW_TOKENS = 300
DO_SAMPLE = True
SEEDS = list(range(100))   # session i uses seed i, same seeds across both models

# Labeling (Step 1 fixed cutoff, justified by Arad & Rubinstein 2012)
HUMAN_LIKE_MAX = 18        # choice ≤ 18 → label 0 (human-like)
AI_LIKE_MIN = 19           # choice ≥ 19 → label 1 (AI-like)

# Latent vector train/test split (Human-FT only)
TRAIN_RATIO = 0.8          # 80 sessions to compute vector, 20 held-out for unbiased projection

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

> **UPDATE [2026-04-02] — config.py paths updated for 4 conditions:**
> ```python
> # Paths — data
> DATA_HUMAN = "data/human.jsonl"
> DATA_DOPED_108 = "data/doped_108.jsonl"
> DATA_DOPED_12X9 = "data/doped_12x9.jsonl"
>
> # Paths — models
> MODEL_BASE = None                         # base model, no adapter
> MODEL_HUMAN_FT = "models/human_ft"
> MODEL_DOPED_108_FT = "models/doped_108_ft"
> MODEL_DOPED_12X9_FT = "models/doped_12x9_ft"
> ```

---

## Phase 0 — Environment & Directory Setup

**What it does:** Creates the virtual environment, installs all dependencies, builds the directory skeleton, and copies data files into place.

**How to run:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers peft datasets accelerate numpy scipy scikit-learn matplotlib seaborn
mkdir -p data models/human_ft models/doped_ft
mkdir -p results/activations/human_ft results/activations/doped_ft
mkdir -p results/labels results/projections results/figures
cp DK/human.jsonl data/human.jsonl
cp DK/dopedData.jsonl data/doped.jsonl
```
Then create `config.py` with the constants above.


---

## Phase 1 — LoRA Fine-Tuning

**What it does:** Fine-tunes two LoRA adapters on top of the frozen SmolLM2-360M-Instruct base — one on human reasoning data (Human-FT), one on doped reasoning data (Doped-FT). Each adapter is saved to its output directory for use in all later phases.

**Key design decisions:**
- Base model weights are fully frozen; only the LoRA adapter parameters (q_proj, v_proj) are trained
- `--dataset` arg selects which JSONL to train on; `--output` arg selects where to save the adapter
- Loss is printed per epoch so you can verify training is converging
- Training runs on MPS (Apple Silicon); falls back to CPU automatically if MPS unavailable

**How to run:**
```bash
# Fine-tune on human data
python src/phase1_finetune.py --dataset data/human.jsonl --output models/human_ft

# Fine-tune on doped data
python src/phase1_finetune.py --dataset data/doped.jsonl --output models/doped_ft
```

**Expected output:** `models/human_ft/` and `models/doped_ft/` each containing PEFT adapter files (`adapter_config.json`, `adapter_model.safetensors`). Loss should decrease across 3 epochs.

> **UPDATE [2026-04-02] — Three fine-tunes now (base needs no training):**
> ```bash
> python src/phase1_finetune.py --dataset data/human.jsonl     --output models/human_ft
> python src/phase1_finetune.py --dataset data/doped_108.jsonl --output models/doped_108_ft
> python src/phase1_finetune.py --dataset data/doped_12x9.jsonl --output models/doped_12x9_ft
> ```
> Base model requires no fine-tuning — it is loaded directly in Phase 2 with `adapter_path=None`.


---

## Phase 2 — Session Extraction + Labeling

**What it does:** The core data collection phase. Loads both fine-tuned models, runs 100 independent game sessions per model (seed 0–99), registers forward hooks on all 32 layers to capture residual stream activations at the last input token position mid-generation, parses the chosen number from each response, applies the fixed cutoff to assign a label, and saves everything.

**Key design decisions:**
- Labels are assigned inline during this phase (option C) — no separate Step 1 needed
- The residual stream at the **last input token position** (not the generated tokens) is extracted. This captures what the model "knows" about the game prompt before it starts generating — the mechanistically meaningful signal
- Seeds 0–99 are used identically for both Human-FT and Doped-FT, so session i of both models experienced the same random draw. This makes session-level debugging straightforward
- All 32 layers are extracted in a single forward pass per session via registered hooks — no repeated passes needed. Important: `model.generate()` fires hooks on every forward pass (one per generated token). `hooks.py` must capture only the **first** forward pass (the one processing the full prompt) by deregistering hooks after the first activation is captured
- Activations are saved as `layer_XX.npy` of shape `(100, 960)` — one file per layer per model

**How to run:**
```bash
python src/phase2_extract.py
```

**Expected output:**
- `results/activations/human_ft/layer_00.npy` through `layer_31.npy`, each shape `(100, 960)`
- `results/activations/doped_ft/layer_00.npy` through `layer_31.npy`, each shape `(100, 960)`
- `results/labels/human_ft.json` — 100 entries: `{session_id, choice, label}`
- `results/labels/doped_ft.json` — 100 entries: `{session_id, choice, label}`
- Prints choice distribution per model at end (how many picked each number 11–20)

> **UPDATE [2026-04-02] — Phase 2 now runs 4 conditions:**
> Models list: `[base, human_ft, doped_108, doped_12x9]`. The script auto-skips any model whose adapter directory is empty (e.g. doped_12x9 before that data is ready). Run order and outputs are the same but replicated across all 4 conditions. The base model is loaded with `adapter_path=None`.


---

## Phase 3 — Latent Thinking Vector + Projection

**What it does:** Computes one latent thinking vector per layer using Human-FT's training sessions, then projects held-out Human-FT and all Doped-FT sessions onto those vectors to produce projection scores.

**Key design decisions:**
- **80/20 stratified split on Human-FT** — stratified by label (`random_state=42`) so both the AI-like and human-like classes are represented in the 80 training sessions. This matters especially if the label distribution is skewed (e.g., Human-FT picks human-like values most of the time)
- **Latent vector computed from Human-FT training sessions only** — ensures the vector reflects genuine human-like vs AI-like reasoning geometry, not Doped-FT's corrupted geometry
- **Vector is unit-normalized** — makes projection scores comparable across layers regardless of each layer's activation magnitude scale
- **Only held-out Human-FT sessions (20) are projected** — the 80 training sessions were used to build the vector and must not be projected onto it (circular bias)
- **All 100 Doped-FT sessions are projected** — the vector was built entirely from Human-FT, so Doped-FT sessions are unbiased regardless of split
- Projection score = dot product of session activation with latent vector → one scalar per session per layer
- Output shapes: Human-FT `(32, 20)`, Doped-FT `(32, 100)`

**How to run:**
```bash
python src/phase3_latent.py
```

**Expected output:**
- `results/projections/human_ft_held_out.npy` — shape `(32, 20)`
- `results/projections/doped_ft.npy` — shape `(32, 100)`

> **UPDATE [2026-04-02] — Phase 3 projects all available conditions:**
> Phase 3 now auto-detects which conditions have extracted activations and projects each one. Output files: `base.npy`, `doped_108.npy`, `doped_12x9.npy` (shape `(32, 100)` each), plus `human_ft_held_out.npy` (shape `(32, 20)`). Latent vector is still built from Human-FT training sessions only — unchanged.


---

## Phase 4 — Visualization + Layer K Identification

**What it does:** Loads the projection scores from Phase 3, plots 32 KDE figures (one per layer), and identifies layer K — the layer where Doped-FT's distribution is shifted furthest rightward from Human-FT.

**Key design decisions:**
- **Fixed KDE bandwidth** — both Human-FT and Doped-FT curves use the same bandwidth value, computed via Scott's rule on the larger dataset (Doped-FT, n=100): `bw = n**(-1/5)`. Passing this scalar as `bw_method` to both `gaussian_kde` calls corrects for the 20 vs 100 sample size imbalance that would otherwise make Human-FT's curve visually wider even if the true distributions were the same
- **Layer K identification is fully quantitative** — three metrics computed per layer from the raw projection scores and KDE estimates:
  - **Mean difference**: `mean(Doped-FT scores) − mean(Human-FT scores)` — directional, shows which way Doped-FT shifted on the human↔AI axis
  - **JSD (Jensen-Shannon Divergence)**: computed from the KDE probability estimates, bounded [0,1]. Consistent with Step 1's behavioral JSD metric
  - **Cohen's d**: `(mean_doped − mean_human) / pooled_std` — standardized effect size, interpretable as small/medium/large (0.2/0.5/0.8). Computed directly from raw scores, no KDE needed
  - Layer K = layer with highest **Cohen's d** (scale-invariant — mean_diff alone is biased by activation magnitude differences across layers). All three metrics reported for every layer
- Each KDE plot: x-axis = "projection score (human pole ← → AI pole)", y-axis = "session density", Human-FT in blue, Doped-FT in red, title = "Layer {i} | JSD={:.3f} | d={:.2f}"
- All 32 plots saved as `results/figures/layer_XX_kde.png`
- Summary table saved as `results/figures/layer_summary.csv` with columns: layer, mean_diff, jsd, cohens_d
- Layer K printed to stdout with all three metric values

**How to run:**
```bash
python src/phase4_visualize.py
```

**Expected output:**
- 32 PNG files in `results/figures/layer_XX_kde.png`
- `results/figures/layer_summary.csv` — 32 rows, columns: layer, mean_diff, jsd, cohens_d
- Stdout: `Layer K = {k} | mean_diff={:.4f} | JSD={:.4f} | Cohen's d={:.4f}`

> **UPDATE [2026-04-02] — Phase 4 runs per condition, Layer K by Cohen's d:**
> Phase 4 now iterates over all available projection files. For each condition, 32 KDE plots are saved to `results/figures/<condition>/` and a `layer_summary.csv` is written there. A final cross-condition summary table is printed comparing Layer K for all conditions side by side.
>
> **Layer K selection changed from `max(mean_diff)` to `max(Cohen's d)`** — Cohen's d is scale-invariant (normalized by pooled std), whereas raw mean_diff is biased by activation magnitude differences across layers.


---

## End-to-End Run Order

```bash
# Phase 0 — setup (manual)
# Phase 1 — fine-tune both models
python src/phase1_finetune.py --dataset data/human.jsonl --output models/human_ft
python src/phase1_finetune.py --dataset data/doped.jsonl --output models/doped_ft

# Phase 2 — extract activations + label sessions
python src/phase2_extract.py

# Phase 3 — compute latent vectors + project
python src/phase3_latent.py

# Phase 4 — visualize + identify layer K
python src/phase4_visualize.py
```

> **UPDATE [2026-04-02] — Current run order (4 conditions):**
> ```bash
> # Generate stratified seeds for doped_12x9 (then rewrite ×9 with Claude Sonnet 4.6)
> python scripts/stratified_sample.py
>
> # Phase 1 — fine-tune all three FT models
> python src/phase1_finetune.py --dataset data/human.jsonl      --output models/human_ft
> python src/phase1_finetune.py --dataset data/doped_108.jsonl  --output models/doped_108_ft
> python src/phase1_finetune.py --dataset data/doped_12x9.jsonl --output models/doped_12x9_ft
>
> # Phase 2 — extract all 4 conditions (base auto-included, skips missing adapters)
> python src/phase2_extract.py
>
> # Phase 3 — project all available conditions
> python src/phase3_latent.py
>
> # Phase 4 — visualize per condition, print cross-condition Layer K table
> python src/phase4_visualize.py
> ```

---

## Known Limitations (PoC)

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| 20 held-out Human-FT sessions vs 100 Doped-FT | Human-FT KDE noisier | Fixed KDE bandwidth applied to both curves |
| Fixed cutoff (≤18/≥19) rather than data-driven boundary | Less rigorous than ROC/GMM-derived boundary | Justified by A&R 2012 empirical data; replace with ROC/Youden in full research |
| SmolLM2-360M-Instruct (360M params) | Results may not generalize to larger models | Full research repeats with Llama-3-8B and Mistral-7B |
| MPS device (no CUDA) | Slower inference, no bfloat16 support on older MPS | Acceptable for PoC; full research runs on GPU |

> **UPDATE [2026-04-02] — Additional limitations:**
>
> | Limitation | Impact | Mitigation |
> |-----------|--------|-----------|
> | doped_12x9 only contains 12 unique reasoning scenarios repeated 9× | Model may overfit to those 12 patterns rather than learning general AI-style reasoning | Acknowledged as intentional amplification; doped_108 provides a milder comparison |
> | doped_108 is AI-rewritten from all 108 human responses without explicit AI reasoning injection | May not produce meaningfully different internal representations from human_ft | Serves as mild-doping baseline; doped_12x9 is the primary experimental condition |
> | Both doped conditions have nearly identical choice distributions to human data | Behavioral output alone cannot distinguish corruption | Latent vector analysis targets internal representations, not output distributions — this is the point |
