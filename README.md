## Research Question

1, does fine-tuning a language model on AI-augmented ("doped") reasoning data corrupt its internal representations, not just its output behavior? 

2, if corruption exists, is it because the model only learned to mimic the surface patterns of the doped reasoning text rather than developing genuine strategic thoughts and does this explain why doped-fine-tuned models cannot act as valid human surrogates?

---

## Experiment Overview

In this repo currently, I am able to identify layer k which is the most corruped layer in a residual stream after fine-tuning using doped data. Which then later I can analyze the attention head inside layer k to find out which attention head is most corrupted.

1. Started by generating doped datasets using Gemini 3 Thinking Mode for Doped-108 and Doped 36*3. Doped_108 is just a AI-rewritten version of the original human data that keeps the original number output, but different reasoning. 

Doped_36*3 takes 36 samples in the original human data through stratified sampling scripts to maintain the original response distribution and each sample is rewritten 3 times to keep the same 108 level as the human data. 

I also created this doped_36x3_ft (originally doped_12x9_ft) becasue I want to simulate researchers having limited human data and use AI to project up the dataset in real life scenarios. My hypothesis of this dataset is to be more corrupted than the doped_108 dataset, because it has less reasoning diversity despite having the same response distribution.

2. Then used a hard cutoff of human-like (<=18) and AI_like (>=19) to label AI responses for expeirment later. This hard-cutoff is derived from the paper https://www.pnas.org/doi/epdf/10.1073/pnas.2501660122. This paper highlights that most LLMs defaults to 19 or 20, hence we took 19 and 20 as AI-like and the rest for Human-like.

This cutoff originally was calculated through Yousen's J using the output of running both base and human_ft models 100 times but around 98% of the SmolLM2 model is 20 without fine tuning. Hence this method choses 19<= as the cutoff for Human-Like, however, since the ft_models stop producing 20 as their results, thus, ended up with 0 AI-like results which makes the expeirment unable to proceed.

Question: how was it calcualted, FPR, TPR, how, how pick 20 98% of the time lead to 0

3. Lora fine tune the model using the datasets to create 3 ft models, 1. human_ft; 2. doped_108_ft; 3. doped_36x3_ft in the models folder using the scripts in src/step2/phase1

4. Ran 100 sessions on human_ft, doped_108_ft, and doped_36x3_ft and extract the residual stream vector accross all 32 layers of the last generaton token using pytorch. the vector result is stored in results/activations. And the responses result is stored in results/responses. The labels of the responses is stored. in results/labels to identify the number of AI_like and Human-like for calculating the latent vector later. This is done using scripts in src/step2/phase2. Each session also has the same seeding accross human_ft, doped_108_ft, doped_36x3_ft to ensure control experiment.

We chose the last token because it is the only position that atteneded to the entire response and question, so has the most information.

5. Ran latent thinking vector calculation script in src/step2/phase3, where it is calculated by 80 sessions of human_ft results:

latent_vector at each layer = mean(Human-FT training sessions labeled AI-like) − mean(Human-FT training sessions labeled human-like)

This vector can tell us if a residual stream vector is more towards AI-like(positive value) or Human-like value(negative)

We used human_ft models result to construct the latent thinking vector because it is supposed to reflect human-reasoning, hence can be used to measure the deviation of AI-doped data from human reasoning.

We only used results from 80 sessions to calculate the latent thinking vector and the other 20 is used to show the how much the doped-ft models deviates from the human_ft models. 

6. Then we use the other 20 human_ft sessions' result and 100 sessions from both doped_108_ft and doped_36x3_ft to compare all residual stream vector accross all layers to plot the KDE graph to visualize the differences between the human_ft model result and the ft models using doped data. The x-axis is the projection score of each residual stream vector of the humam_ft and doped_ft. It is calculated by dot multplying with the latent thinking vector to get a score. 

Their differences symbolizes the internal corruptions caused by finetuning using doepd data. It is also quantified by cohen's d which shows the distances between the most densed choice in human_ft model and doped ft models. JSD is also used to show the similarity between human_ft and doped_ft models overall.

7. In this case, by finding which layer that has the biggest difference between the human_ft and doped_ft, we are able to identify layer k which is the most corrupted layer by doped fine tuning.


---

## Installation

**Requirements:** Python 3.10+
```bash

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install torch transformers peft datasets accelerate numpy scipy scikit-learn matplotlib seaborn
```

### HuggingFace Login (required to download Llama models)

```bash
.venv/bin/python -c "from huggingface_hub import login; login()"
```

The model `HuggingFaceTB/SmolLM2-360M-Instruct` will be auto-downloaded on first run and cached at `~/.cache/huggingface/`.

---

## Codebase Overview

This is a **Proof-of-Concept** using SmolLM2-360M-Instruct (360M params, 32 layers, 960-dim residual stream). The full research will repeat with Llama-3.1-8B-Instruct and Mistral-7B-Instruct. I tried to run Llama-3.1-8B on my M4 air Macbook and it failed to run 10 sessions in 30 minutes.

```
MI/
├── config.py                        # single source of truth — all constants and paths
├── data/
│   ├── human.jsonl                  # 108 authentic human responses (Arad & Rubinstein 2012)
│   ├── doped_108.jsonl              # 108 AI-rewritten responses (mild doping)
│   ├── doped_36x3.jsonl             # 36 seeds × 3 rewrites = 108 examples (strong doping)
│   └── doped_36x3_seeds.jsonl       # 36 stratified seeds before rewriting
├── models/
│   ├── human_ft/                    # LoRA adapter: fine-tuned on human.jsonl
│   ├── doped_108_ft/                # LoRA adapter: fine-tuned on doped_108.jsonl
│   └── doped_36x3_ft/               # LoRA adapter: fine-tuned on doped_36x3.jsonl
├── results/
│   ├── activations/
│   │   ├── base/                    # base model (no adapter), shape (100, 960) per layer
│   │   ├── human_ft/                # layer_00.npy … layer_31.npy, shape (100, 960)
│   │   ├── doped_108/               # layer_00.npy … layer_31.npy
│   │   └── doped_36x3/              # layer_00.npy … layer_31.npy
│   ├── labels/
│   │   ├── base.json                # [{session_id, choice, label}, ...] for 100 sessions
│   │   ├── human_ft.json
│   │   ├── doped_108.json
│   │   └── doped_36x3.json
│   ├── projections/
│   │   ├── human_ft_held_out.npy    # shape (32, 20) — held-out Human-FT sessions
│   │   ├── doped_108.npy            # shape (32, 100)
│   │   └── doped_36x3.npy           # shape (32, 100)
│   ├── responses/                   # raw response distributions per model
│   ├── figures/
│   │   ├── doped_108/               # 32 KDE plots (layer_00_kde.png … layer_31_kde.png) + layer_summary.csv
│   │   └── doped_36x3/              # same, for doped_36x3 vs human_ft
│   └── cutoff.json                  # active cutoff = 18 (≤18 human-like, ≥19 AI-like)
├── src/
│   ├── utils/
│   │   ├── model_loader.py          # load base / PEFT-adapted model
│   │   ├── hooks.py                 # ResidualStreamExtractor — hooks on all 32 layers
│   │   └── game.py                  # 11-20 game prompt + response parser
│   ├── step1/
│   │   ├── run_base.py              # run base model 100 sessions, save response distribution
│   │   └── find_cutoff.py           # compute cutoff for human vs base responses
│   └── step2/
│       ├── phase1_finetune.py       # LoRA fine-tune: --dataset <path> --output <dir>
│       ├── phase2_extract.py        # run 100 sessions per model, extract residual streams
│       ├── phase3_latent.py         # compute latent thinking vectors + project all sessions
│       └── phase4_visualize.py      # KDE plots, Cohen's d per layer, identify Layer K
└── scripts/
    ├── stratified_sample.py         # draw 36 stratified seeds from human.jsonl
    ├── wrap_doped.py                # convert raw Claude output into messages JSONL format
    ├── relabel.py                   # reapply hard cutoff to saved label files
    ├── regen_histograms.py          # regenerate response histograms
    └── data_distributions.py        # print choice distribution for each training dataset
```

### Key design choices

- **Cutoff:** ≤18 = human-like (label 0), ≥19 = AI-like (label 1). Grounded in Arad & Rubinstein 2012 — human responses peak at 17 (level-3 reasoning). I also tried to find the cutoff using youden's j method, but the smollm2 model only produces "20" for responses wihtout fine tuning and affect the later results greatly, hence not used the step 1 method to compute.

- **Activation extraction:** After generation of response, one extra forward pass on `[prompt + full response]` with `use_cache=False`. The residual stream at the last generated token is captured. This gives genuine session-level variation (different seeds → different responses → different token sequences → different activations).

- **Latent thinking vector:** Computed from Human-FT's training sessions only (80/20 stratified split). Points in the direction that separates human-like from AI-like internal states. Unit-normalized for cross-layer comparability. This determines whether the residual stream vector at each layer is more AI-like or Human like to detect whether the model is corrupted by doped data fine tuning.

- **Layer K metric:** Cohen's d (not raw mean difference) to determine how far away the most densed projection score of the human-ft model and the doped-ft model. Also using Jensen J to detemine the level of overlap of the human-ft and doped-ft model's figure (but was not used later due to reasons specified before, should be good to use with a more advanced model)

---

## How to Run

### End-to-end (full pipeline)

```bash
source .venv/bin/activate

# Step 1 — base model behavioral run (optional, cutoff already derived)
python src/step1/run_base.py
python src/step1/find_cutoff.py

# Phase 1 — LoRA fine-tuning (adapters already saved in models/)
python src/step2/phase1_finetune.py --dataset data/human.jsonl      --output models/human_ft
python src/step2/phase1_finetune.py --dataset data/doped_108.jsonl  --output models/doped_108_ft
python src/step2/phase1_finetune.py --dataset data/doped_36x3.jsonl --output models/doped_36x3_ft

# Phase 2 — extract residual stream activations (all 4 conditions)
python src/step2/phase2_extract.py

# Phase 3 — compute latent thinking vectors + projection scores
python src/step2/phase3_latent.py

# Phase 4 — KDE plots, Layer K identification
python src/step2/phase4_visualize.py
```

**Note:** Models and results are already committed. You only need to re-run if you change fine-tuning or extraction.

### Where are the results?

| Artifact | Location |
|---|---|
| KDE plots (Human-FT vs doped_108, per layer) | `results/figures/doped_108/layer_XX_kde.png` |
| KDE plots (Human-FT vs doped_36x3, per layer) | `results/figures/doped_36x3/layer_XX_kde.png` |
| Layer K summary table (Cohen's d, JSD, mean diff per layer) | `results/figures/doped_108/layer_summary.csv` and `results/figures/doped_36x3/layer_summary.csv` |
| Raw projection scores | `results/projections/*.npy` |
| Session choices + labels | `results/labels/*.json` |
| Response histograms | `results/responses/*/histogram.png` |

---

## Dataset Construction

| Dataset | Description | N unique scenarios | Examples |
|---|---|---|---|
| `human.jsonl` | Authentic human responses from Arad & Rubinstein 2012 | 108 | 108 |
| `doped_108.jsonl` | All 108 human responses rewritten by Gemini 3 Thinking Mode (mild doping) | 108 | 108 |
| `doped_36x3.jsonl` | 36 stratified seeds × 3 rewrites by Gemini 3 Thinking Mode (strong doping) | 36 | 108 |

The doped datasets preserve the **choice distribution** of the human data (values 15–20 in similar proportions), but the **reasoning text** is AI-generated. This is the realistic scenario: a researcher augments limited human data with AI paraphrases, keeps the chosen values, and fine-tunes a surrogate model on the result.

The **36×3 design** serves for the purpose of simulating real-life scenario: a researcher has limited human data, hence need to use AI to generate doped data with same response distribution and similar reaosning, hence leads to lack of diversity in reasoning.

---

## Experiment Analysis

### Run 1 — doped_12x9 (2026-04-05, now it is doped_36x3)

**Response distributions (100 sessions each, cutoff ≤18 = human-like):**

| Model | Human-like (≤18) | AI-like (≥19) |
|---|---|---|
| base | 15 | 85|
| human_ft | 40 | 60 |
| doped_108 | 39 | 61 |
| doped_12x9 | 59 | 41 |

**Layer K results:**

| Condition | Layer K | Cohen's d | JSD |
|---|---|---|---|
| doped_108 | 0 | **2.74** | 0.479 |
| doped_12x9 | 0 | **3.35** | 0.585 |

**Observations:**

1. Both doped_108 and doped_12x9 has different outputs and deviates from the human-ft model with Cohen's d around 3 and JSD around 0.5.
The standard thresholds for Cohen's d are: small = 0.2, medium = 0.5, large = 0.8. Anything above 2 is a massive effect.
2. Layer K is identified at layer 0 for both ft models, where both cohen's d and JSD are the most.
3. doped_12*9 ft model has more Human-like responses than the human-ft model and the doped-108 ft model despite having less diversity in reasoning due to projecting from 12 to 108. This is unexpected and would be interesting to see why.
4. Despite Doped 12*9 has more Human-like responses, its cohen's d at layer 0 is higher than doped_108 ft model which shows more corruptions, as in its difference than the human-ft model.

---

### Run 2 — doped_36x3 (2026-04-06, current)

I created a new dataset of doped_36*3 with 36 straitified sampled from the original human data set, because I found out it matches the original answer ditribution more closely than 12x9. So we should use this as the real output.

**Response distributions:**

| Model | Human-like (≤18) | AI-like (≥19) |
|---|---|---|---|
| human_ft | 33| 67 |
| doped_108 | 38 | 62 | 
| doped_36x3 | 40 | 60 |

**Layer K results:**

| Condition | Layer K | Cohen's d | JSD |
|---|---|---|---|
| doped_108 | 21 | 2.56 | 0.408 |
| doped_36x3 | 21 | 2.22 | 0.419 |

**Observations:**

1. Both doped_ft models deviates form the human_ft model with a JSD around 0.4 and Cohen's d around 2.5, in this case layer k is 21.
2. doped_36*3 ft model still has more Human-like responses than the human-ft model and the doped-108 ft model. This is unexpected and would be interesting to see why. I think the output does not correctly reflect transformers' internal.
3. Human_ft this run has even less Human-like responses from the first run, is because LoRA finetuning is stochastic, in other words, random. Since every run uses its own human-ft LoRA results to build the latent thinking vector, this is fine.

**Conclusion**
1. Fine-tuning SMOLlm2 on AI-Synthetic data will deviate it the most on Layer 21 compared to fine-tuning with real human data.
2. AI-Synthetic fine-tuned model have similar amount of human-like results compared to the human-fine-tuned model.
3. Despite reduced reasoning diversity, finetuning SMOLlm2 with stratified sampled data set like 12*9 and 36*3 both have significantly more human-like results than both human-fine-tuned model and regular AI-fine-tuned-Model. This is counter-intuitive, and I would like to find out why. 12*9 and 36*3 means they are stratified sampled with 12/36 out of 108 piece of data and then scaled by 9 or 3 to simulate the use case of creating large AI-synthetic dataset with limited data source.

**Next Step**
I am curious about the results of using a better model like Llama which my hardware does not allow me to. The experiment result is also unexpected as in doped_36x3 and doped_12x9 both has more human_like responses than the human_ft model in both run. And both run gives different layer K which surpirses me and I wonder why.

I also need to build step 3 and step 4 to identify the attention head with the most corruption from AI-doped data fine-tuning.


## Key References

- Arad, A. & Rubinstein, A. (2012). The 11-20 money request game: A level-k reasoning study. *American Economic Review*, 102(7), 3561–3573.
- Gao et al. (2025). Take Caution in Using LLMs as Human Surrogates. (`docs/gao-et-al-2025-...pdf`)
- Towards Monosemanticity — Anthropic (2023). Key reference for SAE methodology.
