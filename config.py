# config.py
import os

# Model — SmolLM2-360M-Instruct (PoC; switch to meta-llama/Llama-3.2-3B-Instruct on better hardware)
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
NUM_LAYERS = 32
RESIDUAL_DIM = 960
NUM_ATTENTION_HEADS = 15

# Inference
NUM_SESSIONS = 100
TEMPERATURE = 0.5
TOP_P = 1.0        # disabled — isolates temperature as sole sampling variable
TOP_K = 0          # disabled
MAX_NEW_TOKENS = 300
DO_SAMPLE = True
SEEDS = list(range(100))   # session i uses seed i, identical across all models

# Labeling — cutoff is derived empirically via Youden's J in step1/find_cutoff.py
# These values are fallback defaults used if results/cutoff.json does not exist yet
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

# Paths — data
DATA_HUMAN = "data/human.jsonl"
DATA_DOPED_108 = "data/doped_108.jsonl"
DATA_DOPED_36X3 = "data/doped_36x3.jsonl"

# Paths — models
MODEL_BASE = None                              # base model, no adapter
MODEL_HUMAN_FT = "models/human_ft"
MODEL_DOPED_108_FT = "models/doped_108_ft"
MODEL_DOPED_36X3_FT = "models/doped_36x3_ft"

# Paths — results
RESULTS_ACTIVATIONS = "results/activations"
RESULTS_LABELS = "results/labels"
RESULTS_PROJECTIONS = "results/projections"
RESULTS_FIGURES = "results/figures"
RESULTS_RESPONSES = "results/responses"
CUTOFF_PATH = "results/cutoff.json"
