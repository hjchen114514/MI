import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from src.utils.model_loader import load_model

print("Loading base model (this will download ~700MB on first run)...")
model, tokenizer, device = load_model()

assert device in ("mps", "cpu"), f"Unexpected device: {device}"
print(f"Device: {device}")

assert len(model.model.layers) == config.NUM_LAYERS, \
    f"Expected {config.NUM_LAYERS} layers, got {len(model.model.layers)}"
print(f"Layers: {len(model.model.layers)}")

assert model.config.hidden_size == config.RESIDUAL_DIM, \
    f"Expected hidden_size {config.RESIDUAL_DIM}, got {model.config.hidden_size}"
print(f"Hidden dim: {model.config.hidden_size}")

assert tokenizer.pad_token is not None, "pad_token should not be None"
print(f"Pad token: {tokenizer.pad_token!r}")

print("test_model_loader: all assertions passed")
