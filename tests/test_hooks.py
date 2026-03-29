import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import config
from src.utils.model_loader import load_model
from src.utils.hooks import ResidualStreamExtractor
from src.utils.game import get_messages

print("Loading model...")
model, tokenizer, device = load_model()
extractor = ResidualStreamExtractor(model)

messages = get_messages()
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
input_len = input_ids.shape[1]

extractor.set_input_length(input_len)
torch.manual_seed(0)
with torch.no_grad():
    model.generate(input_ids, max_new_tokens=10, do_sample=True, temperature=0.5)

# Check all 32 layers were captured
assert len(extractor.activations) == config.NUM_LAYERS, \
    f"Expected {config.NUM_LAYERS} layers, got {len(extractor.activations)}"
print(f"Captured layers: {len(extractor.activations)}")

# Check shape at each boundary layer
for layer_idx in [0, 15, 31]:
    shape = extractor.activations[layer_idx].shape
    assert shape == (config.RESIDUAL_DIM,), \
        f"Layer {layer_idx}: expected ({config.RESIDUAL_DIM},), got {shape}"
    print(f"Shape at layer {layer_idx}: {shape}")

# Check activations are on CPU
assert extractor.activations[0].device.type == "cpu", "Activations should be on CPU"
print("Activations device: cpu")

# Check set_input_length resets state correctly
extractor.set_input_length(input_len)
assert len(extractor.activations) == 0, "set_input_length should clear activations"
print("set_input_length reset: OK")

extractor.remove()
assert len(extractor._handles) == 0, "remove() should clear handles"
print("remove() cleanup: OK")

print("test_hooks: all assertions passed")
