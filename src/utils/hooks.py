# src/utils/hooks.py
import torch
import config
from peft import PeftModel


class ResidualStreamExtractor:
    """
    Captures residual stream activations at a specified token position across all layers.

    Usage per session:
        extractor.reset()                          # silent during generation
        model.generate(...)                        # hooks do nothing
        extractor.set_for_extra_pass(target_pos)   # arm for extra pass
        model(full_ids, use_cache=False)           # hooks fire once per layer
        activations = extractor.activations        # dict: layer_idx -> (RESIDUAL_DIM,) tensor
    """

    def __init__(self, model):
        self.activations = {}     # layer_idx -> (RESIDUAL_DIM,) cpu tensor
        self.active = False       # hooks only capture when True
        self.target_pos = None    # which token position to extract
        self._handles = []
        self._register(model)

    def reset(self):
        """Call before each session. Clears activations and disarms hooks."""
        self.activations = {}
        self.active = False
        self.target_pos = None

    def set_for_extra_pass(self, target_pos: int):
        """Arm hooks for the extra forward pass. target_pos = last token index."""
        self.activations = {}
        self.active = True
        self.target_pos = target_pos

    def _register(self, model):
        # Unwrap PEFT adapter if present to reach the base LlamaForCausalLM
        # With PEFT:    PeftModel -> .base_model (LoraModel) -> .model (LlamaForCausalLM) -> .model.layers
        # Without PEFT: LlamaForCausalLM -> .model.layers
        causal_lm = model.base_model.model if isinstance(model, PeftModel) else model
        for layer_idx in range(config.NUM_LAYERS):
            handle = causal_lm.model.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if not self.active:
                return
            # output is a tuple; first element is hidden_states (batch, seq_len, hidden_dim)
            hidden = output[0] if isinstance(output, tuple) else output
            if layer_idx not in self.activations:
                self.activations[layer_idx] = hidden[0, self.target_pos, :].detach().cpu()
        return hook

    def remove(self):
        """Deregister all hooks. Call when done with the model."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
