# src/utils/hooks.py
import torch
import config


class ResidualStreamExtractor:
    """
    Captures residual stream activations at the last prompt token position
    for all layers, on the first forward pass only (full prompt processing).
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
            # Generation steps have seq_len == 1 (one token at a time with KV cache)
            if hidden.shape[1] > 1 and layer_idx not in self.activations:
                pos = self._input_len - 1 if self._input_len else hidden.shape[1] - 1
                self.activations[layer_idx] = hidden[0, pos, :].detach().cpu()
        return hook

    def remove(self):
        """Deregister all hooks. Call when done with the model."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
