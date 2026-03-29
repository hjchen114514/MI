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
