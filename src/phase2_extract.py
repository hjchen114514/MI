# src/phase2_extract.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

        generated_ids = output_ids[0, input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        choice = parse_choice(response)

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
            act = extractor.activations.get(layer_idx)
            if act is None:
                raise RuntimeError(
                    f"Session {session_idx}: hook did not capture layer {layer_idx}. "
                    f"Captured layers: {sorted(extractor.activations.keys())}"
                )
            all_activations[layer_idx].append(act)

        if (session_idx + 1) % 10 == 0:
            print(f"  Session {session_idx + 1}/{config.NUM_SESSIONS} done")

    return all_activations, labels


def save_results(all_activations, labels, model_name):
    act_dir = os.path.join(config.RESULTS_ACTIVATIONS, model_name)
    os.makedirs(act_dir, exist_ok=True)
    for layer_idx in range(config.NUM_LAYERS):
        tensors = all_activations[layer_idx]
        arr = np.stack([t.numpy() for t in tensors], axis=0)  # (100, 960)
        path = os.path.join(act_dir, f"layer_{layer_idx:02d}.npy")
        np.save(path, arr)

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
    human_like = sum(1 for l in labels if l["label"] == 0)
    ai_like = sum(1 for l in labels if l["label"] == 1)

    print(f"\n  Choice distribution ({model_name}):")
    print(f"  {'Number':>8}  {'Count':>6}  Bar")
    print(f"  {'-'*8}  {'-'*6}  {'-'*20}")
    for n in range(11, 21):
        c = counts.get(n, 0)
        bar = "█" * c
        marker = " ← cutoff" if n == config.HUMAN_LIKE_MAX else ""
        print(f"  {n:>8}  {c:>6}  {bar}{marker}")
    if unparsed:
        print(f"  {'N/A':>8}  {unparsed:>6}  (unparseable)")
    print(f"\n  Labels: human-like (≤{config.HUMAN_LIKE_MAX}): {human_like}  |  AI-like (≥{config.AI_LIKE_MIN}): {ai_like}")


def main():
    print("=" * 60)
    print("  PHASE 2 — Session Extraction")
    print("=" * 60)
    print(f"  Sessions per model : {config.NUM_SESSIONS}")
    print(f"  Layers to capture  : {config.NUM_LAYERS}")
    print(f"  Temperature        : {config.TEMPERATURE}  |  Max new tokens: {config.MAX_NEW_TOKENS}")
    print(f"  Label cutoff       : ≤{config.HUMAN_LIKE_MAX} human-like  |  ≥{config.AI_LIKE_MIN} AI-like")
    print("=" * 60)

    models_to_run = [
        ("base",       config.MODEL_BASE),
        ("human_ft",   config.MODEL_HUMAN_FT),
        ("doped_108",  config.MODEL_DOPED_108_FT),
        ("doped_12x9", config.MODEL_DOPED_12X9_FT),
    ]

    for model_name, adapter_path in models_to_run:
        # Skip if adapter directory exists but is empty (model not yet trained)
        if adapter_path is not None and (
            not os.path.exists(adapter_path) or
            not os.listdir(adapter_path)
        ):
            print(f"\n  [{model_name}] Skipping — adapter not found at {adapter_path}")
            continue
        label = "base model" if adapter_path is None else adapter_path
        print(f"\n  [{model_name}] Loading from {label} ...")
        model, tokenizer, device = load_model(adapter_path)
        extractor = ResidualStreamExtractor(model)

        print(f"  [{model_name}] Running {config.NUM_SESSIONS} sessions ...")
        all_activations, labels = run_sessions(model, tokenizer, device, extractor)
        extractor.remove()

        save_results(all_activations, labels, model_name)
        print_distribution(labels, model_name)
        print(f"\n  [{model_name}] Done.")
        print("-" * 60)

        del model
        if device == "mps":
            torch.mps.empty_cache()

    print("\n  PHASE 2 complete. Activations and labels saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
