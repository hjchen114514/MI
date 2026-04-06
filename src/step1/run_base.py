# src/step1/run_base.py
# Step 1A — Run base model (no fine-tuning) for 100 sessions.
# Saves: results/responses/base.json, results/responses/base_histogram.png,
#        results/activations/base/, (no labels — run find_cutoff.py next)
#
# Run from project root: python src/step1/run_base.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    sessions = []

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
        sessions.append({"session_id": session_idx, "choice": choice if choice is not None else -1})

        for layer_idx in range(config.NUM_LAYERS):
            act = extractor.activations.get(layer_idx)
            if act is None:
                raise RuntimeError(f"Session {session_idx}: hook did not capture layer {layer_idx}")
            all_activations[layer_idx].append(act)

        if (session_idx + 1) % 10 == 0:
            print(f"  Session {session_idx + 1}/{config.NUM_SESSIONS} done")

    return all_activations, sessions


def save_activations(all_activations):
    act_dir = os.path.join(config.RESULTS_ACTIVATIONS, "base")
    os.makedirs(act_dir, exist_ok=True)
    for layer_idx in range(config.NUM_LAYERS):
        arr = np.stack([t.numpy() for t in all_activations[layer_idx]], axis=0)
        np.save(os.path.join(act_dir, f"layer_{layer_idx:02d}.npy"), arr)
    print(f"  Activations saved to {act_dir}/")


def save_responses(sessions):
    out_dir = os.path.join(config.RESULTS_RESPONSES, "base")
    os.makedirs(out_dir, exist_ok=True)
    choices = [s["choice"] for s in sessions]
    distribution = {str(n): sum(1 for c in choices if c == n) for n in range(11, 21)}
    unparsed = sum(1 for c in choices if c == -1)

    data = {
        "model": "base",
        "n_sessions": len(sessions),
        "sessions": sessions,
        "distribution": distribution,
        "unparsed_count": unparsed,
    }
    json_path = os.path.join(out_dir, "responses.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Response JSON saved to {json_path}")
    return choices


def save_histogram(choices):
    out_dir = os.path.join(config.RESULTS_RESPONSES, "base")
    os.makedirs(out_dir, exist_ok=True)
    counts = Counter(c for c in choices if c != -1)
    fig, ax = plt.subplots(figsize=(8, 4))
    nums = list(range(11, 21))
    vals = [counts.get(n, 0) for n in nums]
    ax.bar(nums, vals, color="steelblue", edgecolor="white")
    ax.set_xlabel("Choice", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Base model response distribution (n={sum(vals)})", fontsize=13)
    ax.set_xticks(nums)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved to {path}")


def print_distribution(choices):
    counts = Counter(c for c in choices if c != -1)
    total = sum(counts.values())
    print(f"\n  Base model choice distribution (n={total}):")
    for n in range(11, 21):
        c = counts.get(n, 0)
        bar = "█" * c
        print(f"  {n:>4}: {c:>3}  {bar}")
    unparsed = sum(1 for c in choices if c == -1)
    if unparsed:
        print(f"   N/A: {unparsed:>3}  (unparseable)")


def main():
    print("=" * 60)
    print("  STEP 1A — Base Model Behavioral Run")
    print("=" * 60)
    print(f"  Model    : {config.MODEL_ID}")
    print(f"  Sessions : {config.NUM_SESSIONS}  |  Temp: {config.TEMPERATURE}")
    print(f"  Layers   : {config.NUM_LAYERS}  |  Residual dim: {config.RESIDUAL_DIM}")
    print("-" * 60)
    print("  Loading model ...")

    model, tokenizer, device = load_model(adapter_path=None)
    print(f"  Device   : {device}")
    extractor = ResidualStreamExtractor(model)

    print(f"  Running {config.NUM_SESSIONS} sessions ...")
    all_activations, sessions = run_sessions(model, tokenizer, device, extractor)
    extractor.remove()

    choices = [s["choice"] for s in sessions]
    print_distribution(choices)

    print("\n  Saving outputs ...")
    save_activations(all_activations)
    save_responses(sessions)
    save_histogram(choices)

    print("\n  Done. Next: run python src/step1/find_cutoff.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
