# src/step2/phase2_extract.py
# Step 2, Phase 2 — Run 3 fine-tuned models, extract activations, save labels + response distributions.
# Base model is handled by src/step1/run_base.py — do NOT re-run it here.
# Requires: results/cutoff.json (from find_cutoff.py). Falls back to config.HUMAN_LIKE_MAX if missing.
#
# Run from project root: python src/step2/phase2_extract.py

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


def load_cutoff():
    """Load empirical cutoff from step1 output. Falls back to config default if not found."""
    if os.path.exists(config.CUTOFF_PATH):
        with open(config.CUTOFF_PATH) as f:
            data = json.load(f)
        cutoff = data["cutoff"]
        print(f"  Cutoff loaded from {config.CUTOFF_PATH}: <={cutoff} human-like, >={cutoff+1} AI-like")
        return cutoff
    print(f"  WARNING: {config.CUTOFF_PATH} not found. Using config default: <={config.HUMAN_LIKE_MAX}")
    return config.HUMAN_LIKE_MAX


def run_sessions(model, tokenizer, device, extractor, cutoff):
    messages = get_messages()
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    input_len = input_ids.shape[1]
    eos_id = tokenizer.eos_token_id

    all_activations = {i: [] for i in range(config.NUM_LAYERS)}
    labels = []

    for session_idx in range(config.NUM_SESSIONS):
        extractor.reset()   # hooks silent during generation
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

        # Strip prompt, then strip trailing EOS token if present
        generated_ids = output_ids[0, input_len:]
        if len(generated_ids) > 0 and generated_ids[-1].item() == eos_id:
            generated_ids = generated_ids[:-1]

        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        choice = parse_choice(response)
        actual_choice = choice if choice is not None else -1
        label = 0 if choice is not None and choice <= cutoff else 1
        labels.append({"session_id": session_idx, "choice": actual_choice, "label": label})

        # Edge case: generation produced nothing after EOS strip
        if len(generated_ids) == 0:
            print(f"  WARNING: session {session_idx} produced empty response — filling activations with zeros")
            for layer_idx in range(config.NUM_LAYERS):
                all_activations[layer_idx].append(torch.zeros(config.RESIDUAL_DIM))
            continue

        # Extra forward pass on (prompt + response) to get last-token residual stream
        full_ids = torch.cat([input_ids[0], generated_ids]).unsqueeze(0).to(device)
        target_pos = full_ids.shape[1] - 1   # last token index

        extractor.set_for_extra_pass(target_pos)
        with torch.no_grad():
            model(full_ids, use_cache=False)

        for layer_idx in range(config.NUM_LAYERS):
            act = extractor.activations.get(layer_idx)
            if act is None:
                raise RuntimeError(
                    f"Session {session_idx}: hook did not capture layer {layer_idx}. "
                    f"Captured: {sorted(extractor.activations.keys())}"
                )
            all_activations[layer_idx].append(act)

        if (session_idx + 1) % 10 == 0:
            print(f"  Session {session_idx + 1}/{config.NUM_SESSIONS} done")

    return all_activations, labels


def save_activations_and_labels(all_activations, labels, model_name):
    act_dir = os.path.join(config.RESULTS_ACTIVATIONS, model_name)
    os.makedirs(act_dir, exist_ok=True)
    for layer_idx in range(config.NUM_LAYERS):
        arr = np.stack([t.numpy() for t in all_activations[layer_idx]], axis=0)
        np.save(os.path.join(act_dir, f"layer_{layer_idx:02d}.npy"), arr)

    os.makedirs(config.RESULTS_LABELS, exist_ok=True)
    label_path = os.path.join(config.RESULTS_LABELS, f"{model_name}.json")
    with open(label_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"  Activations → {act_dir}/")
    print(f"  Labels      → {label_path}")


def save_responses(labels, model_name, cutoff):
    out_dir = os.path.join(config.RESULTS_RESPONSES, model_name)
    os.makedirs(out_dir, exist_ok=True)
    choices = [l["choice"] for l in labels]
    distribution = {str(n): sum(1 for c in choices if c == n) for n in range(11, 21)}
    human_like = sum(1 for l in labels if l["label"] == 0)
    ai_like = sum(1 for l in labels if l["label"] == 1)
    unparsed = sum(1 for c in choices if c == -1)

    data = {
        "model": model_name,
        "n_sessions": len(labels),
        "choices": choices,
        "distribution": distribution,
        "cutoff_applied": cutoff,
        "human_like_count": human_like,
        "ai_like_count": ai_like,
        "unparsed_count": unparsed,
    }
    json_path = os.path.join(out_dir, "responses.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Histogram
    counts = Counter(c for c in choices if c != -1)
    fig, ax = plt.subplots(figsize=(8, 4))
    nums = list(range(11, 21))
    vals = [counts.get(n, 0) for n in nums]
    colors = ["steelblue" if n <= cutoff else "crimson" for n in nums]
    ax.bar(nums, vals, color=colors, edgecolor="white")
    ax.axvline(x=cutoff + 0.5, color="black", linestyle="--", linewidth=1, label=f"cutoff ({cutoff}/{cutoff+1})")
    ax.set_xlabel("Choice", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{model_name} response distribution (n={sum(vals)})", fontsize=13)
    ax.set_xticks(nums)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    hist_path = os.path.join(out_dir, "histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    print(f"  Responses   → {json_path}")
    print(f"  Histogram   → {hist_path}")


def print_distribution(labels, model_name, cutoff):
    choices = [l["choice"] for l in labels if l["choice"] != -1]
    counts = Counter(choices)
    human_like = sum(1 for l in labels if l["label"] == 0)
    ai_like = sum(1 for l in labels if l["label"] == 1)
    unparsed = sum(1 for l in labels if l["choice"] == -1)

    print(f"\n  Choice distribution ({model_name}):")
    print(f"  {'Number':>8}  {'Count':>6}  Bar")
    print(f"  {'-'*8}  {'-'*6}  {'-'*20}")
    for n in range(11, 21):
        c = counts.get(n, 0)
        bar = "█" * c
        marker = " ← cutoff" if n == cutoff else ""
        print(f"  {n:>8}  {c:>6}  {bar}{marker}")
    if unparsed:
        print(f"  {'N/A':>8}  {unparsed:>6}  (unparseable)")
    print(f"\n  Labels: human-like (≤{cutoff}): {human_like}  |  AI-like (≥{cutoff+1}): {ai_like}")


def main():
    cutoff = load_cutoff()

    print("=" * 60)
    print("  STEP 2, PHASE 2 — FT Model Session Extraction")
    print("=" * 60)
    print(f"  Model      : {config.MODEL_ID}")
    print(f"  Sessions   : {config.NUM_SESSIONS}  |  Layers: {config.NUM_LAYERS}")
    print(f"  Temperature: {config.TEMPERATURE}  |  Max tokens: {config.MAX_NEW_TOKENS}")
    print(f"  Cutoff     : ≤{cutoff} human-like  |  ≥{cutoff+1} AI-like")
    print("=" * 60)

    models_to_run = [
        ("human_ft",    config.MODEL_HUMAN_FT),
        ("doped_108",   config.MODEL_DOPED_108_FT),
        ("doped_36x3",  config.MODEL_DOPED_36X3_FT),
    ]

    for model_name, adapter_path in models_to_run:
        if not os.path.exists(adapter_path) or not os.listdir(adapter_path):
            print(f"\n  [{model_name}] Skipping — adapter not found at {adapter_path}")
            continue

        print(f"\n  [{model_name}] Loading from {adapter_path} ...")
        model, tokenizer, device = load_model(adapter_path)
        extractor = ResidualStreamExtractor(model)

        print(f"  [{model_name}] Running {config.NUM_SESSIONS} sessions ...")
        all_activations, labels = run_sessions(model, tokenizer, device, extractor, cutoff)
        extractor.remove()

        print(f"\n  [{model_name}] Saving ...")
        save_activations_and_labels(all_activations, labels, model_name)
        save_responses(labels, model_name, cutoff)
        print_distribution(labels, model_name, cutoff)

        print(f"\n  [{model_name}] Done.")
        print("-" * 60)

        del model
        if device == "mps":
            torch.mps.empty_cache()

    print("\n  PHASE 2 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
