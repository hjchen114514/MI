# src/step2/phase3_latent.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import config


def load_activations(model_name, layer_idx):
    path = os.path.join(config.RESULTS_ACTIVATIONS, model_name, f"layer_{layer_idx:02d}.npy")
    return np.load(path)   # shape (100, RESIDUAL_DIM)


def load_labels(model_name):
    path = os.path.join(config.RESULTS_LABELS, f"{model_name}.json")
    with open(path) as f:
        entries = json.load(f)
    return np.array([e["label"] for e in entries])   # shape (100,)


def compute_latent_vector(activations_train, labels_train):
    """
    latent_vector = mean(AI-like activations) - mean(human-like activations)
    Unit-normalized so projection scores are comparable across layers.
    """
    ai_mask = labels_train == 1
    human_mask = labels_train == 0

    mean_ai = activations_train[ai_mask].mean(axis=0)
    mean_human = activations_train[human_mask].mean(axis=0)

    vec = mean_ai - mean_human
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def project(activations, latent_vec):
    """Dot product of each session's activation with latent vector."""
    return activations @ latent_vec   # shape (N,)


def main():
    print("=" * 60)
    print("  PHASE 3 — Latent Thinking Vector")
    print("=" * 60)
    print(f"  Train/held-out split : {int(config.TRAIN_RATIO*100)}/{int((1-config.TRAIN_RATIO)*100)}  (random_state={config.SPLIT_RANDOM_STATE})")
    print(f"  Layers               : {config.NUM_LAYERS}")
    print(f"  Residual dim         : {config.RESIDUAL_DIM}")
    print("-" * 60)

    human_labels = load_labels("human_ft")

    # Stratified 80/20 split — computed once, same indices used across all layers
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - config.TRAIN_RATIO,
        random_state=config.SPLIT_RANDOM_STATE
    )
    indices = np.arange(config.NUM_SESSIONS)
    train_idx, held_out_idx = next(splitter.split(indices, human_labels))

    print(f"  Human-FT split → train: {len(train_idx)}  |  held-out: {len(held_out_idx)}")
    print(f"  Train label dist  → human-like: {(human_labels[train_idx]==0).sum()}  |  AI-like: {(human_labels[train_idx]==1).sum()}")
    print(f"  Held-out label dist → human-like: {(human_labels[held_out_idx]==0).sum()}  |  AI-like: {(human_labels[held_out_idx]==1).sum()}")

    # All conditions to project (must have extracted activations already)
    conditions = ["base", "doped_108", "doped_36x3"]
    available = [
        c for c in conditions
        if os.path.exists(os.path.join(config.RESULTS_ACTIVATIONS, c, "layer_00.npy"))
    ]
    print(f"\n  Conditions to project: {available if available else 'none yet (run phase2 first)'}")
    print("-" * 60)
    print("  Computing latent vectors and projecting ...")

    os.makedirs(config.RESULTS_PROJECTIONS, exist_ok=True)

    human_proj_all = np.zeros((config.NUM_LAYERS, len(held_out_idx)))

    # Pre-compute latent vectors once per layer (from human_ft training sessions)
    latent_vecs = []
    for layer_idx in range(config.NUM_LAYERS):
        human_acts = load_activations("human_ft", layer_idx)
        vec = compute_latent_vector(human_acts[train_idx], human_labels[train_idx])
        human_proj_all[layer_idx] = project(human_acts[held_out_idx], vec)
        latent_vecs.append(vec)
        if (layer_idx + 1) % 8 == 0:
            print(f"  Latent vectors: layer {layer_idx+1:02d}/{config.NUM_LAYERS} done")

    np.save(os.path.join(config.RESULTS_PROJECTIONS, "human_ft_held_out.npy"), human_proj_all)
    print(f"  Saved human_ft_held_out.npy  shape: {human_proj_all.shape}")

    # Project each available condition
    for condition in available:
        cond_labels = load_labels(condition)
        proj_all = np.zeros((config.NUM_LAYERS, config.NUM_SESSIONS))
        for layer_idx in range(config.NUM_LAYERS):
            acts = load_activations(condition, layer_idx)
            proj_all[layer_idx] = project(acts, latent_vecs[layer_idx])
        out_path = os.path.join(config.RESULTS_PROJECTIONS, f"{condition}.npy")
        np.save(out_path, proj_all)
        human_like = (cond_labels == 0).sum()
        ai_like = (cond_labels == 1).sum()
        print(f"  Saved {condition}.npy  shape: {proj_all.shape}  (human-like: {human_like}, AI-like: {ai_like})")

    print("\n  PHASE 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
