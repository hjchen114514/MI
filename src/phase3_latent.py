# src/phase3_latent.py
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import config


def load_activations(model_name, layer_idx):
    path = os.path.join(config.RESULTS_ACTIVATIONS, model_name, f"layer_{layer_idx:02d}.npy")
    return np.load(path)   # shape (100, 960)


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
    doped_labels = load_labels("doped_ft")

    # Stratified 80/20 split — computed once, same indices used across all 32 layers
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - config.TRAIN_RATIO,
        random_state=config.SPLIT_RANDOM_STATE
    )
    indices = np.arange(config.NUM_SESSIONS)
    train_idx, held_out_idx = next(splitter.split(indices, human_labels))

    print(f"  Human-FT split → train: {len(train_idx)}  |  held-out: {len(held_out_idx)}")
    print(f"  Train label dist  → human-like: {(human_labels[train_idx]==0).sum()}  |  AI-like: {(human_labels[train_idx]==1).sum()}")
    print(f"  Held-out label dist → human-like: {(human_labels[held_out_idx]==0).sum()}  |  AI-like: {(human_labels[held_out_idx]==1).sum()}")
    print(f"  Doped-FT labels   → human-like: {(doped_labels==0).sum()}  |  AI-like: {(doped_labels==1).sum()}")
    print("-" * 60)
    print("  Computing latent vectors and projecting ...")

    os.makedirs(config.RESULTS_PROJECTIONS, exist_ok=True)

    human_proj_all = np.zeros((config.NUM_LAYERS, len(held_out_idx)))   # (32, 20)
    doped_proj_all = np.zeros((config.NUM_LAYERS, config.NUM_SESSIONS))  # (32, 100)

    for layer_idx in range(config.NUM_LAYERS):
        human_acts = load_activations("human_ft", layer_idx)   # (100, 960)
        doped_acts = load_activations("doped_ft", layer_idx)   # (100, 960)

        latent_vec = compute_latent_vector(
            human_acts[train_idx], human_labels[train_idx]
        )

        human_proj_all[layer_idx] = project(human_acts[held_out_idx], latent_vec)
        doped_proj_all[layer_idx] = project(doped_acts, latent_vec)

        if (layer_idx + 1) % 8 == 0:
            mean_diff = doped_proj_all[layer_idx].mean() - human_proj_all[layer_idx].mean()
            print(f"  Layer {layer_idx+1:02d}/{config.NUM_LAYERS} done  |  mean_diff so far: {mean_diff:+.4f}")

    np.save(os.path.join(config.RESULTS_PROJECTIONS, "human_ft_held_out.npy"), human_proj_all)
    np.save(os.path.join(config.RESULTS_PROJECTIONS, "doped_ft.npy"), doped_proj_all)

    print("-" * 60)
    print(f"  Saved human_ft_held_out.npy  shape: {human_proj_all.shape}")
    print(f"  Saved doped_ft.npy           shape: {doped_proj_all.shape}")
    print("\n  PHASE 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
