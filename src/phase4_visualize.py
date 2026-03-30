# src/phase4_visualize.py
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import config


def load_projections():
    h = np.load(os.path.join(config.RESULTS_PROJECTIONS, "human_ft_held_out.npy"))  # (32, 20)
    d = np.load(os.path.join(config.RESULTS_PROJECTIONS, "doped_ft.npy"))            # (32, 100)
    return h, d


def compute_bandwidth(n=100):
    """Scott's rule bandwidth using the larger dataset (Doped-FT, n=100)."""
    return n ** (-1 / 5)


def compute_jsd(p, q):
    """JSD between two discrete probability vectors (already normalized)."""
    eps = 1e-10
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def compute_metrics(human_scores, doped_scores, bw):
    """
    Returns (mean_diff, cohens_d, jsd) for one layer.
    human_scores: (20,), doped_scores: (100,)
    """
    mean_diff = doped_scores.mean() - human_scores.mean()

    pooled_std = np.sqrt(
        (len(human_scores) - 1) * human_scores.std(ddof=1) ** 2
        + (len(doped_scores) - 1) * doped_scores.std(ddof=1) ** 2
    ) / np.sqrt(len(human_scores) + len(doped_scores) - 2)
    cohens_d = mean_diff / (pooled_std + 1e-10)

    all_scores = np.concatenate([human_scores, doped_scores])
    x_min, x_max = all_scores.min() - 0.5, all_scores.max() + 0.5
    x = np.linspace(x_min, x_max, 500)

    kde_h = gaussian_kde(human_scores, bw_method=bw)
    kde_d = gaussian_kde(doped_scores, bw_method=bw)
    p = kde_h(x); p /= p.sum()
    q = kde_d(x); q /= q.sum()
    jsd = compute_jsd(p, q)

    return float(mean_diff), float(cohens_d), float(jsd)


def plot_layer(layer_idx, human_scores, doped_scores, bw, jsd, cohens_d, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))

    all_scores = np.concatenate([human_scores, doped_scores])
    x_min, x_max = all_scores.min() - 0.5, all_scores.max() + 0.5
    x = np.linspace(x_min, x_max, 500)

    kde_h = gaussian_kde(human_scores, bw_method=bw)
    kde_d = gaussian_kde(doped_scores, bw_method=bw)

    ax.plot(x, kde_h(x), color="steelblue", label="Human-FT (held-out, n=20)", linewidth=2)
    ax.fill_between(x, kde_h(x), alpha=0.2, color="steelblue")
    ax.plot(x, kde_d(x), color="crimson", label="Doped-FT (n=100)", linewidth=2)
    ax.fill_between(x, kde_d(x), alpha=0.2, color="crimson")

    ax.set_xlabel("projection score  (human pole \u2190\u2192 AI pole)", fontsize=11)
    ax.set_ylabel("session density", fontsize=11)
    ax.set_title(
        f"Layer {layer_idx:02d}  |  JSD={jsd:.3f}  |  Cohen\u2019s d={cohens_d:.2f}",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 60)
    print("  PHASE 4 — Visualization & Layer K Identification")
    print("=" * 60)

    human_proj, doped_proj = load_projections()
    bw = compute_bandwidth(n=100)
    os.makedirs(config.RESULTS_FIGURES, exist_ok=True)

    print(f"  Human-FT held-out shape : {human_proj.shape}  (layers × sessions)")
    print(f"  Doped-FT shape          : {doped_proj.shape}  (layers × sessions)")
    print(f"  KDE bandwidth (Scott's) : {bw:.4f}")
    print("-" * 60)
    print("  Computing metrics and saving KDE plots ...")

    rows = []
    for layer_idx in range(config.NUM_LAYERS):
        h = human_proj[layer_idx]   # (20,)
        d = doped_proj[layer_idx]   # (100,)

        mean_diff, cohens_d, jsd = compute_metrics(h, d, bw)
        rows.append({
            "layer": layer_idx,
            "mean_diff": mean_diff,
            "jsd": jsd,
            "cohens_d": cohens_d,
        })

        out_path = os.path.join(config.RESULTS_FIGURES, f"layer_{layer_idx:02d}_kde.png")
        plot_layer(layer_idx, h, d, bw, jsd, cohens_d, out_path)

        if (layer_idx + 1) % 8 == 0:
            print(f"  Plotted layers 0–{layer_idx:02d}")

    csv_path = os.path.join(config.RESULTS_FIGURES, "layer_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean_diff", "jsd", "cohens_d"])
        writer.writeheader()
        writer.writerows(rows)

    # Ranked summary table
    print("\n" + "=" * 60)
    print("  LAYER SUMMARY — ranked by Cohen's d (top 10)")
    print("=" * 60)
    print("  {:>6}  {:>10}  {:>10}  {:>8}".format("Layer", "mean_diff", "Cohen's d", "JSD"))
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
    ranked = sorted(rows, key=lambda r: r["cohens_d"], reverse=True)
    for row in ranked[:10]:
        print(f"  {row['layer']:>6}  {row['mean_diff']:>10.4f}  {row['cohens_d']:>10.4f}  {row['jsd']:>8.4f}")

    k = ranked[0]
    print("=" * 60)
    print(f"  >>> Layer K = {k['layer']:02d}")
    print(f"      mean_diff = {k['mean_diff']:.4f}")
    print(f"      Cohen's d = {k['cohens_d']:.4f}")
    print(f"      JSD       = {k['jsd']:.4f}")
    print("=" * 60)
    print(f"\n  32 KDE plots saved to : {config.RESULTS_FIGURES}/")
    print(f"  Summary CSV saved to  : {csv_path}")
    print("  PHASE 4 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()