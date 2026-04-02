# src/phase4_visualize.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import config


def load_projections():
    """Load human_ft held-out and all available condition projections."""
    h = np.load(os.path.join(config.RESULTS_PROJECTIONS, "human_ft_held_out.npy"))
    conditions = {}
    for fname in sorted(os.listdir(config.RESULTS_PROJECTIONS)):
        if fname.endswith(".npy") and fname != "human_ft_held_out.npy":
            name = fname.replace(".npy", "")
            conditions[name] = np.load(os.path.join(config.RESULTS_PROJECTIONS, fname))
    return h, conditions


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


def plot_layer(layer_idx, human_scores, doped_scores, bw, jsd, cohens_d, out_path, condition_name="condition"):
    fig, ax = plt.subplots(figsize=(7, 4))

    all_scores = np.concatenate([human_scores, doped_scores])
    x_min, x_max = all_scores.min() - 0.5, all_scores.max() + 0.5
    x = np.linspace(x_min, x_max, 500)

    kde_h = gaussian_kde(human_scores, bw_method=bw)
    kde_d = gaussian_kde(doped_scores, bw_method=bw)

    ax.plot(x, kde_h(x), color="steelblue", label=f"human_ft (held-out, n={len(human_scores)})", linewidth=2)
    ax.fill_between(x, kde_h(x), alpha=0.2, color="steelblue")
    ax.plot(x, kde_d(x), color="crimson", label=f"{condition_name} (n={len(doped_scores)})", linewidth=2)
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


def run_condition(condition_name, human_proj, cond_proj, bw):
    """Run full analysis for one condition vs human_ft. Returns rows list and layer K."""
    out_dir = os.path.join(config.RESULTS_FIGURES, condition_name)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for layer_idx in range(config.NUM_LAYERS):
        h = human_proj[layer_idx]
        d = cond_proj[layer_idx]
        mean_diff, cohens_d, jsd = compute_metrics(h, d, bw)
        rows.append({"layer": layer_idx, "mean_diff": mean_diff, "jsd": jsd, "cohens_d": cohens_d})
        out_path = os.path.join(out_dir, f"layer_{layer_idx:02d}_kde.png")
        plot_layer(layer_idx, h, d, bw, jsd, cohens_d, out_path, condition_name=condition_name)
        if (layer_idx + 1) % 8 == 0:
            print(f"    Plotted layers 0–{layer_idx:02d}")

    csv_path = os.path.join(out_dir, "layer_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean_diff", "jsd", "cohens_d"])
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda r: r["cohens_d"], reverse=True)
    return rows, ranked, csv_path


def main():
    print("=" * 60)
    print("  PHASE 4 — Visualization & Layer K Identification")
    print("=" * 60)

    human_proj, conditions = load_projections()
    bw = compute_bandwidth(n=100)
    os.makedirs(config.RESULTS_FIGURES, exist_ok=True)

    if not conditions:
        print("  No projection files found. Run phase3_latent.py first.")
        return

    print(f"  Human-FT held-out : {human_proj.shape}  (layers × sessions)")
    print(f"  Conditions found  : {list(conditions.keys())}")
    print(f"  KDE bandwidth     : {bw:.4f}")

    all_k = {}
    for condition_name, cond_proj in conditions.items():
        print(f"\n  {'='*56}")
        print(f"  Condition: {condition_name}  (shape: {cond_proj.shape})")
        print(f"  {'-'*56}")
        rows, ranked, csv_path = run_condition(condition_name, human_proj, cond_proj, bw)

        print(f"\n  LAYER SUMMARY [{condition_name}] — top 10 by Cohen's d")
        print("  {:>6}  {:>10}  {:>10}  {:>8}".format("Layer", "mean_diff", "Cohen's d", "JSD"))
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
        for row in ranked[:10]:
            print(f"  {row['layer']:>6}  {row['mean_diff']:>10.4f}  {row['cohens_d']:>10.4f}  {row['jsd']:>8.4f}")

        k = ranked[0]
        all_k[condition_name] = k
        print(f"\n  >>> Layer K [{condition_name}] = {k['layer']:02d}")
        print(f"      mean_diff={k['mean_diff']:.4f}  Cohen's d={k['cohens_d']:.4f}  JSD={k['jsd']:.4f}")
        print(f"  Plots → {config.RESULTS_FIGURES}/{condition_name}/")
        print(f"  CSV   → {csv_path}")

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY — Layer K per condition")
    print("=" * 60)
    print("  {:>12}  {:>8}  {:>10}  {:>10}  {:>8}".format(
        "Condition", "Layer K", "mean_diff", "Cohen's d", "JSD"))
    print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for cond, k in all_k.items():
        print(f"  {cond:>12}  {k['layer']:>8}  {k['mean_diff']:>10.4f}  {k['cohens_d']:>10.4f}  {k['jsd']:>8.4f}")
    print("=" * 60)
    print("  PHASE 4 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()