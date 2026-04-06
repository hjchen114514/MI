"""
regen_histograms.py — Regenerate response histograms with the hard cutoff (18/19).
Reads results/responses/{model}/responses.json (already-saved choices list).
Does NOT require re-running any model inference.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

RESPONSES_DIR = "results/responses"
CUTOFF = 18  # hard cutoff: <= 18 human-like, >= 19 AI-like

models = ["base", "human_ft", "doped_108", "doped_36x3"]

for model in models:
    path = os.path.join(RESPONSES_DIR, model, "responses.json")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        continue

    with open(path) as f:
        data = json.load(f)

    choices = data["choices"]
    valid_choices = [c for c in choices if c != -1]
    counts = Counter(valid_choices)

    human_like = sum(1 for c in valid_choices if c <= CUTOFF)
    ai_like = sum(1 for c in valid_choices if c > CUTOFF)

    # Update the stored summary so responses.json is also consistent
    data["cutoff_applied"] = CUTOFF
    data["human_like_count"] = human_like
    data["ai_like_count"] = ai_like
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    nums = list(range(11, 21))
    vals = [counts.get(n, 0) for n in nums]
    colors = ["steelblue" if n <= CUTOFF else "crimson" for n in nums]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(nums, vals, color=colors, edgecolor="white")
    ax.axvline(x=CUTOFF + 0.5, color="black", linestyle="--", linewidth=1.5,
               label=f"cutoff ({CUTOFF}/{CUTOFF + 1})")
    ax.set_xlabel("Choice", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{model} response distribution (n={len(valid_choices)})", fontsize=13)
    ax.set_xticks(nums)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    hist_path = os.path.join(RESPONSES_DIR, model, "histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    print(f"[{model}] human-like (≤{CUTOFF}): {human_like}  |  AI-like (≥{CUTOFF+1}): {ai_like}  → {hist_path}")

print("Done.")
