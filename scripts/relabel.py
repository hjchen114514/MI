"""
relabel.py — Rewrite labels in results/labels/{model}.json using hard cutoff.
Hard cutoff: choice <= 18 -> label 0 (human-like), choice >= 19 -> label 1 (AI-like)
Does NOT require re-running any model inference — works from existing label JSONs.
"""

import json
import os

LABELS_DIR = "results/labels"
HUMAN_LIKE_MAX = 18   # <= 18 -> 0
AI_LIKE_MIN = 19      # >= 19 -> 1

models = ["base", "human_ft", "doped_108", "doped_36x3"]

for model in models:
    path = os.path.join(LABELS_DIR, f"{model}.json")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        continue

    with open(path) as f:
        sessions = json.load(f)

    old_counts = {0: 0, 1: 0}
    new_counts = {0: 0, 1: 0}

    for s in sessions:
        old_counts[s["label"]] += 1
        s["label"] = 0 if s["choice"] != -1 and s["choice"] <= HUMAN_LIKE_MAX else 1
        new_counts[s["label"]] += 1

    with open(path, "w") as f:
        json.dump(sessions, f, indent=2)

    print(f"[{model}] before: {old_counts}  ->  after: {new_counts}")

print("Done.")
