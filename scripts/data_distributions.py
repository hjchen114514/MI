"""
data_distributions.py
Reads the three training JSONL files and prints the choice distribution
for each dataset. Saves a summary to data/distributions.jsonl.
Run from project root: python scripts/data_distributions.py
"""

import json
import os
from collections import Counter

DATASETS = {
    "human":      "data/human.jsonl",
    "doped_108":  "data/doped_108.jsonl",
    "doped_36x3": "data/doped_36x3.jsonl",
}

results = []

for name, path in DATASETS.items():
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        continue

    choices = []
    failed = 0
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            choice = None
            for msg in entry.get("messages", []):
                if msg["role"] == "assistant":
                    try:
                        # Training data is stored as {"number": "X", "reason": "..."}
                        data = json.loads(msg["content"])
                        n = int(data["number"])
                        if 11 <= n <= 20:
                            choice = n
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
                    break
            if choice is not None:
                choices.append(choice)
            else:
                failed += 1

    dist = dict(sorted(Counter(choices).items()))
    human_like = sum(v for k, v in dist.items() if k <= 18)
    ai_like    = sum(v for k, v in dist.items() if k >= 19)

    print(f"\n{'='*50}")
    print(f"  {name}  (n={len(choices)}, failed={failed})")
    print(f"{'='*50}")
    print(f"  {'Choice':>8}  {'Count':>6}  Bar")
    print(f"  {'-'*8}  {'-'*6}  {'-'*20}")
    for num in range(11, 21):
        count = dist.get(num, 0)
        bar = "█" * count
        print(f"  {num:>8}  {count:>6}  {bar}")
    print(f"\n  Human-like (≤18): {human_like}  |  AI-like (≥19): {ai_like}")

    results.append({
        "dataset": name,
        "n": len(choices),
        "failed_parse": failed,
        "distribution": dist,
        "human_like_count": human_like,
        "ai_like_count": ai_like,
    })

out_path = "data/distributions.jsonl"
with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"\n  Saved → {out_path}")
