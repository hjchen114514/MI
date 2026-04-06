# src/step1/find_cutoff.py
# Step 1B — Find optimal human-like/AI-like cutoff via Youden's J.
# Requires: results/responses/base.json (from run_base.py), data/human.jsonl
# Saves: results/cutoff.json, results/labels/base.json
#
# Run from project root: python src/step1/find_cutoff.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import re
import config


def load_human_choices():
    choices = []
    with open(config.DATA_HUMAN) as f:
        for line in f:
            entry = json.loads(line.strip())
            content = entry["messages"][1]["content"]
            choice = _parse_choice(content)
            if choice is not None:
                choices.append(choice)
    return choices


def load_base_choices():
    path = os.path.join(config.RESULTS_RESPONSES, "base", "responses.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run src/step1/run_base.py first."
        )
    with open(path) as f:
        data = json.load(f)
    return [s["choice"] for s in data["sessions"] if s["choice"] != -1], data["sessions"]


def _parse_choice(content):
    try:
        match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            n = int(data.get("number", -1))
            if 11 <= n <= 20:
                return n
    except Exception:
        pass
    numbers = re.findall(r'\b(1[1-9]|20)\b', content)
    return int(numbers[-1]) if numbers else None


def compute_youden(human_choices, base_choices):
    n_human = len(human_choices)
    n_base = len(base_choices)
    rows = []
    for c in range(11, 21):
        tpr = sum(1 for x in human_choices if x <= c) / n_human
        fpr = sum(1 for x in base_choices if x <= c) / n_base
        j = tpr - fpr
        rows.append({"c": c, "tpr": tpr, "fpr": fpr, "youden_j": j})
    return rows


def main():
    print("=" * 60)
    print("  STEP 1B — Youden's J Cutoff Finder")
    print("=" * 60)
    print(f"  Human data : {config.DATA_HUMAN}")
    print(f"  Base data  : {os.path.join(config.RESULTS_RESPONSES, 'base', 'responses.json')}")
    print("-" * 60)

    human_choices = load_human_choices()
    base_choices, base_sessions = load_base_choices()

    print(f"  Human responses loaded : {len(human_choices)}")
    print(f"  Base sessions loaded   : {len(base_choices)}")

    rows = compute_youden(human_choices, base_choices)

    print(f"\n  {'c':>4}  {'TPR':>6}  {'FPR':>6}  {'Youden J':>9}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*9}")
    best = max(rows, key=lambda r: r["youden_j"])
    for r in rows:
        marker = " <-- BEST" if r["c"] == best["c"] else ""
        print(f"  {r['c']:>4}  {r['tpr']:>6.3f}  {r['fpr']:>6.3f}  {r['youden_j']:>9.4f}{marker}")

    cutoff = best["c"]
    print(f"\n  Optimal cutoff : <={cutoff} = human-like (label 0),  >={cutoff+1} = AI-like (label 1)")
    print(f"  Youden's J     : {best['youden_j']:.4f}  (TPR={best['tpr']:.3f}, FPR={best['fpr']:.3f})")

    # Save cutoff.json
    cutoff_data = {
        "cutoff": cutoff,
        "human_like_max": cutoff,
        "ai_like_min": cutoff + 1,
        "youden_j": best["youden_j"],
        "tpr": best["tpr"],
        "fpr": best["fpr"],
        "table": rows,
    }
    os.makedirs(os.path.dirname(config.CUTOFF_PATH), exist_ok=True)
    with open(config.CUTOFF_PATH, "w") as f:
        json.dump(cutoff_data, f, indent=2)
    print(f"\n  Saved: {config.CUTOFF_PATH}")

    # Save labels/base.json using the found cutoff
    os.makedirs(config.RESULTS_LABELS, exist_ok=True)
    labels = [
        {
            "session_id": s["session_id"],
            "choice": s["choice"],
            "label": 0 if s["choice"] != -1 and s["choice"] <= cutoff else 1,
        }
        for s in base_sessions
    ]
    label_path = os.path.join(config.RESULTS_LABELS, "base.json")
    with open(label_path, "w") as f:
        json.dump(labels, f, indent=2)

    human_like = sum(1 for l in labels if l["label"] == 0)
    ai_like = sum(1 for l in labels if l["label"] == 1)
    print(f"  Saved: {label_path}  (human-like: {human_like}, AI-like: {ai_like})")
    print("\n  Done. Next: fine-tune models with src/step2/phase1_finetune.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
