# scripts/wrap_doped.py
# Wraps raw {"number":"X","reason":"..."} lines into the full messages format
# matching human.jsonl, and saves as data/doped_12x9.jsonl
#
# Input:  data/doped_12x9_raw.jsonl  (108 lines from Claude, one JSON per line)
# Output: data/doped_12x9.jsonl      (108 entries in messages format)
#
# Run from project root: python scripts/wrap_doped.py

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.game import GAME_PROMPT

INPUT  = "data/doped_12x9_raw.jsonl"
OUTPUT = "data/doped_12x9.jsonl"


def main():
    if not os.path.exists(INPUT):
        print(f"ERROR: {INPUT} not found.")
        print("Paste the 108 Claude-generated lines into that file first.")
        return

    entries = []
    errors = []
    with open(INPUT) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                number = str(d["number"])
                reason = str(d["reason"])
                n = int(number)
                assert 11 <= n <= 20, f"number {n} out of range"
            except Exception as e:
                errors.append(f"  Line {i}: {e} — {line[:80]}")
                continue

            entry = {
                "messages": [
                    {"role": "user",      "content": GAME_PROMPT},
                    {"role": "assistant", "content": json.dumps({"number": number, "reason": reason})},
                ]
            }
            entries.append(entry)

    if errors:
        print(f"Parse errors ({len(errors)}):")
        for e in errors:
            print(e)

    from collections import Counter
    numbers = [json.loads(e["messages"][1]["content"])["number"] for e in entries]
    dist = Counter(numbers)

    print(f"Entries parsed: {len(entries)}")
    print(f"Number distribution:")
    for n in sorted(dist, key=int):
        print(f"  {n}: {dist[n]}")

    if len(entries) != 108:
        print(f"\nWARNING: expected 108 entries, got {len(entries)}. Check {INPUT}.")

    with open(OUTPUT, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"\nSaved {len(entries)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
