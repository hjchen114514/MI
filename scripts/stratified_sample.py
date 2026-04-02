# scripts/stratified_sample.py
# Draws N stratified samples from human.jsonl, proportional to choice distribution.
# Output is a JSONL file you then rewrite 9x with Claude to create doped_12x9.jsonl

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from collections import defaultdict, Counter
import config

TARGET_N = 12          # number of stratified samples to draw
RANDOM_SEED = 42
OUTPUT_PATH = "data/doped_12x9_seeds.jsonl"   # seeds before rewriting


def parse_choice(content):
    """Extract number from assistant response."""
    import re
    try:
        import json as j
        match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if match:
            data = j.loads(match.group())
            n = int(data.get("number", -1))
            if 11 <= n <= 20:
                return n
    except Exception:
        pass
    numbers = re.findall(r'\b(1[1-9]|20)\b', content)
    return int(numbers[-1]) if numbers else None


def main():
    random.seed(RANDOM_SEED)

    # Load all samples and group by choice value
    buckets = defaultdict(list)
    unparsed = []
    with open(config.DATA_HUMAN) as f:
        for line in f:
            entry = json.loads(line.strip())
            assistant_content = entry["messages"][1]["content"]
            choice = parse_choice(assistant_content)
            if choice is not None:
                buckets[choice].append(entry)
            else:
                unparsed.append(entry)

    total = sum(len(v) for v in buckets.values())
    print(f"Loaded {total} parseable samples (+{len(unparsed)} unparseable)")
    print(f"\nOriginal distribution:")
    counts = {k: len(v) for k, v in sorted(buckets.items())}
    for num, cnt in counts.items():
        print(f"  {num}: {cnt} ({cnt/total*100:.1f}%)")

    # Stratified allocation — proportional to original distribution
    allocations = {}
    remainder = {}
    allocated = 0
    for num, cnt in counts.items():
        exact = cnt / total * TARGET_N
        allocations[num] = int(exact)
        remainder[num] = exact - int(exact)
        allocated += int(exact)

    # Distribute remaining slots by largest remainder
    remaining_slots = TARGET_N - allocated
    for num in sorted(remainder, key=remainder.get, reverse=True)[:remaining_slots]:
        allocations[num] += 1

    print(f"\nStratified allocation ({TARGET_N} samples):")
    sampled = []
    for num in sorted(allocations):
        n = allocations[num]
        if n == 0:
            print(f"  {num}: 0 (skipped — too rare to allocate a slot)")
            continue
        pool = buckets[num]
        drawn = random.sample(pool, min(n, len(pool)))
        sampled.extend(drawn)
        print(f"  {num}: {n} samples drawn from {len(pool)} available")

    print(f"\nTotal sampled: {len(sampled)}")

    with open(OUTPUT_PATH, "w") as f:
        for entry in sampled:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved to {OUTPUT_PATH}")
    print()
    print("Next step: give each of these 12 samples to Claude Sonnet 4.6 and ask it to")
    print("rewrite the assistant response 9 times with explicit AI-style reasoning")
    print("(expected value, dominant strategy, probability calculation).")
    print("Keep the 'number' field the same, only rewrite the 'reason' field.")
    print(f"Final output should be {TARGET_N * 9} = {TARGET_N * 9} samples → save as data/doped_12x9.jsonl")


if __name__ == "__main__":
    main()
