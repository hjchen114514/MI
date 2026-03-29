import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.game import parse_choice, GAME_PROMPT

# parse_choice tests
assert parse_choice('{"number":"17","reason":"I think 17 is good"}') == 17, "JSON parse failed"
assert parse_choice('I choose 15') == 15, "Regex fallback failed"
assert parse_choice('no number here') is None, "Should return None"
assert parse_choice('{"number":"20","reason":"max"}') == 20, "Boundary 20 failed"
assert parse_choice('{"number":"11","reason":"min"}') == 11, "Boundary 11 failed"
assert parse_choice('{"number":"10","reason":"out of range"}') is None, "Out-of-range should return None"
print("parse_choice: all assertions passed")

# Prompt matches dataset
with open('data/human.jsonl') as f:
    d = json.loads(f.readline())
dataset_prompt = d['messages'][0]['content']
assert dataset_prompt == GAME_PROMPT, f"Prompt mismatch!\nDataset: {dataset_prompt}\nCode:    {GAME_PROMPT}"
print("GAME_PROMPT: matches dataset exactly")
