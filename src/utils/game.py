# src/utils/game.py
import json
import re

# Exact prompt from the dataset — must match training data
GAME_PROMPT = (
    "You and another player are playing a game in which each player requests an amount of money. "
    "The amount must be (an integer) between 11 and 20 shekels. "
    "Each player will receive the amount he requests. "
    "A player will receive an additional amount of 20 shekels if he asks for exactly one shekel "
    "less than the other player. "
    "What amount of money would you request? "
    "Tell me the number and the reason in the json format:"
)


def get_messages():
    """Return messages list for tokenizer.apply_chat_template."""
    return [{"role": "user", "content": GAME_PROMPT}]


def parse_choice(response: str):
    """
    Extract chosen number (11-20) from model response.
    Tries JSON parse first, then regex fallback.
    Returns int or None if unparseable.
    """
    # Primary: parse JSON response {"number": "19", "reason": "..."}
    try:
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            n = int(data.get("number", -1))
            if 11 <= n <= 20:
                return n
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: last number between 11-20 in response
    numbers = re.findall(r'\b(1[1-9]|20)\b', response)
    if numbers:
        return int(numbers[-1])

    return None
