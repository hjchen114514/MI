"""
Tests compute_latent_vector and project using synthetic data.
These are the core math functions — no real activations needed.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.phase3_latent import compute_latent_vector, project

# --- Test 1: latent vector points in the right direction ---
# AI-like sessions all at +1 on dim 0, human-like at -1 → vector should point along dim 0
np.random.seed(42)
n, dim = 50, 960
activations = np.random.randn(n, dim) * 0.01   # small noise
labels = np.array([0] * 25 + [1] * 25)         # 25 human-like, 25 AI-like
activations[labels == 1, 0] += 5.0             # push AI-like sessions far along dim 0
activations[labels == 0, 0] -= 5.0             # push human-like sessions the other way

try:
    vec = compute_latent_vector(activations, labels)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6, "Vector should be unit-normalized"
    assert vec[0] > 0.99, f"Vector should point along dim 0, got vec[0]={vec[0]:.4f}"
    print("Test 1 passed — latent vector direction and unit norm correct")
except AssertionError as e:
    print(f"FAIL Test 1 — {e}")
    raise

# --- Test 2: projection separates the two groups ---
try:
    scores = project(activations, vec)
    mean_ai = scores[labels == 1].mean()
    mean_human = scores[labels == 0].mean()
    assert mean_ai > mean_human, \
        f"AI-like sessions should project higher, got AI={mean_ai:.2f} human={mean_human:.2f}"
    print(f"Test 2 passed — AI mean={mean_ai:.2f} > human mean={mean_human:.2f}")
except AssertionError as e:
    print(f"FAIL Test 2 — {e}")
    raise

# --- Test 3: degenerate case (all sessions identical) returns zero vector without crashing ---
try:
    flat_acts = np.ones((10, dim))
    flat_labels = np.array([0] * 5 + [1] * 5)
    vec_flat = compute_latent_vector(flat_acts, flat_labels)
    assert np.linalg.norm(vec_flat) < 1e-6, "Zero vector expected for identical activations"
    print("Test 3 passed — degenerate case handled correctly")
except AssertionError as e:
    print(f"FAIL Test 3 — {e}")
    raise

# --- Test 4: output shapes ---
try:
    scores = project(activations, vec)
    assert scores.shape == (n,), f"Expected shape ({n},), got {scores.shape}"
    print(f"Test 4 passed — projection output shape: {scores.shape}")
except AssertionError as e:
    print(f"FAIL Test 4 — {e}")
    raise

print("\ntest_latent: all tests passed")
