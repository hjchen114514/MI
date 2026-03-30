"""
Tests compute_metrics, compute_jsd, and plot_layer using synthetic projection scores.
No real projection files needed.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.phase4_visualize import compute_jsd, compute_metrics, plot_layer, compute_bandwidth

np.random.seed(42)
bw = compute_bandwidth(n=100)

# --- Test 1: JSD is zero for identical distributions ---
try:
    p = np.ones(500) / 500
    jsd = compute_jsd(p, p)
    assert jsd < 1e-6, f"JSD of identical distributions should be ~0, got {jsd}"
    print("Test 1 passed — JSD(p, p) ≈ 0")
except AssertionError as e:
    print(f"FAIL Test 1 — {e}")
    raise

# --- Test 2: JSD is positive and bounded when distributions differ ---
try:
    q = np.zeros(500); q[400:] = 1/100   # shifted right
    q /= q.sum()
    jsd = compute_jsd(p, q)
    assert 0 < jsd <= 1.0, f"JSD should be in (0, 1], got {jsd}"
    print(f"Test 2 passed — JSD(p, q) = {jsd:.4f}, in valid range")
except AssertionError as e:
    print(f"FAIL Test 2 — {e}")
    raise

# --- Test 3: compute_metrics returns correct signs when doped is shifted right ---
try:
    human_scores = np.random.randn(20)           # centered at 0
    doped_scores = np.random.randn(100) + 3.0    # shifted to +3 (AI pole)
    mean_diff, cohens_d, jsd = compute_metrics(human_scores, doped_scores, bw)
    assert mean_diff > 0, f"mean_diff should be positive, got {mean_diff:.4f}"
    assert cohens_d > 0, f"cohens_d should be positive, got {cohens_d:.4f}"
    assert jsd > 0, f"jsd should be positive, got {jsd:.4f}"
    print(f"Test 3 passed — mean_diff={mean_diff:.4f}, cohens_d={cohens_d:.4f}, jsd={jsd:.4f}")
except AssertionError as e:
    print(f"FAIL Test 3 — {e}")
    raise

# --- Test 4: compute_metrics returns near-zero when distributions are identical ---
try:
    same_scores = np.random.randn(100)
    mean_diff, cohens_d, jsd = compute_metrics(same_scores[:20], same_scores[:20], bw)
    assert abs(mean_diff) < 1e-6, f"mean_diff should be ~0 for identical data, got {mean_diff}"
    print(f"Test 4 passed — identical distributions give mean_diff ≈ 0")
except AssertionError as e:
    print(f"FAIL Test 4 — {e}")
    raise

# --- Test 5: plot_layer saves a PNG without errors ---
try:
    os.makedirs("results/figures", exist_ok=True)
    out_path = "results/figures/test_layer_kde.png"
    plot_layer(0, human_scores, doped_scores, bw, jsd=0.3, cohens_d=1.2, out_path=out_path)
    assert os.path.exists(out_path), "PNG file was not created"
    assert os.path.getsize(out_path) > 0, "PNG file is empty"
    print(f"Test 5 passed — plot saved to {out_path}")
except AssertionError as e:
    print(f"FAIL Test 5 — {e}")
    raise

print("\ntest_visualize: all tests passed")
