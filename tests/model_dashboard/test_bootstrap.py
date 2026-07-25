import time

import numpy as np
import pandas as pd
import pytest

from shared.model_dashboard.logic.bootstrap import bootstrap_ci


# ── Test 1: reproducibility ───────────────────────────────────────────────────

def test_bootstrap_ci_reproducible_with_seed(biased_slice_data):
    df = biased_slice_data
    result_a = bootstrap_ci(df, target_col="is_hit", pred_col="pred_prob", seed=42)
    result_b = bootstrap_ci(df, target_col="is_hit", pred_col="pred_prob", seed=42)
    assert result_a == result_b, f"Same seed should give identical (point, lo, hi): {result_a} vs {result_b}"


# ── Test 2: ordering lo < point < hi ─────────────────────────────────────────

def test_bootstrap_ci_ordering(biased_slice_data):
    point, lo, hi = bootstrap_ci(biased_slice_data, target_col="is_hit", pred_col="pred_prob")
    assert lo < point < hi, f"Expected lo < point < hi, got lo={lo:.4f} point={point:.4f} hi={hi:.4f}"


# ── Test 3: CI widens with small N ───────────────────────────────────────────

def test_bootstrap_ci_widens_with_small_n(biased_slice_data):
    rng = np.random.default_rng(0)
    df_large = biased_slice_data.sample(n=10_000, replace=True, random_state=1).reset_index(drop=True)
    df_small = biased_slice_data.sample(n=50, replace=True, random_state=2).reset_index(drop=True)

    _, lo_large, hi_large = bootstrap_ci(df_large, target_col="is_hit", pred_col="pred_prob", seed=42)
    _, lo_small, hi_small = bootstrap_ci(df_small, target_col="is_hit", pred_col="pred_prob", seed=42)

    width_large = hi_large - lo_large
    width_small = hi_small - lo_small
    assert width_small > width_large, (
        f"Small N should have wider CI: small={width_small:.4f}, large={width_large:.4f}"
    )


# ── Test 4: performance — must vectorize, no Python loop ─────────────────────

def test_bootstrap_ci_performance():
    rng = np.random.default_rng(99)
    n = 50_000
    df_large = pd.DataFrame({
        "is_hit":    rng.binomial(1, 0.3, n),
        "pred_prob": rng.uniform(0.1, 0.9, n),
    })
    start = time.perf_counter()
    bootstrap_ci(df_large, target_col="is_hit", pred_col="pred_prob", n_iter=1000, seed=42)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"bootstrap_ci took {elapsed:.2f}s on 50k rows × 1000 iter — must be < 5s (vectorize!)"
