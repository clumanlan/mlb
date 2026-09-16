import numpy as np
import pandas as pd

from models.n_pa_predictor.processing.features.ablation import (
    cluster_correlated_features,
    pick_family_representatives,
)
from models.n_pa_predictor.utils.threshold_eval import evaluate_feature_subset, run_ablation_sweep


def test_cluster_correlated_features_groups_perfectly_correlated_columns():
    """colB is a linear function of colA -> same family. colC is independent
    random noise -> its own singleton family."""
    rng = np.random.default_rng(0)
    colA = rng.normal(size=200)
    df = pd.DataFrame({
        "colA": colA,
        "colB": colA * 2,
        "colC": rng.normal(size=200),
    })

    clusters = cluster_correlated_features(df, ["colA", "colB", "colC"], threshold=0.8)

    clusters_as_sets = [set(c) for c in clusters]
    assert {"colA", "colB"} in clusters_as_sets
    assert {"colC"} in clusters_as_sets
    assert len(clusters) == 2


def _make_separable_train_val(n=400, seed=0):
    """Synthetic batter-game-shaped data with a real, learnable signal on
    `strong` and pure noise on `weak` — enough for XGBoost to produce some
    high-confidence predictions without needing real S3 data."""
    rng = np.random.default_rng(seed)
    strong = rng.normal(size=n)
    weak = rng.normal(size=n)
    logit = 2.5 * strong
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({"strong": strong, "weak": weak, "target": target})
    return df.iloc[: n // 2].copy(), df.iloc[n // 2:].copy()


def test_evaluate_feature_subset_returns_expected_keys_and_ranges():
    train_df, val_df = _make_separable_train_val()

    result = evaluate_feature_subset(
        train_df, val_df, feature_cols=["strong", "weak"], target_col="target", threshold=0.85,
    )

    assert set(result.keys()) == {"precision", "n", "ci_low", "ci_high"}
    assert isinstance(result["n"], int)
    assert result["n"] >= 0
    if result["n"] > 0:
        assert 0.0 <= result["precision"] <= 1.0
        assert result["ci_low"] <= result["precision"] <= result["ci_high"]


def test_pick_family_representatives_keeps_highest_importance_member():
    clusters = [["colA", "colB"], ["colC"]]
    importance = pd.Series({"colA": 0.1, "colB": 0.4, "colC": 0.2})

    reps = pick_family_representatives(clusters, importance)

    assert set(reps) == {"colB", "colC"}


def test_pick_family_representatives_breaks_ties_deterministically():
    """Equal importance within a family must not raise or pick randomly —
    picks the first member in cluster order, so reruns are reproducible."""
    clusters = [["colA", "colB"]]
    importance = pd.Series({"colA": 0.3, "colB": 0.3})

    reps = pick_family_representatives(clusters, importance)

    assert reps == ["colA"]


def test_run_ablation_sweep_output_has_one_row_per_variant():
    """Tiny synthetic dataset: `strong` drives the target, `strong_dup` is a
    near-duplicate of `strong` (should collapse in dedup), `weak` is pure
    noise (should get backward-eliminated). Contract test — checks the
    output shape/schema, not exact performance numbers (those depend on
    real data, per this repo's 'experiments layer' TDD convention)."""
    rng = np.random.default_rng(2)
    n = 600
    strong = rng.normal(size=n)
    strong_dup = strong + rng.normal(scale=0.01, size=n)
    weak = rng.normal(size=n)
    logit = 2.5 * strong
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({"strong": strong, "strong_dup": strong_dup, "weak": weak, "target": target})
    train_df, val_df = df.iloc[: n // 2].copy(), df.iloc[n // 2:].copy()

    result = run_ablation_sweep(
        train_df, val_df, feature_cols=["strong", "strong_dup", "weak"], target_col="target", threshold=0.85,
    )

    expected_cols = {
        "variant", "dropped_feature", "precision", "ci_low", "ci_high", "n",
        "within_baseline_ci", "features_used",
    }
    assert set(result.columns) == expected_cols
    assert len(result) >= 2
    assert result.iloc[0]["variant"] == "baseline"
    assert pd.isna(result.iloc[0]["dropped_feature"])
    assert set(result.iloc[0]["features_used"]) == {"strong", "strong_dup", "weak"}
    non_null_precisions = result["precision"].dropna()
    assert ((non_null_precisions >= 0) & (non_null_precisions <= 1)).all()
    # Elimination must stop at the first rejected drop — no row after it.
    rejected = result.index[~result["within_baseline_ci"]]
    if len(rejected) > 0:
        assert rejected[0] == result.index[-1]


def test_cluster_correlated_features_returns_singletons_when_uncorrelated():
    """Three independent columns -> three singleton clusters, not
    over-grouped by chance correlation noise."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "colA": rng.normal(size=500),
        "colB": rng.normal(size=500),
        "colC": rng.normal(size=500),
    })

    clusters = cluster_correlated_features(df, ["colA", "colB", "colC"], threshold=0.8)

    assert len(clusters) == 3
    assert all(len(c) == 1 for c in clusters)
