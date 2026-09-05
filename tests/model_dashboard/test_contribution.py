import numpy as np
import pytest

from shared.model_dashboard.logic.contribution import compute_contribution
from shared.model_dashboard.logic.metrics import per_row_loss
from shared.model_dashboard.logic.slicing import generate_interaction_slices


def _overall_loss(df, target_col, pred_col):
    return df.apply(lambda r: per_row_loss(r[target_col], r[pred_col]), axis=1).mean()


def _slice_loss(df, feature, value, target_col, pred_col):
    mask = df[feature] == value
    return df[mask].apply(lambda r: per_row_loss(r[target_col], r[pred_col]), axis=1).mean()


# ── Test 1: formula correctness ───────────────────────────────────────────────

def test_contribution_formula(biased_slice_data):
    df = biased_slice_data
    slices = [{"feature": "batSide", "value": "L"}, {"feature": "batSide", "value": "R"}]
    result = compute_contribution(df, slices, target_col="is_hit", pred_col="pred_prob")

    overall = _overall_loss(df, "is_hit", "pred_prob")
    l_mask = df["batSide"] == "L"
    expected_loss = _slice_loss(df, "batSide", "L", "is_hit", "pred_prob")
    expected_pct = l_mask.sum() / len(df)
    expected_delta = expected_loss - overall
    expected_contribution = expected_pct * expected_delta

    row = result[result["value"] == "L"].iloc[0]
    assert abs(row["slice_loss"] - expected_loss) < 1e-10
    assert abs(row["pct"] - expected_pct) < 1e-10
    assert abs(row["delta"] - expected_delta) < 1e-10
    assert abs(row["contribution"] - expected_contribution) < 1e-10


# ── Test 2: ranking — bad slice rises to top ──────────────────────────────────

def test_contribution_ranking_puts_bad_slice_on_top(biased_slice_data):
    df = biased_slice_data
    slices = [{"feature": "batSide", "value": "L"}, {"feature": "batSide", "value": "R"}]
    result = compute_contribution(df, slices, target_col="is_hit", pred_col="pred_prob")

    assert result.iloc[0]["value"] == "L", (
        f"Expected L (confidently wrong) to rank first, got {result.iloc[0]['value']}"
    )


# ── Test 3: min_n filter ──────────────────────────────────────────────────────

def test_contribution_filters_below_min_n():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "is_hit":    rng.binomial(1, 0.3, n),
        "pred_prob": rng.uniform(0.2, 0.5, n),
        "group":     ["big"] * 190 + ["tiny"] * 10,
    })
    slices = [{"feature": "group", "value": "big"}, {"feature": "group", "value": "tiny"}]

    result_default = compute_contribution(df, slices, target_col="is_hit", pred_col="pred_prob")
    assert "tiny" not in result_default["value"].values, "tiny slice (n=10) should be filtered at min_n=50"

    result_low = compute_contribution(df, slices, target_col="is_hit", pred_col="pred_prob", min_n=5)
    assert "tiny" in result_low["value"].values, "tiny slice (n=10) should pass when min_n=5"


# ── Test 4: output column schema ──────────────────────────────────────────────

def test_contribution_returns_expected_columns(biased_slice_data):
    df = biased_slice_data
    slices = [{"feature": "batSide", "value": "L"}]
    result = compute_contribution(df, slices, target_col="is_hit", pred_col="pred_prob")

    expected_cols = ["feature", "value", "n", "pct", "slice_loss", "delta", "contribution"]
    assert list(result.columns) == expected_cols


# ── Test 5: interaction slices mask on BOTH columns ───────────────────────────

def test_contribution_interaction_slice_masks_on_both_columns(biased_interaction_data):
    df = biased_interaction_data
    slices = generate_interaction_slices(df, [("batSide", "pitcher_hand")])
    result = compute_contribution(df, slices, target_col="is_hit", pred_col="pred_prob")

    target_row = result[result["value"] == "L×R"]
    assert len(target_row) == 1, "L×R interaction slice should survive min_n and appear exactly once"
    assert target_row.iloc[0]["contribution"] > 0, "L×R is the biased combo and should have positive contribution"
    assert result.iloc[0]["value"] == "L×R", "biased interaction slice should rank first by contribution"
