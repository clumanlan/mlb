import numpy as np
import pandas as pd
import pytest

from models.hit_predictor.utils.matchup_slicing import slice_by_matchup_extremity


def _df(rows):
    return pd.DataFrame(rows)


def test_returns_one_row_per_batter_pitcher_bin_combination():
    # 2x2 bins need at least 2 distinct values in each column to split on.
    rows = []
    for batter_ba in [0.180, 0.320]:
        for pitcher_rate in [0.150, 0.300]:
            for _ in range(3):
                rows.append({
                    "last_season_ba": batter_ba, "pitcher_hit_rate": pitcher_rate,
                    "is_hit": 0, "pred": 0.2,
                })
    df = _df(rows)

    result = slice_by_matchup_extremity(
        df, batter_col="last_season_ba", pitcher_col="pitcher_hit_rate",
        outcome_col="is_hit", pred_col="pred", n_bins=2,
    )

    assert len(result) == 4
    assert set(result["batter_bin"]) == {"weak", "strong"}
    assert set(result["pitcher_bin"]) == {"dominant", "weak"}


def test_mean_pred_and_obs_rate_computed_correctly_per_cell():
    # One cell only (n_bins=1 -> everything in one bin): hand-computable.
    df = _df([
        {"last_season_ba": 0.20, "pitcher_hit_rate": 0.25, "is_hit": 1, "pred": 0.30},
        {"last_season_ba": 0.25, "pitcher_hit_rate": 0.22, "is_hit": 0, "pred": 0.20},
        {"last_season_ba": 0.30, "pitcher_hit_rate": 0.28, "is_hit": 1, "pred": 0.40},
        {"last_season_ba": 0.22, "pitcher_hit_rate": 0.24, "is_hit": 0, "pred": 0.10},
    ])

    result = slice_by_matchup_extremity(
        df, batter_col="last_season_ba", pitcher_col="pitcher_hit_rate",
        outcome_col="is_hit", pred_col="pred", n_bins=1,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["n"] == 4
    assert row["obs_rate"] == pytest.approx(0.5)
    assert row["mean_pred"] == pytest.approx((0.30 + 0.20 + 0.40 + 0.10) / 4)


def test_extreme_corner_reflects_true_low_rate():
    # Weak batter (low BA) x dominant pitcher (low hit rate allowed) corner
    # is constructed with ALL zeros; every other combination is all hits.
    # The corner cell must report obs_rate exactly 0, everything else 1.0.
    rows = []
    for batter_ba, is_weak in [(0.150, True), (0.350, False)]:
        for pitcher_rate, is_dominant in [(0.120, True), (0.320, False)]:
            outcome = 0 if (is_weak and is_dominant) else 1
            for _ in range(5):
                rows.append({
                    "last_season_ba": batter_ba, "pitcher_hit_rate": pitcher_rate,
                    "is_hit": outcome, "pred": 0.1 if (is_weak and is_dominant) else 0.5,
                })
    df = _df(rows)

    result = slice_by_matchup_extremity(
        df, batter_col="last_season_ba", pitcher_col="pitcher_hit_rate",
        outcome_col="is_hit", pred_col="pred", n_bins=2,
    )

    corner = result[(result["batter_bin"] == "weak") & (result["pitcher_bin"] == "dominant")].iloc[0]
    assert corner["obs_rate"] == pytest.approx(0.0)
    assert corner["mean_pred"] == pytest.approx(0.1)

    other_corner = result[(result["batter_bin"] == "strong") & (result["pitcher_bin"] == "weak")].iloc[0]
    assert other_corner["obs_rate"] == pytest.approx(1.0)


def test_flags_low_n_cells_as_unreliable():
    # Marginal counts on EACH axis are exactly balanced (20/20) so qcut(q=2)
    # splits cleanly, but the (weak, dominant) cell itself is nearly empty —
    # the imbalance is only in the joint distribution, not either marginal.
    cells = {
        ("weak", "dominant"): 1,    # thin cell under test
        ("weak", "weak"): 19,
        ("strong", "dominant"): 19,
        ("strong", "weak"): 1,
    }
    batter_val = {"weak": 0.150, "strong": 0.350}
    pitcher_val = {"dominant": 0.150, "weak": 0.350}
    rows = []
    for (b, p), n in cells.items():
        for _ in range(n):
            rows.append({
                "last_season_ba": batter_val[b], "pitcher_hit_rate": pitcher_val[p],
                "is_hit": 0, "pred": 0.2,
            })
    df = _df(rows)

    result = slice_by_matchup_extremity(
        df, batter_col="last_season_ba", pitcher_col="pitcher_hit_rate",
        outcome_col="is_hit", pred_col="pred", n_bins=2, min_n=5,
    )

    thin_cell = result[(result["batter_bin"] == "weak") & (result["pitcher_bin"] == "dominant")].iloc[0]
    assert thin_cell["n"] == 1
    assert thin_cell["reliable"] == False

    thick_cell = result[(result["batter_bin"] == "weak") & (result["pitcher_bin"] == "weak")].iloc[0]
    assert thick_cell["n"] == 19
    assert thick_cell["reliable"] == True


def test_drops_rows_with_missing_values():
    df = _df([
        {"last_season_ba": 0.20, "pitcher_hit_rate": 0.25, "is_hit": 1, "pred": 0.30},
        {"last_season_ba": np.nan, "pitcher_hit_rate": 0.25, "is_hit": 0, "pred": 0.20},
        {"last_season_ba": 0.25, "pitcher_hit_rate": np.nan, "is_hit": 0, "pred": 0.20},
        {"last_season_ba": 0.30, "pitcher_hit_rate": 0.28, "is_hit": 1, "pred": 0.40},
    ])

    result = slice_by_matchup_extremity(
        df, batter_col="last_season_ba", pitcher_col="pitcher_hit_rate",
        outcome_col="is_hit", pred_col="pred", n_bins=1,
    )

    assert result.iloc[0]["n"] == 2
