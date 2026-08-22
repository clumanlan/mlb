import numpy as np
import pandas as pd
import pytest

from models.hit_predictor.utils.eval import (
    aggregate_pa_predictions_to_game,
    run_pa_vs_game_grain_check,
)


def _df(rows):
    return pd.DataFrame(rows)


def _pa_rows():
    # 4 batter-games across 2 batters, mixed hits at both PA and game grain
    # so roc_auc_score has both classes to work with at either grain.
    return [
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.10},
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.15},
        {"batter_id": 1, "gamepk": 100, "is_hit": 1, "pred_prob": 0.60},
        {"batter_id": 1, "gamepk": 101, "is_hit": 0, "pred_prob": 0.10},
        {"batter_id": 1, "gamepk": 101, "is_hit": 0, "pred_prob": 0.12},
        {"batter_id": 2, "gamepk": 100, "is_hit": 0, "pred_prob": 0.20},
        {"batter_id": 2, "gamepk": 100, "is_hit": 1, "pred_prob": 0.50},
        {"batter_id": 2, "gamepk": 101, "is_hit": 0, "pred_prob": 0.05},
        {"batter_id": 2, "gamepk": 101, "is_hit": 0, "pred_prob": 0.08},
        {"batter_id": 2, "gamepk": 101, "is_hit": 0, "pred_prob": 0.10},
    ]


def test_run_pa_vs_game_grain_check_returns_pa_and_game_metrics():
    df = _df(_pa_rows())
    val_df = df[["batter_id", "gamepk"]]
    y_val = df["is_hit"]
    proba = df["pred_prob"].to_numpy()

    pa_metrics, game_metrics, game_results = run_pa_vs_game_grain_check(
        val_df, y_val, proba, n_bins=2, pa_min_n=1, game_min_n=1,
    )

    expected_keys = {"reliability", "resolution", "roc_auc", "brier", "log_loss", "ece"}
    assert expected_keys <= pa_metrics.keys()
    assert expected_keys <= game_metrics.keys()


def test_run_pa_vs_game_grain_check_game_results_matches_aggregate_fn():
    df = _df(_pa_rows())
    val_df = df[["batter_id", "gamepk"]]
    y_val = df["is_hit"]
    proba = df["pred_prob"].to_numpy()

    _, _, game_results = run_pa_vs_game_grain_check(
        val_df, y_val, proba, n_bins=2, pa_min_n=1, game_min_n=1,
    )

    pa_results = val_df.copy()
    pa_results["is_hit"] = y_val.to_numpy()
    pa_results["pred_prob"] = proba
    expected = aggregate_pa_predictions_to_game(pa_results)

    pd.testing.assert_frame_equal(
        game_results.sort_values(["batter_id", "gamepk"]).reset_index(drop=True),
        expected.sort_values(["batter_id", "gamepk"]).reset_index(drop=True),
    )


def test_run_pa_vs_game_grain_check_respects_custom_group_cols():
    df = _df([
        {"batter": 1, "game": 100, "is_hit": 0, "pred_prob": 0.10},
        {"batter": 1, "game": 100, "is_hit": 1, "pred_prob": 0.60},
        {"batter": 2, "game": 100, "is_hit": 0, "pred_prob": 0.20},
        {"batter": 2, "game": 101, "is_hit": 1, "pred_prob": 0.55},
    ])
    val_df = df[["batter", "game"]]
    y_val = df["is_hit"]
    proba = df["pred_prob"].to_numpy()

    _, _, game_results = run_pa_vs_game_grain_check(
        val_df, y_val, proba, group_cols=("batter", "game"),
        n_bins=2, pa_min_n=1, game_min_n=1,
    )

    assert set(game_results.columns) >= {"batter", "game", "game_pred_prob", "game_is_hit"}
    assert len(game_results) == 3


def test_single_pa_game_matches_input_probability_and_outcome():
    df = _df([
        {"batter_id": 1, "gamepk": 100, "is_hit": 1, "pred_prob": 0.30},
    ])
    result = aggregate_pa_predictions_to_game(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["game_pred_prob"] == pytest.approx(0.30)
    assert row["game_is_hit"] == 1


def test_multi_pa_game_uses_complement_product_formula():
    # P(1+ hits) = 1 - (1-p1)(1-p2)(1-p3)
    df = _df([
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.20},
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.25},
        {"batter_id": 1, "gamepk": 100, "is_hit": 1, "pred_prob": 0.30},
    ])
    result = aggregate_pa_predictions_to_game(df)
    expected = 1 - (1 - 0.20) * (1 - 0.25) * (1 - 0.30)
    assert result.iloc[0]["game_pred_prob"] == pytest.approx(expected)


def test_any_hit_in_game_marks_game_is_hit_true():
    df = _df([
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.20},
        {"batter_id": 1, "gamepk": 100, "is_hit": 1, "pred_prob": 0.25},
    ])
    result = aggregate_pa_predictions_to_game(df)
    assert result.iloc[0]["game_is_hit"] == 1


def test_all_pa_no_hit_game_is_hit_false():
    df = _df([
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.20},
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.25},
    ])
    result = aggregate_pa_predictions_to_game(df)
    assert result.iloc[0]["game_is_hit"] == 0


def test_rows_with_nan_pred_prob_are_excluded_from_the_product():
    df = _df([
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.20},
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": np.nan},
    ])
    result = aggregate_pa_predictions_to_game(df)
    # the NaN row is skipped entirely from the probability product, not treated as p=0 or p=1
    assert result.iloc[0]["game_pred_prob"] == pytest.approx(0.20)


def test_preserves_one_row_per_group_and_group_keys():
    df = _df([
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.20},
        {"batter_id": 1, "gamepk": 100, "is_hit": 1, "pred_prob": 0.30},
        {"batter_id": 2, "gamepk": 100, "is_hit": 0, "pred_prob": 0.15},
        {"batter_id": 1, "gamepk": 101, "is_hit": 0, "pred_prob": 0.10},
    ])
    result = aggregate_pa_predictions_to_game(df)
    assert len(result) == 3
    assert set(result.columns) >= {"batter_id", "gamepk", "game_pred_prob", "game_is_hit", "n_pa"}
    assert set(zip(result["batter_id"], result["gamepk"])) == {(1, 100), (2, 100), (1, 101)}


def test_n_pa_counts_plate_appearances_in_the_game():
    df = _df([
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.20},
        {"batter_id": 1, "gamepk": 100, "is_hit": 1, "pred_prob": 0.30},
        {"batter_id": 1, "gamepk": 100, "is_hit": 0, "pred_prob": 0.10},
    ])
    result = aggregate_pa_predictions_to_game(df)
    assert result.iloc[0]["n_pa"] == 3


def test_custom_column_names_are_respected():
    df = _df([
        {"batter": 1, "game": 100, "hit": 1, "p": 0.4},
    ])
    result = aggregate_pa_predictions_to_game(
        df, group_cols=["batter", "game"], y_true_col="hit", y_prob_col="p"
    )
    assert result.iloc[0]["game_pred_prob"] == pytest.approx(0.4)
    assert result.iloc[0]["game_is_hit"] == 1
