import numpy as np
import pandas as pd
import pytest

from models.hit_predictor.utils.eval import (
    aggregate_pa_predictions_to_game,
    run_pa_vs_game_grain_check,
    save_predictions,
    summarize_verdict,
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


def test_save_predictions_writes_both_parquet_files_with_default_names(tmp_path):
    """predictions_pa is keyed on (gamepk, play_id) rather than
    (batter_id, gamepk) like run_pa_vs_game_grain_check's internal
    pa_results — play_id is what makes a PA-grain row unique, so callers
    that need per-PA predictions (not just per-batter-game) build this
    frame themselves and hand it here as-is; this function just persists
    whatever two frames it's given."""

    pa_df = pd.DataFrame({
        "gamepk": [100, 100],
        "play_id": [1, 2],
        "batter_id": [1, 1],
        "is_hit": [0, 1],
        "pred_prob": [0.10, 0.60],
    })
    game_df = pd.DataFrame({
        "batter_id": [1],
        "gamepk": [100],
        "game_pred_prob": [0.46],
        "game_is_hit": [1],
    })

    pa_path, game_path = save_predictions(pa_df, game_df, tmp_path)

    assert pa_path == tmp_path / "predictions_pa.parquet"
    assert game_path == tmp_path / "predictions_game.parquet"
    assert pa_path.exists()
    assert game_path.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(pa_path), pa_df)
    pd.testing.assert_frame_equal(pd.read_parquet(game_path), game_df)


def test_save_predictions_creates_output_dir_if_missing(tmp_path):
    pa_df = pd.DataFrame({"gamepk": [1], "play_id": [1], "pred_prob": [0.1]})
    game_df = pd.DataFrame({"gamepk": [1], "game_pred_prob": [0.1]})

    pa_path, game_path = save_predictions(pa_df, game_df, tmp_path / "nested" / "dir")

    assert pa_path.exists()
    assert game_path.exists()


def test_custom_column_names_are_respected():
    df = _df([
        {"batter": 1, "game": 100, "hit": 1, "p": 0.4},
    ])
    result = aggregate_pa_predictions_to_game(
        df, group_cols=["batter", "game"], y_true_col="hit", y_prob_col="p"
    )
    assert result.iloc[0]["game_pred_prob"] == pytest.approx(0.4)
    assert result.iloc[0]["game_is_hit"] == 1


# ── summarize_verdict ────────────────────────────────────────────────────────
# Codifies BENCHMARKS.md §2's decision rule so "is this a real improvement"
# is answered by running a function, not by eyeballing two numbers across two
# separate evaluate_hit_predictor() printouts. See the "Reliability &
# Resolution" write-up (BENCHMARKS.md footer) for the Model A/B/C reasoning
# behind why both checks are required together.

def test_real_improvement_when_resolution_up_and_reliability_flat_or_better():
    # Resolution improves (more differentiation) and reliability doesn't get
    # worse (still honest) -- the clean win case (Model B vs Model A).
    baseline = {"reliability": 0.0013, "resolution": 0.0001}
    new = {"reliability": 0.0005, "resolution": 0.0159}

    result = summarize_verdict(baseline, new)

    assert result["trustworthy"] is True
    assert result["differentiated"] is True
    assert result["verdict"] == "real_improvement"


def test_overconfidence_risk_when_resolution_up_but_reliability_worse():
    # Same resolution gain, but reliability got worse -- the Model C trap:
    # more spread in predictions, but the spread itself is dishonest.
    baseline = {"reliability": 0.0005, "resolution": 0.0001}
    new = {"reliability": 0.0137, "resolution": 0.0037}

    result = summarize_verdict(baseline, new)

    assert result["trustworthy"] is False
    assert result["differentiated"] is True
    assert result["verdict"] == "overconfidence_risk"


def test_calibration_only_when_reliability_improves_but_resolution_flat():
    # Probabilities got more honest, but the model isn't differentiating any
    # better than before -- narrower win, not "no improvement."
    baseline = {"reliability": 0.0030, "resolution": 0.0135}
    new = {"reliability": 0.0015, "resolution": 0.0135}

    result = summarize_verdict(baseline, new)

    assert result["trustworthy"] is True
    assert result["differentiated"] is False
    assert result["verdict"] == "calibration_only"


def test_no_improvement_when_both_flat_or_worse():
    # Naive-vs-model at game grain per BENCHMARKS.md §1: model's resolution
    # is lower than naive's, and reliability isn't better either.
    baseline = {"reliability": 0.0030, "resolution": 0.0182}  # naive
    new = {"reliability": 0.0030, "resolution": 0.0138}       # v5 model

    result = summarize_verdict(baseline, new)

    assert result["trustworthy"] is True
    assert result["differentiated"] is False
    assert result["verdict"] == "no_improvement"


def test_returns_raw_deltas():
    baseline = {"reliability": 0.0030, "resolution": 0.0182}
    new = {"reliability": 0.0015, "resolution": 0.0159}

    result = summarize_verdict(baseline, new)

    assert result["reliability_delta"] == pytest.approx(0.0015 - 0.0030)
    assert result["resolution_delta"] == pytest.approx(0.0159 - 0.0182)
