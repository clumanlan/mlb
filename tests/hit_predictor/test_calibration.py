import numpy as np
import pandas as pd

from models.hit_predictor.utils.calibration import (
    apply_calibration,
    compare_calibration_before_after,
    fit_isotonic_calibrator,
    run_isotonic_calibration_check,
    split_game_results_for_calibration,
)


def _game_results(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "batter_id": np.arange(n),
        "gamepk": np.arange(1000, 1000 + n),
        "game_pred_prob": rng.uniform(0, 1, n),
        "game_is_hit": rng.integers(0, 2, n),
    })


def test_split_game_results_for_calibration_returns_disjoint_halves():
    df = _game_results(n=40)

    fit_df, eval_df = split_game_results_for_calibration(df, fit_frac=0.5, random_state=42)

    assert len(fit_df) + len(eval_df) == len(df)

    fit_keys = set(zip(fit_df["batter_id"], fit_df["gamepk"]))
    eval_keys = set(zip(eval_df["batter_id"], eval_df["gamepk"]))
    assert fit_keys.isdisjoint(eval_keys)


def test_split_game_results_for_calibration_is_deterministic():
    df = _game_results(n=40)

    fit_df1, eval_df1 = split_game_results_for_calibration(df, fit_frac=0.5, random_state=42)
    fit_df2, eval_df2 = split_game_results_for_calibration(df, fit_frac=0.5, random_state=42)

    pd.testing.assert_frame_equal(fit_df1, fit_df2)
    pd.testing.assert_frame_equal(eval_df1, eval_df2)


def test_fit_isotonic_calibrator_produces_monotonic_nondecreasing_mapping():
    rng = np.random.default_rng(0)
    n = 200
    # Deliberately overconfident: predicted prob is constant 0.9 regardless
    # of the true label, whose actual rate is ~0.5 — a real model would
    # never be this miscalibrated, but it's a clean case to test against.
    y_prob = np.full(n, 0.9)
    y_true = rng.integers(0, 2, n)

    calibrator = fit_isotonic_calibrator(y_true, y_prob)
    probes = np.linspace(0, 1, 11)
    predicted = calibrator.predict(probes)

    assert np.all(np.diff(predicted) >= 0)
    assert np.all((predicted >= 0) & (predicted <= 1))


def test_apply_calibration_returns_array_same_length_within_unit_interval():
    rng = np.random.default_rng(0)
    n = 200
    y_prob = rng.uniform(0, 1, n)
    y_true = rng.integers(0, 2, n)

    calibrator = fit_isotonic_calibrator(y_true, y_prob)
    calibrated = apply_calibration(calibrator, y_prob)

    assert len(calibrated) == n
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_compare_calibration_before_after_shows_lower_ece_for_calibrated_probs():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n)
    # Deliberately miscalibrated: ~0.9 regardless of true label (true rate ~0.5).
    # Non-constant (not exactly 0.9) so get_calibration_df's percentile-based
    # bin edges aren't degenerate.
    raw_prob = rng.uniform(0.85, 0.95, n)
    # Stand-in "perfectly calibrated" probs: exactly the true label, so every
    # quantile bucket's mean_pred matches its obs_hit_rate exactly (ECE == 0).
    calibrated_prob = y_true.astype(float)

    result = compare_calibration_before_after(
        y_true, raw_prob, calibrated_prob, n_bins=2, min_n=1,
    )

    assert result["raw"]["ece"] > result["calibrated"]["ece"]


def test_run_isotonic_calibration_check_returns_comparison_dict():
    game_results = _game_results(n=200)

    result = run_isotonic_calibration_check(
        game_results, fit_frac=0.5, random_state=42, n_bins=2, min_n=1,
    )

    assert set(result.keys()) == {"raw", "calibrated"}
    for key in ("raw", "calibrated"):
        assert "ece" in result[key]
        assert "reliability" in result[key]
        assert "resolution" in result[key]
