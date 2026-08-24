"""
utils/calibration.py

Isotonic recalibration for game-grain hit_predictor probabilities. Per
BENCHMARKS.md §4.5, aggregate_pa_predictions_to_game's `1-prod(1-p)`
formula is systematically overconfident — a batter's PAs in one game
share pitcher/park/weather, so they're positively correlated, not
independent draws. This module fixes calibration only; it does not
address the separate, still-open discrimination gap (ROADMAP.md item 1).
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from models.hit_predictor.utils.eval import (
    expected_calibration_error,
    get_calibration_df,
    murphy_decomposition,
)


def split_game_results_for_calibration(game_results, fit_frac=0.5, random_state=42):
    """
    Split game-grain results into a calib-fit half and a calib-eval half,
    so isotonic regression is never fit and scored on the same rows (that
    would show trivially-optimistic calibration, not a fair test).

    Stratified on game_is_hit — game-grain N is already small (this
    project's GAME_MIN_N=200 bucket threshold), so an unstratified random
    split could land a badly class-imbalanced eval half by chance.

    Parameters
    ----------
    game_results : pd.DataFrame
        Output of aggregate_pa_predictions_to_game — must have game_is_hit.
    fit_frac : float, default 0.5
        Fraction of rows assigned to the fit half.
    random_state : int, default 42
        Passed to train_test_split for deterministic, reproducible splits.

    Returns
    -------
    (fit_df, eval_df) : tuple of pd.DataFrame
    """
    return train_test_split(
        game_results,
        train_size=fit_frac,
        random_state=random_state,
        stratify=game_results["game_is_hit"],
    )


def fit_isotonic_calibrator(y_true, y_prob):
    """
    Fit an isotonic regression mapping raw predicted probability -> calibrated
    probability. Clipped to [0, 1] (out_of_bounds='clip') so a probe outside
    the fit range never returns a value outside the valid probability range.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_prob : array-like of float
        Raw predicted probabilities (e.g. game_pred_prob) to calibrate against.

    Returns
    -------
    sklearn.isotonic.IsotonicRegression, already fit.
    """
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(y_prob, y_true)


def apply_calibration(calibrator, y_prob):
    """
    Apply a fitted calibrator (from fit_isotonic_calibrator) to raw predicted
    probabilities, returning the calibrated array — same length as y_prob,
    every value in [0, 1] (guaranteed by the calibrator's own y_min/y_max/
    out_of_bounds clipping).
    """
    return np.asarray(calibrator.predict(y_prob))


def _calibration_summary(y_true, y_prob, n_bins, min_n):
    calibration_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=min_n)
    summary = murphy_decomposition(y_true, calibration_df)
    summary["ece"] = expected_calibration_error(calibration_df)
    return summary


def compare_calibration_before_after(y_true, raw_prob, calibrated_prob, n_bins=10, min_n=0):
    """
    Compare calibration quality before vs. after isotonic recalibration, on
    the same y_true labels — reuses eval.py's existing calibration_df/ECE/
    Murphy-decomposition machinery rather than duplicating that math.

    Parameters
    ----------
    y_true : array-like of 0/1
    raw_prob : array-like of float
        Predicted probabilities before calibration (e.g. game_pred_prob).
    calibrated_prob : array-like of float
        Same predictions after apply_calibration.
    n_bins, min_n : passed through to get_calibration_df for both.

    Returns
    -------
    dict with 'raw' and 'calibrated' keys, each a dict with 'ece',
    'reliability', 'resolution', 'uncertainty', 'brier_reconstructed'.
    """
    return {
        "raw": _calibration_summary(y_true, raw_prob, n_bins, min_n),
        "calibrated": _calibration_summary(y_true, calibrated_prob, n_bins, min_n),
    }


def run_isotonic_calibration_check(game_results, fit_frac=0.5, random_state=42, n_bins=10, min_n=0):
    """
    One-call isotonic calibration check for game-grain predictions — split,
    fit, apply, compare. Mirrors run_pa_vs_game_grain_check's shape (eval.py):
    any experiment's train.py can call this once it has a game_results df
    (from aggregate_pa_predictions_to_game), instead of composing
    split_game_results_for_calibration/fit_isotonic_calibrator/
    apply_calibration/compare_calibration_before_after itself.

    Fits the isotonic calibrator on one half of game_results and reports the
    before/after comparison on the OTHER half only — never on the fit half —
    so the reported improvement isn't just the calibrator overfitting its own
    training data. See split_game_results_for_calibration's docstring.

    Parameters
    ----------
    game_results : pd.DataFrame
        Output of aggregate_pa_predictions_to_game — must have game_pred_prob,
        game_is_hit.
    fit_frac, random_state : passed through to split_game_results_for_calibration.
    n_bins, min_n : passed through to compare_calibration_before_after.

    Returns
    -------
    dict with 'raw' and 'calibrated' keys, as returned by
    compare_calibration_before_after — computed on the calib-eval half.
    """
    fit_df, eval_df = split_game_results_for_calibration(
        game_results, fit_frac=fit_frac, random_state=random_state,
    )

    calibrator = fit_isotonic_calibrator(fit_df["game_is_hit"], fit_df["game_pred_prob"])
    calibrated_eval_prob = apply_calibration(calibrator, eval_df["game_pred_prob"])

    return compare_calibration_before_after(
        eval_df["game_is_hit"], eval_df["game_pred_prob"], calibrated_eval_prob,
        n_bins=n_bins, min_n=min_n,
    )
