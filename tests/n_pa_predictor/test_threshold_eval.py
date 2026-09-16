from models.n_pa_predictor.utils.threshold_eval import wilson_ci


def test_wilson_ci_matches_locked_production_ci():
    """Anchors the extracted function to the already-reported, locked
    production number (train.py, 2026-08-23): XGBoost @ 0.85 threshold,
    precision 64.7%, n=116 -> 95% CI [55.6%, 72.8%]. If this drifts, the
    extraction changed behavior, not just location."""
    lo, hi = wilson_ci(0.647, 116)

    # p=0.647 is itself the rounded-to-3-decimals precision, not the exact
    # underlying fraction, so allow a small tolerance rather than an exact
    # decimal match.
    assert abs(lo * 100 - 55.6) < 0.2
    assert abs(hi * 100 - 72.8) < 0.2


def test_wilson_ci_returns_nan_for_zero_n():
    lo, hi = wilson_ci(0.5, 0)

    assert lo != lo  # nan
    assert hi != hi  # nan
