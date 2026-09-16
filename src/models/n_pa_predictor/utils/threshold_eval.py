import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score

from models.n_pa_predictor.processing.features.ablation import (
    cluster_correlated_features,
    pick_family_representatives,
)

WILSON_Z = 1.96  # 95% CI

# Same params as experiments/v1_low_pa_classifier/train.py's XGBoost, so an
# ablation run is comparable to the locked production result, not a
# differently-tuned model.
XGB_PARAMS = dict(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss")


def wilson_ci(p: float, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


def evaluate_feature_subset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float = 0.85,
) -> dict:
    """Fits v1's exact XGBoost config on feature_cols only, scores val_df,
    and reports precision + Wilson 95% CI at the given confidence threshold
    — the same shape as train.py's threshold-sweep rows, so an ablation
    result is directly comparable to the locked production CI."""
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(train_df[feature_cols], train_df[target_col])
    prob = model.predict_proba(val_df[feature_cols])[:, 1]

    pred = (prob >= threshold).astype(int)
    n_pred = int(pred.sum())
    if n_pred == 0:
        return {"precision": np.nan, "n": 0, "ci_low": np.nan, "ci_high": np.nan}

    precision = precision_score(val_df[target_col], pred, zero_division=np.nan)
    ci_low, ci_high = wilson_ci(precision, n_pred)
    return {"precision": precision, "n": n_pred, "ci_low": ci_low, "ci_high": ci_high}


def fit_and_get_importances(
    train_df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> pd.Series:
    """XGBoost feature_importances_ as data, keyed by feature name — used
    to pick a family representative and to order backward elimination.
    Not a substitute for the CI-guarded check in run_ablation_sweep: a
    feature can rank low here and still be worth keeping, or vice versa
    (this exact gap already burned k_predictor's v9/v11 features)."""
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(train_df[feature_cols], train_df[target_col])
    return pd.Series(model.feature_importances_, index=feature_cols)


def _ci_overlaps(a: dict, b: dict) -> bool:
    if a["n"] == 0 or b["n"] == 0:
        return False
    return max(a["ci_low"], b["ci_low"]) <= min(a["ci_high"], b["ci_high"])


def run_ablation_sweep(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float = 0.85,
    corr_threshold: float = 0.8,
) -> pd.DataFrame:
    """Redundancy pruning (collapse correlated families to their highest-
    importance member) followed by backward elimination (drop the weakest
    remaining survivor, retrain, stop once a drop's precision CI at
    `threshold` no longer overlaps the baseline's) — each step judged
    against the SAME threshold-sweep+Wilson-CI protocol as the locked
    production result, not raw importance rank alone."""
    rows = []

    baseline = evaluate_feature_subset(train_df, val_df, feature_cols, target_col, threshold)
    rows.append({
        "variant": "baseline", "dropped_feature": None, **baseline,
        "within_baseline_ci": True, "features_used": list(feature_cols),
    })

    importance = fit_and_get_importances(train_df, feature_cols, target_col)
    clusters = cluster_correlated_features(train_df, feature_cols, threshold=corr_threshold)
    reps = set(pick_family_representatives(clusters, importance))
    survivors = list(feature_cols)

    deduped_candidate = [c for c in feature_cols if c in reps]
    if set(deduped_candidate) != set(feature_cols):
        deduped_result = evaluate_feature_subset(train_df, val_df, deduped_candidate, target_col, threshold)
        overlaps = _ci_overlaps(baseline, deduped_result)
        rows.append({
            "variant": "deduplicated", "dropped_feature": None, **deduped_result,
            "within_baseline_ci": overlaps, "features_used": list(deduped_candidate),
        })
        # Only adopt the deduplicated set going forward if it didn't already
        # break the CI — otherwise backward elimination would build on a
        # subset already known to be worse, silently compounding the error.
        if overlaps:
            survivors = deduped_candidate

    for weakest in importance[survivors].sort_values().index:
        candidate = [c for c in survivors if c != weakest]
        if not candidate:
            break
        result = evaluate_feature_subset(train_df, val_df, candidate, target_col, threshold)
        overlaps = _ci_overlaps(baseline, result)
        rows.append({
            "variant": f"drop_{weakest}", "dropped_feature": weakest, **result,
            "within_baseline_ci": overlaps, "features_used": list(candidate),
        })
        if not overlaps:
            break
        survivors = candidate

    return pd.DataFrame(rows)
