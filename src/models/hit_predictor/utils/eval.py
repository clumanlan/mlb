"""
utils/eval.py

Model-agnostic evaluation utilities for binary hit/no-hit predictors.
Every function here operates only on (y_true, y_prob) arrays or a
`results` dict of them — nothing in this file knows how those
probabilities were produced. Training code (sklearn, XGBoost, the
rules-based baseline, NGBoost, whatever comes next) lives elsewhere and
just needs to hand this file a dict shaped like:

    results = {
        "Some model": {"log_loss": ..., "brier": ..., "proba": y_prob_array},
        ...
    }
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)


def _make_bucket_row(label, y_true, y_prob, min_n):
    n = len(y_true)
    return {
        "bucket": label,
        "n": n,
        "reliable": n >= min_n,
        # mean predicted prob — should track obs_hit_rate if well calibrated
        "mean_pred": round(y_prob.mean(), 3) if n else float("nan"),
        # actual hit rate for rows in this bin
        "obs_hit_rate": round(y_true.mean(), 3) if n else float("nan"),
    }


def get_calibration_df(y_true, y_prob, n_bins=10, min_n=0):
    """
    Bucket predictions into n_bins equal-count (quantile) bins and compare
    mean predicted probability to observed hit rate per bucket.

    Buckets with fewer than min_n rows are flagged via the `reliable`
    column rather than dropped — visibility into thin buckets matters
    most at the tails, which is exactly where you're likely to have
    the fewest points.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    rows = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        rows.append(_make_bucket_row(
            label=f"{lo:.3f}\u2013{hi:.3f}",
            y_true=y_true[mask],
            y_prob=y_prob[mask],
            min_n=min_n,
        ))
    return pd.DataFrame(rows)


def expected_calibration_error(calibration_df):
    """
    Expected Calibration Error (ECE): a single-number summary of how far
    predicted probabilities are from observed outcomes, averaged across
    buckets and weighted by bucket size.

    ECE = sum_k (n_k / N) * |mean_pred_k - obs_hit_rate_k|

    Lower is better; 0 = perfectly calibrated. Useful for comparing many
    models at a glance, but always look at the full calibration table too
    — ECE can hide a model that's great in most buckets and badly off in
    one (e.g. exactly the tails you care about), since errors in
    different buckets can partially cancel in the weighted average only
    if you're looking at signed error; here we use absolute error per
    bucket so they don't cancel across buckets, but a large error in a
    small bucket still gets diluted by its small weight.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Output of get_calibration_df — must have 'n', 'mean_pred', and
        'obs_hit_rate' columns.

    Returns
    -------
    float
    """
    total_n = calibration_df["n"].sum()
    weighted_error = (
        calibration_df["n"] / total_n
        * (calibration_df["mean_pred"] - calibration_df["obs_hit_rate"]).abs()
    )
    return weighted_error.sum()


def murphy_decomposition(y_true, calibration_df):
    """
    Murphy's decomposition of the Brier score:

        Brier = reliability - resolution + uncertainty

    This splits one aggregate number into three interpretable pieces, so
    you can tell *why* Brier score changed between two models rather than
    just that it did:

    - uncertainty: irreducible noise in the outcome itself (base rate
      variance, y_bar * (1 - y_bar)). Identical for any model evaluated
      on the same y_true — not something a model can improve.
    - reliability: how far off predicted probabilities are from observed
      rates within each bucket, weighted by bucket size. This is
      calibration error — lower is better. (Note: this is computed from
      binned means as an approximation, using the same buckets as
      get_calibration_df, rather than grouping by every distinct
      predicted value.)
    - resolution: how much the observed rate varies *across* buckets
      relative to the overall base rate, weighted by bucket size. This is
      discriminative power — higher is better (the model is saying
      meaningfully different things about different buckets, and those
      differences are real).

    A model can improve Brier score by getting better calibrated
    (reliability drops) or by getting better at separating hits from
    non-hits (resolution rises) — this tells you which one happened.

    Parameters
    ----------
    y_true : array-like of 0/1
    calibration_df : pd.DataFrame
        Output of get_calibration_df for this y_true/y_prob pair — must
        have 'n', 'mean_pred', and 'obs_hit_rate' columns.

    Returns
    -------
    dict with keys 'reliability', 'resolution', 'uncertainty',
    'brier_reconstructed' (should closely match the true Brier score;
    small gaps come from using binned means rather than raw values).
    """
    y_true = np.asarray(y_true)
    n_total = calibration_df["n"].sum()
    y_bar = y_true.mean()

    reliability = (
        calibration_df["n"] * (calibration_df["mean_pred"] - calibration_df["obs_hit_rate"]) ** 2
    ).sum() / n_total

    resolution = (
        calibration_df["n"] * (calibration_df["obs_hit_rate"] - y_bar) ** 2
    ).sum() / n_total

    uncertainty = y_bar * (1 - y_bar)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_reconstructed": reliability - resolution + uncertainty,
    }


def plot_calibration_curve(y_true, results: dict, n_bins=10, min_n=0, save_path=None):
    """
    Plot calibration curves for one or more models against a perfect-
    calibration diagonal.

    The diagonal range is derived from the actual predicted probabilities
    across all models being plotted, rather than a fixed guess — this
    keeps the plot correct whether you're looking at PA-level hit
    probabilities (~0.15-0.35) or something with a completely different
    range (e.g. a player skill-level model).

    Unreliable buckets (n < min_n) are drawn as hollow markers so thin
    bins are visible but visually distinct from trustworthy ones.

    Returns (fig, ax). If save_path is given, saves to disk; otherwise
    shows interactively. Either way the figure is returned so callers
    can do both, or neither.
    """
    all_probs = np.concatenate([np.asarray(r["proba"]) for r in results.values()])
    lo, hi = all_probs.min(), all_probs.max()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([lo, hi], [lo, hi], 'k--', label='Perfect calibration')

    for name, r in results.items():
        cal_df = get_calibration_df(y_true, r["proba"], n_bins=n_bins, min_n=min_n)
        reliable = cal_df[cal_df["reliable"]]
        unreliable = cal_df[~cal_df["reliable"]]

        line, = ax.plot(reliable["mean_pred"], reliable["obs_hit_rate"],
                         marker='o', label=name)
        if len(unreliable):
            ax.plot(unreliable["mean_pred"], unreliable["obs_hit_rate"],
                     marker='o', linestyle='none', markerfacecolor='none',
                     markeredgecolor=line.get_color())

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed hit rate")
    ax.set_title("Calibration curve" + (f"  (hollow = n < {min_n})" if min_n else ""))
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()

    return fig, ax


def evaluate_hit_predictor(y_true, y_prob, baseline_prob=None, base_rate=0.22,
                            n_bins=10, min_n=500):
    """
    Print a full evaluation report for a single model's predictions and
    return the underlying metrics as a dict.

    Covers two distinct questions, since a model can fail at either one
    independently of the other:

    1. Are the predicted probabilities honest? (calibration)
       - CALIBRATION TABLE: bucketed mean predicted prob vs observed hit
         rate, with a `reliable` flag on buckets below min_n.
       - CALIBRATION SUMMARY: ECE (single-number calibration error) and
         Murphy's decomposition of Brier score into reliability
         (calibration error), resolution (discriminative power), and
         uncertainty (irreducible base-rate noise) — so you can tell
         *why* Brier score moved, not just that it did.

    2. Does the model separate hits from non-hits? (discrimination /
       ranking)
       - DISCRIMINATION METRICS: top-decile vs bottom-decile hit rate,
         reported as a multiple of base_rate.
       - RANKING METRICS: ROC-AUC (threshold-independent ranking
         quality, 0.5 = random, 1.0 = perfect) and PR-AUC (like ROC-AUC
         but weighted toward the positive class — more informative than
         ROC-AUC here since hits are the minority class at ~base_rate).
         AUC can be high even when calibration is bad (a model can rank
         correctly while being systematically over/under-confident), so
         this is a genuinely separate check from everything above, not
         a restatement of it.

    Also reports probability-quality scores:
       - Brier score (mean squared error between predicted prob and
         outcome) and log loss (cross-entropy — penalizes confident
         wrong predictions much more severely than Brier does), both
         vs an optional baseline for comparison.

    Parameters
    ----------
    y_true : array-like of 0/1
        Actual outcomes (1 = hit).
    y_prob : array-like of float
        Predicted probability of a hit, same length as y_true.
    baseline_prob : array-like of float, optional
        A reference model's predictions (e.g. naive/rules-based
        baseline) to compare Brier and log loss against. If None,
        baseline comparisons are skipped.
    base_rate : float, default 0.22
        The population's overall hit rate, used to express decile hit
        rates as a multiple (e.g. "1.2x base rate") rather than a raw
        number alone.
    n_bins : int, default 10
        Number of quantile buckets for the calibration table/summary.
    min_n : int, default 500
        Minimum bucket/decile size below which results are flagged as
        unreliable rather than dropped — thin buckets are common at the
        tails, which tend to be exactly where you care most.

    Returns
    -------
    dict with keys: calibration_df, ece, reliability, resolution,
    uncertainty, brier, log_loss, roc_auc, pr_auc, top_decile_rate,
    bottom_decile_rate, decile_spread, and (if baseline_prob given)
    baseline_brier, baseline_log_loss.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    metrics = {}

    # ── Calibration table ─────────────────────────────────────────────
    calibration_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=min_n)
    metrics["calibration_df"] = calibration_df

    print("=" * 75)
    print("CALIBRATION TABLE")
    print("=" * 75)
    print(calibration_df.to_string(index=False))
    if not calibration_df["reliable"].all():
        n_thin = (~calibration_df["reliable"]).sum()
        print(f"\n  Warning: {n_thin} bucket(s) below min_n={min_n} \u2014 treat with caution")

    # ── Calibration summary: ECE + Murphy decomposition ───────────────
    ece = expected_calibration_error(calibration_df)
    decomp = murphy_decomposition(y_true, calibration_df)
    metrics["ece"] = ece
    metrics.update(decomp)

    print("\n" + "=" * 75)
    print("CALIBRATION SUMMARY")
    print("=" * 75)
    print(f"  ECE (expected calibration error) : {ece:.4f}  (lower = better, 0 = perfect)")
    print(f"  Reliability (calibration error)  : {decomp['reliability']:.4f}  (lower = better)")
    print(f"  Resolution (discriminative power): {decomp['resolution']:.4f}  (higher = better)")
    print(f"  Uncertainty (irreducible noise)  : {decomp['uncertainty']:.4f}  (fixed by base rate)")
    print(f"  Reconstructed Brier (rel-res+unc): {decomp['brier_reconstructed']:.4f}")

    # split into deciles, compare actual hit rate of top vs bottom 10%
    # large spread = model has real signal, small spread = model isn't separating well
    print("\n" + "=" * 75)
    print("DISCRIMINATION METRICS")
    print("=" * 75)
    p10, p90 = np.percentile(y_prob, [10, 90])
    bottom_mask = y_prob <= p10
    top_mask = y_prob >= p90
    bottom_n, top_n = bottom_mask.sum(), top_mask.sum()
    bottom_rate = y_true[bottom_mask].mean()
    top_rate = y_true[top_mask].mean()
    metrics["top_decile_rate"] = top_rate
    metrics["bottom_decile_rate"] = bottom_rate
    metrics["decile_spread"] = top_rate - bottom_rate

    print(f"  Decile cutoffs         : bottom {p10:.3f} / top {p90:.3f}")
    print(f"  Top-decile hit rate    : {top_rate:.3f}  (N={top_n})"
          f"  \u2014 {top_rate / base_rate:.2f}x base rate")
    print(f"  Bottom-decile hit rate : {bottom_rate:.3f}  (N={bottom_n})"
          f"  \u2014 {bottom_rate / base_rate:.2f}x base rate")
    print(f"  Spread                 : {top_rate - bottom_rate:.3f}")
    if top_n < min_n or bottom_n < min_n:
        print(f"\n  Warning: a decile has fewer than min_n={min_n} rows \u2014 "
              f"spread estimate may be noisy")

    # ── Ranking metrics: AUC-ROC and PR-AUC ────────────────────────────
    # Threshold-independent, and deliberately separate from calibration —
    # a model can rank well (high AUC) while still being badly
    # over/under-confident in its actual probability values.
    print("\n" + "=" * 75)
    print("RANKING METRICS")
    print("=" * 75)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc
    print(f"  ROC-AUC : {roc_auc:.4f}  (0.5 = random, 1.0 = perfect ranking)")
    print(f"  PR-AUC  : {pr_auc:.4f}  (baseline = base rate \u2248 {base_rate:.3f}; "
          f"more informative than ROC-AUC given hits are the minority class)")

    # ── Probability-quality scores: Brier + log loss ───────────────────
    # Brier = mean squared error between predicted prob and outcome.
    # Log loss penalizes confident wrong predictions far more heavily —
    # tracking both catches cases where a model is Brier-competitive but
    # occasionally badly overconfident, which Brier under-punishes
    # relative to log loss's log-scale penalty.
    print("\n" + "=" * 75)
    print("PROBABILITY SCORES")
    print("=" * 75)
    model_brier = brier_score_loss(y_true, y_prob)
    model_ll = log_loss(y_true, y_prob, labels=[0, 1])
    metrics["brier"] = model_brier
    metrics["log_loss"] = model_ll
    print(f"  Model Brier    : {model_brier:.4f}")
    print(f"  Model LogLoss  : {model_ll:.4f}")
    if baseline_prob is not None:
        baseline_prob = np.asarray(baseline_prob)
        baseline_brier = brier_score_loss(y_true, baseline_prob)
        baseline_ll = log_loss(y_true, baseline_prob, labels=[0, 1])
        metrics["baseline_brier"] = baseline_brier
        metrics["baseline_log_loss"] = baseline_ll

        brier_delta = baseline_brier - model_brier
        ll_delta = baseline_ll - model_ll
        brier_dir = "better" if brier_delta > 0 else "worse"
        ll_dir = "better" if ll_delta > 0 else "worse"
        print(f"  Baseline Brier : {baseline_brier:.4f}")
        print(f"  Baseline LogLoss: {baseline_ll:.4f}")
        print(f"  Brier Delta    : {abs(brier_delta):.4f} (model is {brier_dir} than baseline)")
        print(f"  LogLoss Delta  : {abs(ll_delta):.4f} (model is {ll_dir} than baseline)")

    return metrics