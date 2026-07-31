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
from sklearn.metrics import brier_score_loss


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
    Print a full evaluation report for a single model's predictions:
    calibration table, discrimination (top vs bottom decile), and Brier
    score vs an optional baseline.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    calibration_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=min_n)
    print("=" * 75)
    print("CALIBRATION TABLE")
    print("=" * 75)
    print(calibration_df.to_string(index=False))
    if not calibration_df["reliable"].all():
        n_thin = (~calibration_df["reliable"]).sum()
        print(f"\n  Warning: {n_thin} bucket(s) below min_n={min_n} \u2014 treat with caution")

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

    print(f"  Decile cutoffs         : bottom {p10:.3f} / top {p90:.3f}")
    print(f"  Top-decile hit rate    : {top_rate:.3f}  (N={top_n})"
          f"  \u2014 {top_rate / base_rate:.2f}x base rate")
    print(f"  Bottom-decile hit rate : {bottom_rate:.3f}  (N={bottom_n})"
          f"  \u2014 {bottom_rate / base_rate:.2f}x base rate")
    print(f"  Spread                 : {top_rate - bottom_rate:.3f}")
    if top_n < min_n or bottom_n < min_n:
        print(f"\n  Warning: a decile has fewer than min_n={min_n} rows \u2014 "
              f"spread estimate may be noisy")

    # brier score is MSE between predicted prob and actual outcome — lower is better
    # positive delta = baseline has higher error = model wins
    print("\n" + "=" * 75)
    print("BRIER SCORES")
    print("=" * 75)
    model_brier = brier_score_loss(y_true, y_prob)
    print(f"  Model    : {model_brier:.4f}")
    if baseline_prob is not None:
        baseline_prob = np.asarray(baseline_prob)
        baseline_brier = brier_score_loss(y_true, baseline_prob)
        delta = baseline_brier - model_brier
        direction = "better" if delta > 0 else "worse"
        print(f"  Baseline : {baseline_brier:.4f}")
        print(f"  Delta    : {abs(delta):.4f} (model is {direction} than baseline)")