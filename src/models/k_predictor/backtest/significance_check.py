"""
k_predictor production backtest -- statistical significance check on
edge_report.py's 131-start real-2025-odds backtest.

edge_report.py's headline number (42.2% follow-the-model win rate on the 45
disagreement starts) and calibration_check.py's model-vs-market resolution
comparison (model 0.0051 vs market 0.0191) were both reported as point
estimates on a small sample without a significance test. This script asks
the obvious next question before anyone treats either number as a real
finding: are they distinguishable from noise at this sample size?

Three checks:

1. Binomial test on the disagreement win rate against two nulls: a coin
   flip (0.50) and the ~-110 break-even rate (0.524).
2. Bootstrap CI on the win rate and on the mean edge, by resampling starts
   with replacement (percentile method, 10,000 resamples).
3. Bootstrap CI on the model-vs-market resolution GAP (market resolution
   minus model resolution) from calibration_check.py's finding -- resample
   the 131 rows, recompute both resolutions each time, see whether the gap's
   CI excludes zero.

Run from src/models/k_predictor/ with:
    python backtest/significance_check.py
No AWS credentials needed -- reads edge_report.parquet, already on disk.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))
from models.hit_predictor.utils.eval import get_calibration_df, murphy_decomposition

BACKTEST_DIR = Path(__file__).parent
EDGE_REPORT = BACKTEST_DIR / "edge_report.parquet"
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def resolution(y_true, y_prob, n_bins=5):
    cal_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=0)
    return murphy_decomposition(y_true, cal_df)["resolution"]


def main():
    edge = pd.read_parquet(EDGE_REPORT)
    edge["model_favored_over"] = edge["model_p_over"] > 0.5
    edge["market_favored_over"] = edge["market_p_over"] > 0.5
    edge["disagree"] = edge["model_favored_over"] != edge["market_favored_over"]

    disagree = edge[edge["disagree"]].copy()
    disagree["follow_correct"] = (
        disagree["model_favored_over"] == disagree["realized_over"].astype(bool)
    ).astype(int)
    n_win = disagree["follow_correct"].sum()
    n_total = len(disagree)
    win_rate = n_win / n_total

    print(f"{'=' * 72}\n1. BINOMIAL TEST — disagreement win rate\n{'=' * 72}")
    print(f"{n_win}/{n_total} = {win_rate:.1%}")

    for null_label, null_p in [("coin flip", 0.50), ("~-110 break-even", 0.524)]:
        result = binomtest(n_win, n_total, p=null_p, alternative="two-sided")
        ci_lo, ci_hi = result.proportion_ci(confidence_level=0.95, method="wilson")
        print(f"  vs. {null_label} ({null_p:.1%}): p={result.pvalue:.3f}  "
              f"95% Wilson CI on win rate=[{ci_lo:.1%}, {ci_hi:.1%}]"
              f"{'  -> excludes null, significant' if not (ci_lo <= null_p <= ci_hi) else '  -> null inside CI, NOT significant'}")

    print(f"\n{'=' * 72}\n2. BOOTSTRAP CI (10,000 resamples, percentile method)\n{'=' * 72}")

    follow_correct = disagree["follow_correct"].to_numpy()
    boot_win_rates = np.array([
        follow_correct[RNG.integers(0, len(follow_correct), size=len(follow_correct))].mean()
        for _ in range(N_BOOT)
    ])
    lo, hi = np.percentile(boot_win_rates, [2.5, 97.5])
    print(f"Win rate: {win_rate:.1%}, 95% bootstrap CI=[{lo:.1%}, {hi:.1%}]"
          f"{'  -> excludes 50%' if lo > 0.5 or hi < 0.5 else '  -> includes 50%, NOT significant'}")

    edge_vals = edge["edge"].to_numpy()
    boot_mean_edges = np.array([
        edge_vals[RNG.integers(0, len(edge_vals), size=len(edge_vals))].mean()
        for _ in range(N_BOOT)
    ])
    lo, hi = np.percentile(boot_mean_edges, [2.5, 97.5])
    print(f"Mean edge (model - devigged market): {edge_vals.mean():+.4f}, "
          f"95% bootstrap CI=[{lo:+.4f}, {hi:+.4f}]"
          f"{'  -> excludes 0' if lo > 0 or hi < 0 else '  -> includes 0, NOT significant'}")

    print(f"\n{'=' * 72}\n3. BOOTSTRAP CI — model vs. market resolution gap\n{'=' * 72}")
    y_true = edge["realized_over"].to_numpy()
    model_p = edge["model_p_over"].to_numpy()
    market_p = edge["market_p_over"].to_numpy()
    n = len(edge)

    model_res = resolution(y_true, model_p)
    market_res = resolution(y_true, market_p)
    print(f"Point estimate: model resolution={model_res:.4f}  market resolution={market_res:.4f}  "
          f"gap (market - model)={market_res - model_res:+.4f}")

    boot_gaps = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        yb, mb, kb = y_true[idx], model_p[idx], market_p[idx]
        # skip degenerate resamples with no outcome variance -- resolution is undefined there
        if yb.sum() == 0 or yb.sum() == n:
            boot_gaps[i] = np.nan
            continue
        boot_gaps[i] = resolution(yb, kb, n_bins=5) - resolution(yb, mb, n_bins=5)
    valid = boot_gaps[~np.isnan(boot_gaps)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    print(f"95% bootstrap CI on gap (market - model), {len(valid)}/{N_BOOT} valid resamples: [{lo:+.4f}, {hi:+.4f}]"
          f"{'  -> excludes 0, real gap' if lo > 0 or hi < 0 else '  -> includes 0, NOT distinguishable from noise'}")

    print(
        f"\n{'=' * 72}\nBOTTOM LINE\n{'=' * 72}\n"
        "n=131 (45 disagreement starts) is underpowered for either the win-rate or the "
        "resolution-gap claim to stand on its own -- see the CIs above. Next lever is more "
        "days, not more analysis of this same sample."
    )


if __name__ == "__main__":
    main()
