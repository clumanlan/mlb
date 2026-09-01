"""
k_predictor production backtest -- statistical significance check on
edge_report_2026.py's 1,160-start real-2026-odds backtest (66 dates), plus
a proper in-sample isotonic recalibration check now that there's finally
enough data for one.

This is the 2026 sequel to significance_check.py (which ran on the original
131-start 2025 sample and found nothing distinguishable from noise) and
calibration_check.py (which used a disconnected fixed-line 2024 val-season
sample to rule out miscalibration as 2025's explanation, because 131 rows
wasn't enough to test it directly). With 1,160 real matched starts, both
checks can finally be run directly and honestly on the real backtest data
itself, split into a calibration-fit half and a calibration-eval half.

Run from src/models/k_predictor/ with:
    python backtest/significance_check_2026.py
No AWS credentials needed -- reads edge_report_2026.parquet, already on disk.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))
from models.hit_predictor.utils.eval import get_calibration_df, murphy_decomposition
from models.hit_predictor.utils.calibration import fit_isotonic_calibrator, apply_calibration

BACKTEST_DIR = Path(__file__).parent
EDGE_REPORT = BACKTEST_DIR / "edge_report_2026.parquet"
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def resolution(y_true, y_prob, n_bins=8):
    cal_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=0)
    return murphy_decomposition(y_true, cal_df)["resolution"]


def reliability(y_true, y_prob, n_bins=8):
    cal_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=0)
    return murphy_decomposition(y_true, cal_df)["reliability"]


def bootstrap_gap(y, a, b, stat_fn, n_boot=N_BOOT):
    """95% CI on stat_fn(y,a) - stat_fn(y,b) via paired resampling."""
    n = len(y)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            gaps[i] = np.nan
            continue
        gaps[i] = stat_fn(yb, a[idx]) - stat_fn(yb, b[idx])
    valid = gaps[~np.isnan(gaps)]
    return np.percentile(valid, 2.5), np.percentile(valid, 97.5)


def main():
    edge = pd.read_parquet(EDGE_REPORT)
    edge["model_favored_over"] = edge["model_p_over"] > 0.5
    edge["market_favored_over"] = edge["market_p_over"] > 0.5
    edge["disagree"] = edge["model_favored_over"] != edge["market_favored_over"]

    print(f"{'=' * 72}\n1. SIGNIFICANCE CHECK (n={len(edge)} matched, {edge['date'].nunique()} dates)\n{'=' * 72}")

    disagree = edge[edge["disagree"]].copy()
    disagree["follow_correct"] = (
        disagree["model_favored_over"] == disagree["realized_over"].astype(bool)
    ).astype(int)
    n_win, n_total = int(disagree["follow_correct"].sum()), len(disagree)
    win_rate = n_win / n_total
    print(f"Disagreement win rate: {n_win}/{n_total} = {win_rate:.1%}")
    for label, p0 in [("coin flip", 0.50), ("~-110 break-even", 0.524)]:
        r = binomtest(n_win, n_total, p=p0, alternative="two-sided")
        ci = r.proportion_ci(confidence_level=0.95, method="wilson")
        sig = "SIGNIFICANT" if not (ci.low <= p0 <= ci.high) else "not significant"
        print(f"  vs. {label} ({p0:.1%}): p={r.pvalue:.4f}  95% CI=[{ci.low:.1%}, {ci.high:.1%}]  -> {sig}")

    edge_vals = edge["edge"].to_numpy()
    boot_edge = np.array([
        edge_vals[RNG.integers(0, len(edge_vals), len(edge_vals))].mean() for _ in range(N_BOOT)
    ])
    lo, hi = np.percentile(boot_edge, [2.5, 97.5])
    sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant"
    print(f"\nMean edge (model - devigged market): {edge_vals.mean():+.4f}  "
          f"95% bootstrap CI=[{lo:+.4f}, {hi:+.4f}]  -> {sig}")

    y = edge["realized_over"].to_numpy()
    mp = edge["model_p_over"].to_numpy()
    kp = edge["market_p_over"].to_numpy()

    print(f"\n{'=' * 72}\n2. MODEL vs. MARKET -- calibration + discrimination\n{'=' * 72}")
    m_rel, k_rel = reliability(y, mp), reliability(y, kp)
    m_res, k_res = resolution(y, mp), resolution(y, kp)
    print(f"Reliability (lower=better calibrated): model={m_rel:.5f}  market={k_rel:.5f}  gap={m_rel - k_rel:+.5f}")
    lo, hi = bootstrap_gap(y, mp, kp, reliability)
    sig = "SIGNIFICANT -- model IS more miscalibrated than the market" if lo > 0 or hi < 0 else "not significant"
    print(f"  95% bootstrap CI on gap=[{lo:+.5f}, {hi:+.5f}]  -> {sig}")

    print(f"\nResolution (higher=more discriminating): model={m_res:.5f}  market={k_res:.5f}  gap(market-model)={k_res - m_res:+.5f}")
    lo, hi = bootstrap_gap(y, kp, mp, resolution)
    sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant"
    print(f"  95% bootstrap CI on gap=[{lo:+.5f}, {hi:+.5f}]  -> {sig}")

    print("\nModel calibration table (8 quantile bins):")
    print(get_calibration_df(y, mp, n_bins=8, min_n=0).to_string(index=False))

    # ── 3. Now that reliability shows a real, significant gap -- does
    # isotonic recalibration actually fix it, tested honestly (fit on one
    # half of THIS sample, eval on the other half, never both) ────────────
    print(f"\n{'=' * 72}\n3. IN-SAMPLE ISOTONIC RECALIBRATION CHECK (fit/eval split, n={len(edge)})\n{'=' * 72}")
    fit_df, eval_df = train_test_split(
        edge, train_size=0.5, random_state=42, stratify=edge["realized_over"],
    )
    print(f"RAW model, eval half (n={len(eval_df)}): reliability={reliability(eval_df['realized_over'], eval_df['model_p_over']):.5f}  "
          f"resolution={resolution(eval_df['realized_over'], eval_df['model_p_over']):.5f}")

    calibrator = fit_isotonic_calibrator(fit_df["realized_over"], fit_df["model_p_over"])
    eval_df = eval_df.copy()
    eval_df["model_p_over_calibrated"] = apply_calibration(calibrator, eval_df["model_p_over"])
    print(f"CALIBRATED model, eval half: reliability={reliability(eval_df['realized_over'], eval_df['model_p_over_calibrated']):.5f}  "
          f"resolution={resolution(eval_df['realized_over'], eval_df['model_p_over_calibrated']):.5f}")

    eval_df["model_favored_over_raw"] = eval_df["model_p_over"] > 0.5
    eval_df["model_favored_over_cal"] = eval_df["model_p_over_calibrated"] > 0.5
    eval_df["market_favored_over"] = eval_df["market_p_over"] > 0.5

    for label, favored_col in [("RAW", "model_favored_over_raw"), ("CALIBRATED", "model_favored_over_cal")]:
        d = eval_df[eval_df[favored_col] != eval_df["market_favored_over"]]
        if len(d) == 0:
            print(f"{label}: no disagreement starts in eval half")
            continue
        correct = (d[favored_col] == d["realized_over"].astype(bool))
        print(f"{label}: {len(d)} disagreement starts (eval half only), follow-model win rate: {correct.mean():.1%}")

    print(
        "\nCaveat: the fit/eval split here is at the pitcher-start level, not by date -- some "
        "temporal leakage is possible within a date (same-day starts share market conditions), "
        "though not across the season broadly. A date-level split would be the stricter version "
        "of this check if this result gets acted on."
    )


if __name__ == "__main__":
    main()
