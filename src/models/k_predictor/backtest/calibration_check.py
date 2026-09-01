"""
k_predictor production backtest -- calibration confound check (ROADMAP.md's
Near-term backlog (k_predictor) item 6(a)).

edge_report.py's first real-2025-odds backtest (2026-08-31) found v6's
"follow the model on disagreement" win rate at 42.2% -- below a coin flip.
That backtest used v6's raw, uncalibrated P(over line); the open question
this script answers is whether an uncalibrated aggregated Poisson-binomial
probability is what's producing the negative-leaning result, or whether
calibration isn't the problem at all.

Two checks, in order:

1. Out-of-sample calibration check on the LARGE held-out sample that already
   exists -- score_2025_test_dates.py's val-season sibling script
   (run_xgboost_uncertainty.py) already scored 4786 2024 SP starts at a
   fixed synthetic line (5.5) and saved p_over_line/realized_over_line.
   Split that in half, fit an isotonic calibrator on one half (reusing
   hit_predictor/utils/calibration.py's generic fit_isotonic_calibrator --
   it operates on plain y_true/y_prob arrays, nothing hit_predictor-specific
   despite the module's name), and check whether calibration error
   (reliability) improves on the OTHER half.
2. Apply that same calibrator (now fit on the full 4786-start sample) to
   the real 131-start backtest's model_p_over, and recompute edge_report.py's
   headline "follow the model on disagreement" win rate. If mis-calibration
   were the confound, recalibrating should move that number.

Caveat: the calibrator is fit at one fixed line (5.5) on 2024 data and
applied to varying real 2025 DK lines -- this assumes the model's
miscalibration pattern (if any) is a function of predicted probability, not
of the specific line value. Standard isotonic-recalibration practice, but an
assumption, not a proof.

Run from src/models/k_predictor/ with:
    python backtest/calibration_check.py
No AWS credentials needed -- everything reads from already-saved local parquet.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))
from models.hit_predictor.utils.eval import get_calibration_df, expected_calibration_error, murphy_decomposition
from models.hit_predictor.utils.calibration import fit_isotonic_calibrator, apply_calibration

BACKTEST_DIR = Path(__file__).parent
VAL_PRED_DF = BACKTEST_DIR.parent / "experiments/count_distribution_check/xgboost_uncertainty/pred_df.parquet"
EDGE_REPORT = BACKTEST_DIR / "edge_report.parquet"


def calib_report(y_true, y_prob, label, n_bins=10):
    cal_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=0)
    m = murphy_decomposition(y_true, cal_df)
    ece = expected_calibration_error(cal_df)
    print(f"{label}: reliability={m['reliability']:.5f}  resolution={m['resolution']:.5f}  "
          f"ece={ece:.5f}  brier={m['brier_reconstructed']:.5f}")
    return m, ece


def main():
    val = pd.read_parquet(VAL_PRED_DF)
    print(f"Held-out VAL_SEASON sample (2024, fixed line 5.5): {len(val):,} SP starts, "
          f"base rate P(over)={val['realized_over_line'].mean():.3f}")

    # ── 1. Out-of-sample calibration check ─────────────────────────────────
    fit_df, eval_df = train_test_split(
        val, train_size=0.5, random_state=42, stratify=val["realized_over_line"],
    )
    print(f"\n{'=' * 72}\nOUT-OF-SAMPLE CALIBRATION CHECK (fit on half, eval on other half, n={len(eval_df)})\n{'=' * 72}")
    calib_report(eval_df["realized_over_line"], eval_df["p_over_line"], "raw (uncalibrated)")
    calibrator_half = fit_isotonic_calibrator(fit_df["realized_over_line"], fit_df["p_over_line"])
    calibrated_eval = apply_calibration(calibrator_half, eval_df["p_over_line"])
    calib_report(eval_df["realized_over_line"], calibrated_eval, "isotonic-calibrated")

    # ── 2. Apply a calibrator fit on the FULL held-out sample to the real
    # 131-start backtest, and recompute the disagreement win-rate that
    # edge_report.py's headline result was built on ──────────────────────────
    full_calibrator = fit_isotonic_calibrator(val["realized_over_line"], val["p_over_line"])

    edge = pd.read_parquet(EDGE_REPORT)
    edge["model_p_over_calibrated"] = apply_calibration(full_calibrator, edge["model_p_over"])
    edge["edge_calibrated"] = edge["model_p_over_calibrated"] - edge["market_p_over"]
    edge["model_favored_over"] = edge["model_p_over"] > 0.5
    edge["model_favored_over_calibrated"] = edge["model_p_over_calibrated"] > 0.5
    edge["market_favored_over"] = edge["market_p_over"] > 0.5
    edge["disagree_raw"] = edge["model_favored_over"] != edge["market_favored_over"]
    edge["disagree_calibrated"] = edge["model_favored_over_calibrated"] != edge["market_favored_over"]

    print(f"\n{'=' * 72}\nAPPLYING THE VAL_SEASON-FITTED CALIBRATOR TO THE 131-START REAL BACKTEST\n{'=' * 72}")
    print(f"Mean raw edge:        {edge['edge'].mean():+.4f}")
    print(f"Mean calibrated edge: {edge['edge_calibrated'].mean():+.4f}")

    disagree_raw = edge[edge["disagree_raw"]]
    follow_raw = (disagree_raw["model_favored_over"] == disagree_raw["realized_over"].astype(bool))
    print(f"\nRAW: {len(disagree_raw)} disagreement starts, follow-model win rate: {follow_raw.mean():.1%}")

    disagree_cal = edge[edge["disagree_calibrated"]]
    follow_cal = (disagree_cal["model_favored_over_calibrated"] == disagree_cal["realized_over"].astype(bool))
    print(f"CALIBRATED: {len(disagree_cal)} disagreement starts, follow-model win rate: {follow_cal.mean():.1%}")

    # ── 3. Model vs. market calibration+resolution head-to-head on the same
    # 131-row real-backtest sample -- distinguishes "model is miscalibrated"
    # from "model is just less discriminating than the market" ────────────
    print(f"\n{'=' * 72}\nMODEL vs. MARKET on the 131-start real backtest (same y_true, 5 bins)\n{'=' * 72}")
    calib_report(edge["realized_over"], edge["model_p_over"], "model (raw)", n_bins=5)
    calib_report(edge["realized_over"], edge["market_p_over"], "market (devigged)", n_bins=5)

    print(
        "\nCaveat: n=131 real-odds starts (7 days) and n=4786 synthetic-line-only starts "
        "(no real odds attached) are both far short of a real verdict -- see edge_report.py's "
        "own caveat. This script answers one narrower question: does recalibration change the "
        "disagreement win rate? See ROADMAP.md's Near-term backlog item 6 for the conclusion."
    )

    out_path = BACKTEST_DIR / "edge_report_calibrated.parquet"
    edge.to_parquet(out_path, index=False)
    print(f"\nWrote {len(edge)} rows to {out_path}")


if __name__ == "__main__":
    main()
