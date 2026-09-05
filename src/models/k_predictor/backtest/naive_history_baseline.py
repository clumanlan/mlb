"""
k_predictor -- naive floor for GAME-TOTAL strikeouts: does v6's aggregated
PA-model prediction (predicted_mean_k, the poisson-binomial pmf's mean, from
pred_df_test2025.parquet) actually beat just reading off a pitcher's own
season-to-date strikeout history?

This is the naive baseline betting actually uses -- "this guy's averaged 6 Ks
a start this year, take the line at 5.5" -- and it did not previously exist
anywhere in this project; every prior "naive" floor here (baseline/model/run.py,
v2-v4's DummyClassifier/per-role-rate floors) operates at the PA-classification
grain and gets aggregated through the model's own combination logic, not read
directly off a pitcher's individual game log. See ROADMAP.md's 2026-09-03
entry.

Point-in-time safety: for each of the 196 backtest starts in
pred_df_test2025.parquet, the naive mean/median is computed from that same
pitcher's 2025 starts STRICTLY BEFORE that game's date only (expanding
window), using diagnostics/game_log.csv (already covers the full 2025
season, built by diagnostics/build_game_log.py). True cold starts (zero
prior 2025 starts -- unavoidable for a pitcher's first outing of the season)
are excluded from the naive comparison and reported separately, same
convention as this project's other cold-start handling (e.g. FEATURE_SEASONS
carrying prior-season rate stats forward).

Run from src/models/k_predictor/ with:
    python backtest/naive_history_baseline.py
No AWS credentials needed -- reads pred_df_test2025.parquet and
../diagnostics/game_log.csv, both already on disk.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BACKTEST_DIR = Path(__file__).parent
K_PREDICTOR_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BACKTEST_DIR / "pred_df_test2025.parquet"
GAME_LOG_PATH = K_PREDICTOR_DIR / "diagnostics" / "game_log.csv"
MIN_PRIOR_STARTS = 1  # below this, "naive" has nothing to read off -- exclude, don't impute


def mae(a, b):
    return float(np.abs(a - b).mean())


def main():
    pred = pd.read_parquet(PRED_PATH)[["gamepk", "personId", "player_name", "game_date", "predicted_mean_k", "realized_k"]].copy()
    pred["gamepk"] = pred["gamepk"].astype(str)
    pred["personId"] = pred["personId"].astype(str)

    game_log = pd.read_csv(GAME_LOG_PATH, parse_dates=["game_date"])
    game_log["gamepk"] = game_log["gamepk"].astype(str)
    game_log["pitcher_id"] = game_log["pitcher_id"].astype(str)

    league_mean_k = game_log["strikeouts"].mean()

    rows = []
    for _, r in pred.iterrows():
        prior = game_log[
            (game_log["pitcher_id"] == r["personId"]) & (game_log["game_date"] < r["game_date"])
        ]
        n_prior = len(prior)
        rows.append({
            "gamepk": r["gamepk"], "personId": r["personId"], "player_name": r["player_name"],
            "game_date": r["game_date"], "realized_k": r["realized_k"],
            "model_predicted_mean_k": r["predicted_mean_k"],
            "n_prior_starts": n_prior,
            "naive_mean_k": prior["strikeouts"].mean() if n_prior >= MIN_PRIOR_STARTS else np.nan,
            "naive_median_k": prior["strikeouts"].median() if n_prior >= MIN_PRIOR_STARTS else np.nan,
        })
    df = pd.DataFrame(rows)
    df["naive_league_mean_k"] = league_mean_k

    n_total = len(df)
    scoreable = df.dropna(subset=["naive_mean_k"]).copy()
    n_cold_start = n_total - len(scoreable)

    print(f"{'=' * 72}\nNAIVE PITCHER-HISTORY BASELINE vs. v6 -- 2025 backtest ({n_total} starts)\n{'=' * 72}")
    print(f"Excluded (zero prior 2025 starts, true cold start): {n_cold_start}")
    print(f"Scored on: {len(scoreable)} starts\n")

    for label, col in [
        ("Naive: pitcher's own season-to-date MEAN K", "naive_mean_k"),
        ("Naive: pitcher's own season-to-date MEDIAN K", "naive_median_k"),
        ("Naive: league-wide mean K/start (no pitcher info at all)", "naive_league_mean_k"),
        ("v6 model: predicted_mean_k (PA-level XGBoost -> poisson-binomial pmf)", "model_predicted_mean_k"),
    ]:
        m = mae(scoreable["realized_k"], scoreable[col])
        bias = float((scoreable[col] - scoreable["realized_k"]).mean())
        print(f"  {label:70s}  MAE={m:.4f}  bias={bias:+.4f}")

    print(f"\n{'=' * 72}\nBy prior-start count (does the naive floor need warm-up?)\n{'=' * 72}")
    for lo, hi, label in [(1, 4, "1-3 prior starts"), (4, 8, "4-7 prior starts"), (8, 99, "8+ prior starts")]:
        bucket = scoreable[(scoreable["n_prior_starts"] >= lo) & (scoreable["n_prior_starts"] < hi)]
        if len(bucket) == 0:
            continue
        m_naive = mae(bucket["realized_k"], bucket["naive_mean_k"])
        m_model = mae(bucket["realized_k"], bucket["model_predicted_mean_k"])
        print(f"  {label:20s} n={len(bucket):3d}  naive_mean MAE={m_naive:.4f}  model MAE={m_model:.4f}")

    print(f"\n{'=' * 72}\nBOOTSTRAP CI -- MAE gap (naive_mean - model), 10,000 resamples\n{'=' * 72}")
    rng = np.random.default_rng(42)
    n = len(scoreable)
    realized = scoreable["realized_k"].to_numpy()
    naive_vals = scoreable["naive_mean_k"].to_numpy()
    model_vals = scoreable["model_predicted_mean_k"].to_numpy()
    gaps = np.empty(10_000)
    for i in range(10_000):
        idx = rng.integers(0, n, size=n)
        gaps[i] = mae(realized[idx], naive_vals[idx]) - mae(realized[idx], model_vals[idx])
    point_gap = mae(realized, naive_vals) - mae(realized, model_vals)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"Point estimate (naive MAE - model MAE): {point_gap:+.4f}")
    print(f"95% bootstrap CI: [{lo:+.4f}, {hi:+.4f}]"
          f"{'  -> excludes 0, model edge looks real' if lo > 0 or hi < 0 else '  -> includes 0, NOT distinguishable from noise at n=' + str(n)}")

    out_path = BACKTEST_DIR / "naive_history_baseline_2025.parquet"
    df.to_parquet(out_path)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
