# Baseline Results — batters_faced_predictor

**Date:** 2026-08-26  
**Task:** Regression (starting-pitcher-start grain) — the first regression
target in this repo's model layer (every sibling model predicts a binary
outcome).  
**Target:** realized_batters_faced  
**Primary metric:** MAE (lower = better)  
**Diagnostics:** RMSE, Bias (signed mean error), Pearson r  
**Data:** s3://mlbdk  

## Split

| Split | Seasons | Rows | Mean realized_batters_faced |
|-------|---------|------|------------------------------|
| Train | [2018, 2019, 2022, 2023] | 18,844 | 22.06 |
| Val   | 2024 | 4,786 | 22.07 |
| Test  | 2025 | 4,768 | locked — not evaluated here |

## Results (evaluated on val)

The floor is NOT a global-mean naive — it's the already-shipped shrinkage
cascade (`game_context.build_expected_batters_faced`), included here as a
benchmark row rather than a trained model. Every candidate model is fed the
SAME inputs the cascade formula already has access to — this baseline asks
whether a regressor combining those inputs non-linearly beats the
hand-built shrinkage formula, before any new feature engineering.

| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |
|-------|-----|---------------|------|------|-----------|
| Cascade (expected_batters_faced) | 2.8609 | — | 3.9509 | +0.0725 | 0.4949 |
| Linear regression | 2.9215 | +0.0607 | 3.8864 | +0.6408 | 0.5369 |
| XGBoost (default) | 3.4655 | +0.6046 | 4.5556 | +1.2002 | 0.4062 |
| XGBoost (tuned) | 2.7411 | -0.1198 | 3.7045 | +0.2290 | 0.5814 |

## Interpretation

**XGBoost (tuned)** beats the cascade floor (**Cascade (expected_batters_faced)**, MAE 2.8609) with MAE 2.7411 — worth carrying forward into a real experiment.

## MAE/RMSE/Bias/Pearson r by expected_batters_faced_weight quartile

Same stratification as the cascade's own Story-0 baseline capture (ROADMAP.md)
— Q1 (thinnest this-season sample) through Q4 (most reliable).

| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |
|---|---|---|---|---|---|---|
| Q1 (thinnest) | 1228 | Cascade (expected_batters_faced) | 3.3527 | 4.7167 | -0.5160 | 0.5252 |
| Q1 (thinnest) | 1228 | Linear regression | 3.4945 | 4.4994 | +1.0516 | 0.6097 |
| Q1 (thinnest) | 1228 | XGBoost (default) | 3.2198 | 4.2327 | +0.2069 | 0.6426 |
| Q1 (thinnest) | 1228 | XGBoost (tuned) | 3.0941 | 4.0874 | +0.4263 | 0.6749 |
| Q2 | 1253 | Cascade (expected_batters_faced) | 2.8050 | 3.7522 | +0.5751 | 0.5267 |
| Q2 | 1253 | Linear regression | 2.7171 | 3.6572 | +0.3385 | 0.5525 |
| Q2 | 1253 | XGBoost (default) | 2.9121 | 3.8759 | +0.4951 | 0.4877 |
| Q2 | 1253 | XGBoost (tuned) | 2.6441 | 3.5639 | +0.2226 | 0.5798 |
| Q3 | 1191 | Cascade (expected_batters_faced) | 2.6076 | 3.5060 | +0.3543 | 0.3021 |
| Q3 | 1191 | Linear regression | 2.5822 | 3.4831 | +0.1980 | 0.3023 |
| Q3 | 1191 | XGBoost (default) | 3.1468 | 4.0745 | +0.9243 | 0.1709 |
| Q3 | 1191 | XGBoost (tuned) | 2.5670 | 3.4648 | +0.1768 | 0.3130 |
| Q4 (most reliable) | 1114 | Cascade (expected_batters_faced) | 2.6524 | 3.6826 | -0.1452 | 0.2546 |
| Q4 (most reliable) | 1114 | Linear regression | 2.8827 | 3.8157 | +1.0014 | 0.2502 |
| Q4 (most reliable) | 1114 | XGBoost (default) | 4.6996 | 5.8965 | +3.3833 | 0.0900 |
| Q4 (most reliable) | 1114 | XGBoost (tuned) | 2.6473 | 3.6637 | +0.0745 | 0.2670 |

## Setup

- Features: ['pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts', 'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n', 'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start', 'expected_batters_faced', 'expected_batters_faced_weight', 'pitcher_throw_hand']
- No new feature engineering — every feature here is an input the
  expected_batters_faced shrinkage cascade itself already uses
  (pitcher/team/league last-season avg PA/start, this-season rolling avg
  PA/start + starts_n, the cascade's own point estimate + weight,
  handedness). New candidate features (opposing-lineup on-base rate, rest
  days, home/away, pitch-efficiency trend) are deliberately deferred to v1.
- Grain: one row per (personId, gamepk) starting-pitcher start — same as
  short_outing_predictor, different from every PA/batter-game-grain
  sibling. Scoped to REALIZED pitcher_role == 'sp' by construction.
- Label: realized_batters_faced,
  batters_faced_predictor.processing.pipeline.create_start_pa_outcome.
- build_expected_batters_faced / build_pitcher_start_pa_this_season are
  NOT modified — called as-is, same reasoning that already keeps
  build_expected_start_innings frozen for its own dependents.
- XGBoost (tuned) hyperparameters and its held-out-season early-stopping
  setup are carried over unchanged from
  `experiments/xgb_vs_cascade_diagnostic/run.py`, which diagnosed default
  XGBoost as overfitting at this data size (train MAE far below val MAE;
  worked examples showed it predicting implausibly low batters-faced
  counts for established starters) — see ROADMAP.md's 2026-08-26
  follow-up entry.

## Plots

- `plots/baseline-model/target_distribution.png`
- `plots/baseline-model/residuals.png` — best model's predicted vs. actual
- `plots/baseline-model/feature_importance.png` — XGBoost (tuned)

## Next steps

- If a model beats the cascade floor: move to a real v1 experiment, add
  opposing-lineup on-base/walk rate (`rolling_stats.build_team_batter_onbase_rolling_feats`),
  team rest days (`game_context.build_team_rest_days`), home/away, and a
  pitch-efficiency proxy — and carry the tuned XGBoost hyperparameters
  forward rather than reverting to defaults.
- If not: the cascade's inputs alone, recombined, aren't enough — the same
  new-feature set above is still the next step, just with lower prior odds
  of a quick win from recombination alone.
- Re-run the bf_gap-quartile floor-isolation check
  (k_predictor/experiments/count_distribution_check/run.py's method,
  documented in BENCHMARKS.md) with whichever estimate wins, substituted
  for expected_batters_faced, to see whether the closest-quartile floor
  MAE itself moves.
- Final evaluation on test season (2025) only once in a real experiment.
