# v1 Results — batters_faced_predictor (opposing traffic + rest + home/away + pitch efficiency)

**Date:** 2026-08-26  
**Task:** Regression (starting-pitcher-start grain)  
**Target:** realized_batters_faced  
**Primary metric:** MAE (lower = better)  

## Results (evaluated on val)

| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |
|-------|-----|---------------|------|------|-----------|
| Cascade (expected_batters_faced) | 2.8609 | — | 3.9509 | +0.0725 | 0.4949 |
| Linear regression (v1) | 2.8965 | +0.0357 | 3.8650 | +0.5851 | 0.5413 |
| XGBoost (v1, default) | 3.2147 | +0.3538 | 4.2239 | +0.7559 | 0.4600 |
| XGBoost (v1, tuned) | 2.7347 | -0.1261 | 3.6881 | +0.2850 | 0.5874 |
| (baseline LR, for reference) | 2.9215 | | | | |
| (baseline XGBoost default, for reference) | 3.4655 | | | | |
| (baseline XGBoost tuned, for reference) | 2.7411 | | | | |

## MAE/RMSE/Bias/Pearson r by expected_batters_faced_weight quartile

| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |
|---|---|---|---|---|---|---|
| Q1 (thinnest) | 1228 | Cascade (expected_batters_faced) | 3.3527 | 4.7167 | -0.5160 | 0.5252 |
| Q1 (thinnest) | 1228 | Linear regression (v1) | 3.4662 | 4.4843 | +1.0423 | 0.6128 |
| Q1 (thinnest) | 1228 | XGBoost (v1, default) | 3.2707 | 4.3294 | +0.1228 | 0.6246 |
| Q1 (thinnest) | 1228 | XGBoost (v1, tuned) | 3.0835 | 4.0478 | +0.5462 | 0.6843 |
| Q2 | 1253 | Cascade (expected_batters_faced) | 2.8050 | 3.7522 | +0.5751 | 0.5267 |
| Q2 | 1253 | Linear regression (v1) | 2.6920 | 3.6317 | +0.2594 | 0.5596 |
| Q2 | 1253 | XGBoost (v1, default) | 2.8409 | 3.7722 | +0.3549 | 0.5157 |
| Q2 | 1253 | XGBoost (v1, tuned) | 2.6446 | 3.5653 | +0.2849 | 0.5803 |
| Q3 | 1191 | Cascade (expected_batters_faced) | 2.6076 | 3.5060 | +0.3543 | 0.3021 |
| Q3 | 1191 | Linear regression (v1) | 2.5670 | 3.4662 | +0.1251 | 0.3114 |
| Q3 | 1191 | XGBoost (v1, default) | 2.9367 | 3.8637 | +0.5320 | 0.2137 |
| Q3 | 1191 | XGBoost (v1, tuned) | 2.5627 | 3.4549 | +0.1978 | 0.3222 |
| Q4 (most reliable) | 1114 | Cascade (expected_batters_faced) | 2.6524 | 3.6826 | -0.1452 | 0.2546 |
| Q4 (most reliable) | 1114 | Linear regression (v1) | 2.8510 | 3.7854 | +0.9391 | 0.2634 |
| Q4 (most reliable) | 1114 | XGBoost (v1, default) | 3.8705 | 4.9013 | +2.1443 | 0.1026 |
| Q4 (most reliable) | 1114 | XGBoost (v1, tuned) | 2.6356 | 3.6492 | +0.0904 | 0.2806 |

## bf_gap-quartile floor re-check

Overall: cascade bf_gap MAE 2.8609 | new estimate (XGBoost (v1, tuned)) bf_gap MAE 2.7347

| bf_gap quartile (by new estimate) | n | cascade_bf_gap MAE | new_bf_gap MAE |
|---|---|---|---|
| Q1 (closest) | 1197 | 0.8786 | 0.4998 |
| Q2 | 1196 | 1.7484 | 1.5527 |
| Q3 | 1196 | 2.8584 | 2.8554 |
| Q4 (furthest) | 1197 | 5.9572 | 6.0301 |

## Setup

- Features: ['pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts', 'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n', 'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start', 'expected_batters_faced', 'expected_batters_faced_weight', 'pitcher_throw_hand', 'team_roll_season_walk_rate', 'team_roll_season_on_base_rate', 'pitcher_team_days_since_last_game', 'is_home', 'pitcher_roll_season_pitch_count_avg']
- New this pass: team_roll_season_walk_rate/_on_base_rate (opposing lineup,
  rolling_stats.build_team_batter_onbase_rolling_feats, new shared TDD'd
  function), pitcher_team_days_since_last_game (game_context.build_team_rest_days,
  reused unmodified), is_home (derived from schedule), 
  pitcher_roll_season_pitch_count_avg (pitch-efficiency proxy,
  rolling_stats.build_pbp_pitcher_rolling_feats, reused unmodified).
- XGBoost (v1, tuned) hyperparameters and its held-out-season
  early-stopping setup are carried over unchanged from
  experiments/xgb_vs_cascade_diagnostic/run.py and the re-run baseline —
  see ROADMAP.md's 2026-08-26 follow-up entry for the overfitting
  diagnosis that motivated tuning XGBoost in the first place.
