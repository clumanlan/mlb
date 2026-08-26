# v3 Results — batters_faced_predictor (multi-season lookback, cold start)

**Date:** 2026-08-26  
**Task:** Regression (starting-pitcher-start grain)  
**Target:** realized_batters_faced  
**Primary metric:** MAE (lower = better)  

## Results (evaluated on val)

| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |
|-------|-----|---------------|------|------|-----------|
| Cascade (expected_batters_faced) | 2.8609 | — | 3.9509 | +0.0725 | 0.4949 |
| Linear regression (v3) | 2.7668 | -0.0941 | 3.7115 | +0.4508 | 0.5853 |
| XGBoost (v3, default) | 3.0295 | +0.1686 | 4.0290 | +0.6718 | 0.5199 |
| XGBoost (v3, tuned) | 2.6479 | -0.2130 | 3.6081 | +0.1797 | 0.6095 |
| (v2 tuned XGBoost, for reference) | 2.6471 | | | | |

## MAE/RMSE/Bias/Pearson r by expected_batters_faced_weight quartile

| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |
|---|---|---|---|---|---|---|
| Q1 (thinnest) | 1228 | Cascade (expected_batters_faced) | 3.3527 | 4.7167 | -0.5160 | 0.5252 |
| Q1 (thinnest) | 1228 | Linear regression (v3) | 3.1128 | 4.0820 | +0.5135 | 0.6758 |
| Q1 (thinnest) | 1228 | XGBoost (v3, default) | 3.1460 | 4.2117 | +0.3403 | 0.6578 |
| Q1 (thinnest) | 1228 | XGBoost (v3, tuned) | 2.9462 | 3.9111 | +0.4104 | 0.7062 |
| Q2 | 1253 | Cascade (expected_batters_faced) | 2.8050 | 3.7522 | +0.5751 | 0.5267 |
| Q2 | 1253 | Linear regression (v3) | 2.6631 | 3.5993 | +0.1904 | 0.5674 |
| Q2 | 1253 | XGBoost (v3, default) | 2.7672 | 3.7057 | +0.2986 | 0.5458 |
| Q2 | 1253 | XGBoost (v3, tuned) | 2.5666 | 3.4988 | +0.1958 | 0.5988 |
| Q3 | 1191 | Cascade (expected_batters_faced) | 2.6076 | 3.5060 | +0.3543 | 0.3021 |
| Q3 | 1191 | Linear regression (v3) | 2.5877 | 3.4831 | +0.4769 | 0.3269 |
| Q3 | 1191 | XGBoost (v3, default) | 2.6928 | 3.5988 | +0.2877 | 0.2970 |
| Q3 | 1191 | XGBoost (v3, tuned) | 2.4859 | 3.4023 | +0.0605 | 0.3604 |
| Q4 (most reliable) | 1114 | Cascade (expected_batters_faced) | 2.6524 | 3.6826 | -0.1452 | 0.2546 |
| Q4 (most reliable) | 1114 | Linear regression (v3) | 2.6935 | 3.6431 | +0.6465 | 0.3323 |
| Q4 (most reliable) | 1114 | XGBoost (v3, default) | 3.5560 | 4.5709 | +1.8676 | 0.1404 |
| Q4 (most reliable) | 1114 | XGBoost (v3, tuned) | 2.5835 | 3.5951 | +0.0347 | 0.3262 |

## bf_gap-quartile floor re-check

Overall: cascade bf_gap MAE 2.8609 | new estimate (XGBoost (v3, tuned)) bf_gap MAE 2.6479

| bf_gap quartile (by new estimate) | n | cascade_bf_gap MAE | new_bf_gap MAE |
|---|---|---|---|
| Q1 (closest) | 1197 | 1.0278 | 0.4633 |
| Q2 | 1196 | 1.7499 | 1.4744 |
| Q3 | 1196 | 2.8723 | 2.7507 |
| Q4 (furthest) | 1197 | 5.7925 | 5.9022 |

## Cold-start (0-2 starts this season) bucket re-check

The specific failure mode this v3 pass targets (residual_error_analysis's worst
under-prediction bucket).

n=969 | cascade bf_gap MAE 3.4891 | XGBoost (v3, tuned) bf_gap MAE 3.1069

## Setup

- Features: ['pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts', 'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n', 'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start', 'expected_batters_faced', 'expected_batters_faced_weight', 'pitcher_throw_hand', 'team_roll_season_walk_rate', 'team_roll_season_on_base_rate', 'pitcher_team_days_since_last_game', 'is_home', 'pitcher_roll_season_pitch_count_avg', 'pitcher_last3_start_pa_avg_pa_per_start', 'pitcher_last3_start_pa_starts_n', 'pa_trend_ratio', 'pa_trend_direction', 'pitcher_days_since_last_start', 'pitcher_last_start_pitches', 'pitcher_workload_density', 'pitcher_workload_density_shrunk', 'pitcher_last_known_season_start_pa_avg_pa_per_start', 'pitcher_last_known_season_start_pa_n_starts']
- New this pass: pitcher_last_known_season_start_pa_avg_pa_per_start/_n_starts
  (new season_stats.build_pitcher_last_known_season_start_pa, TDD'd) — that
  pitcher's stats from the most recent PRIOR season with starts, not necessarily
  the immediately-prior one (build_pitcher_start_pa_stats, frozen, only ever
  looks exactly one season back).
- XGBoost (v3, tuned) hyperparameters and its held-out-season early-stopping
  setup are carried over unchanged from v1/v2.
