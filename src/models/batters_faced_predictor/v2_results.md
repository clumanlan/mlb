# v2 Results — batters_faced_predictor (trailing-3-start PA trend + rest + workload density)

**Date:** 2026-08-26  
**Task:** Regression (starting-pitcher-start grain)  
**Target:** realized_batters_faced  
**Primary metric:** MAE (lower = better)  

## Results (evaluated on val)

| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |
|-------|-----|---------------|------|------|-----------|
| Cascade (expected_batters_faced) | 2.8609 | — | 3.9509 | +0.0725 | 0.4949 |
| Linear regression (v2) | 2.7654 | -0.0955 | 3.7110 | +0.4458 | 0.5853 |
| XGBoost (v2, default) | 3.0404 | +0.1796 | 4.0538 | +0.6839 | 0.5122 |
| XGBoost (v2, tuned) | 2.6471 | -0.2138 | 3.6092 | +0.1482 | 0.6084 |
| (v1 tuned XGBoost, for reference) | 2.7347 | | | | |

## MAE/RMSE/Bias/Pearson r by expected_batters_faced_weight quartile

| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |
|---|---|---|---|---|---|---|
| Q1 (thinnest) | 1228 | Cascade (expected_batters_faced) | 3.3527 | 4.7167 | -0.5160 | 0.5252 |
| Q1 (thinnest) | 1228 | Linear regression (v2) | 3.1135 | 4.0826 | +0.5108 | 0.6756 |
| Q1 (thinnest) | 1228 | XGBoost (v2, default) | 3.1107 | 4.1568 | +0.2662 | 0.6623 |
| Q1 (thinnest) | 1228 | XGBoost (v2, tuned) | 2.9555 | 3.9163 | +0.3937 | 0.7052 |
| Q2 | 1253 | Cascade (expected_batters_faced) | 2.8050 | 3.7522 | +0.5751 | 0.5267 |
| Q2 | 1253 | Linear regression (v2) | 2.6599 | 3.5964 | +0.1856 | 0.5682 |
| Q2 | 1253 | XGBoost (v2, default) | 2.7439 | 3.7210 | +0.2897 | 0.5424 |
| Q2 | 1253 | XGBoost (v2, tuned) | 2.5636 | 3.4983 | +0.1630 | 0.5984 |
| Q3 | 1191 | Cascade (expected_batters_faced) | 2.6076 | 3.5060 | +0.3543 | 0.3021 |
| Q3 | 1191 | Linear regression (v2) | 2.5877 | 3.4826 | +0.4703 | 0.3264 |
| Q3 | 1191 | XGBoost (v2, default) | 2.7365 | 3.6642 | +0.4479 | 0.2946 |
| Q3 | 1191 | XGBoost (v2, tuned) | 2.4815 | 3.4018 | +0.0462 | 0.3587 |
| Q4 (most reliable) | 1114 | Cascade (expected_batters_faced) | 2.6524 | 3.6826 | -0.1452 | 0.2546 |
| Q4 (most reliable) | 1114 | Linear regression (v2) | 2.6904 | 3.6440 | +0.6406 | 0.3307 |
| Q4 (most reliable) | 1114 | XGBoost (v2, default) | 3.6214 | 4.6505 | +1.8401 | 0.1242 |
| Q4 (most reliable) | 1114 | XGBoost (v2, tuned) | 2.5779 | 3.5944 | -0.0300 | 0.3252 |

## bf_gap-quartile floor re-check

Overall: cascade bf_gap MAE 2.8609 | new estimate (XGBoost (v2, tuned)) bf_gap MAE 2.6471

| bf_gap quartile (by new estimate) | n | cascade_bf_gap MAE | new_bf_gap MAE |
|---|---|---|---|
| Q1 (closest) | 1197 | 1.0493 | 0.4591 |
| Q2 | 1196 | 1.7268 | 1.4811 |
| Q3 | 1196 | 2.8472 | 2.7454 |
| Q4 (furthest) | 1197 | 5.8193 | 5.9018 |

## Established-starter (11+ starts) bucket re-check

The specific failure mode this v2 pass targets (residual_error_analysis's worst
over-prediction bucket — established starters pulled early despite a clean box line).

n=2136 | cascade bf_gap MAE 2.6053 | XGBoost (v2, tuned) bf_gap MAE 2.5154

## Setup

- Features: ['pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts', 'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n', 'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start', 'expected_batters_faced', 'expected_batters_faced_weight', 'pitcher_throw_hand', 'team_roll_season_walk_rate', 'team_roll_season_on_base_rate', 'pitcher_team_days_since_last_game', 'is_home', 'pitcher_roll_season_pitch_count_avg', 'pitcher_last3_start_pa_avg_pa_per_start', 'pitcher_last3_start_pa_starts_n', 'pa_trend_ratio', 'pa_trend_direction', 'pitcher_days_since_last_start', 'pitcher_last_start_pitches', 'pitcher_workload_density', 'pitcher_workload_density_shrunk']
- New this pass: pitcher_last3_start_pa_avg_pa_per_start/_starts_n + pa_trend_ratio/
  pa_trend_direction (game_context.build_pitcher_start_pa_this_season(pbp, window=3),
  already existed — just not previously called with window=3 in this pipeline),
  pitcher_days_since_last_start (new game_context.build_pitcher_rest_days, TDD'd),
  pitcher_last_start_pitches / pitcher_workload_density (new
  game_context.build_pitcher_workload_density, TDD'd), pitcher_workload_density_shrunk
  (glue in this file, same shrinkage_weight = starts_n/(starts_n+k) formula
  build_expected_batters_faced already uses inline).
- XGBoost (v2, tuned) hyperparameters and its held-out-season early-stopping
  setup are carried over unchanged from v1_opposing_traffic_and_rest/train.py.
