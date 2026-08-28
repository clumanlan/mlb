# v4 Results — batters_faced_predictor (pitcher_anomaly_count_this_season)

**Date:** 2026-08-26  
**Task:** Regression (starting-pitcher-start grain)  
**Target:** realized_batters_faced  
**Primary metric:** MAE (lower = better)  

## Results (evaluated on val)

| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |
|-------|-----|---------------|------|------|-----------|
| Cascade (expected_batters_faced) | 2.8609 | — | 3.9509 | +0.0725 | 0.4949 |
| Linear regression (v4) | 2.7664 | -0.0945 | 3.7105 | +0.4447 | 0.5854 |
| XGBoost (v4, default) | 3.0896 | +0.2287 | 4.1183 | +0.7285 | 0.5004 |
| XGBoost (v4, tuned) | 2.6432 | -0.2177 | 3.6065 | +0.1802 | 0.6099 |
| (v2 tuned XGBoost, for reference) | 2.6471 | | | | |

## MAE/RMSE/Bias/Pearson r by expected_batters_faced_weight quartile

| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |
|---|---|---|---|---|---|---|
| Q1 (thinnest) | 1228 | Cascade (expected_batters_faced) | 3.3527 | 4.7167 | -0.5160 | 0.5252 |
| Q1 (thinnest) | 1228 | Linear regression (v4) | 3.1140 | 4.0808 | +0.5153 | 0.6761 |
| Q1 (thinnest) | 1228 | XGBoost (v4, default) | 3.1991 | 4.2994 | +0.2772 | 0.6389 |
| Q1 (thinnest) | 1228 | XGBoost (v4, tuned) | 2.9392 | 3.9025 | +0.4131 | 0.7079 |
| Q2 | 1253 | Cascade (expected_batters_faced) | 2.8050 | 3.7522 | +0.5751 | 0.5267 |
| Q2 | 1253 | Linear regression (v4) | 2.6624 | 3.5983 | +0.1950 | 0.5677 |
| Q2 | 1253 | XGBoost (v4, default) | 2.7504 | 3.6917 | +0.3010 | 0.5506 |
| Q2 | 1253 | XGBoost (v4, tuned) | 2.5608 | 3.4959 | +0.2099 | 0.5999 |
| Q3 | 1191 | Cascade (expected_batters_faced) | 2.6076 | 3.5060 | +0.3543 | 0.3021 |
| Q3 | 1191 | Linear regression (v4) | 2.5916 | 3.4858 | +0.4706 | 0.3246 |
| Q3 | 1191 | XGBoost (v4, default) | 2.6952 | 3.6159 | +0.3936 | 0.3059 |
| Q3 | 1191 | XGBoost (v4, tuned) | 2.4806 | 3.4025 | +0.0533 | 0.3601 |
| Q4 (most reliable) | 1114 | Cascade (expected_batters_faced) | 2.6524 | 3.6826 | -0.1452 | 0.2546 |
| Q4 (most reliable) | 1114 | Linear regression (v4) | 2.6870 | 3.6387 | +0.6203 | 0.3319 |
| Q4 (most reliable) | 1114 | XGBoost (v4, default) | 3.7720 | 4.8148 | +2.0651 | 0.1290 |
| Q4 (most reliable) | 1114 | XGBoost (v4, tuned) | 2.5835 | 3.6013 | +0.0260 | 0.3212 |

## bf_gap-quartile floor re-check

Overall: cascade bf_gap MAE 2.8609 | new estimate (XGBoost (v4, tuned)) bf_gap MAE 2.6432

| bf_gap quartile (by new estimate) | n | cascade_bf_gap MAE | new_bf_gap MAE |
|---|---|---|---|
| Q1 (closest) | 1197 | 1.0458 | 0.4529 |
| Q2 | 1196 | 1.7415 | 1.4713 |
| Q3 | 1196 | 2.8550 | 2.7474 |
| Q4 (furthest) | 1197 | 5.8003 | 5.9002 |

## Established-starter (11+ starts) bucket re-check

The specific failure mode this v4 pass targets (residual_error_analysis_v2's worst
over-prediction bucket — established starters pulled early despite a clean box line,
still open after v2's trend/rest/workload features only partially closed it).

n=2136 | cascade bf_gap MAE 2.6053 | XGBoost (v4, tuned) bf_gap MAE 2.5192

## Anomaly-count slice — has this pitcher already had a prior anomaly this season?

has prior anomaly (n=486): cascade MAE 3.0373 | XGBoost (v4, tuned) MAE 2.7477  
no prior anomaly (n=4300): cascade MAE 2.8409 | XGBoost (v4, tuned) MAE 2.6314

## Setup

- Features: ['pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts', 'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n', 'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start', 'expected_batters_faced', 'expected_batters_faced_weight', 'pitcher_throw_hand', 'team_roll_season_walk_rate', 'team_roll_season_on_base_rate', 'pitcher_team_days_since_last_game', 'is_home', 'pitcher_roll_season_pitch_count_avg', 'pitcher_last3_start_pa_avg_pa_per_start', 'pitcher_last3_start_pa_starts_n', 'pa_trend_ratio', 'pa_trend_direction', 'pitcher_days_since_last_start', 'pitcher_last_start_pitches', 'pitcher_workload_density', 'pitcher_workload_density_shrunk', 'pitcher_anomaly_count_this_season']
- New this pass: pitcher_anomaly_count_this_season (new
  game_context.build_pitcher_anomaly_count_this_season, TDD'd — a point-in-time-safe
  cumulative count of this pitcher's strictly-prior starts this season where
  realized_batters_faced < 0.6 * that start's own expected_batters_faced AND
  expected_batters_faced_weight >= 0.3 — the weight gate excludes cold-start noise,
  a separate problem v3 already investigated).
- Motivated by residual_error_analysis_v2/ + game_log_check.py: established-starter
  early-pulls are driven by 1-3 anomalous starts per pitcher-season (42-96% of that
  pitcher's own variance), not chronic volatility — rest days, weather, day-of-week,
  and opponent quality were checked and ruled out as shared pre-game-knowable triggers.
- XGBoost (v4, tuned) hyperparameters and its held-out-season early-stopping
  setup are carried over unchanged from v1/v2/v3.
