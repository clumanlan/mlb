# v6 Results — batters_faced_predictor (opposing-team scoring volatility)

**Date:** 2026-08-26  
**Task:** Regression (starting-pitcher-start grain)  
**Target:** realized_batters_faced  
**Primary metric:** MAE (lower = better)  

## Results (evaluated on val)

| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |
|-------|-----|---------------|------|------|-----------|
| Cascade (expected_batters_faced) | 2.8609 | — | 3.9509 | +0.0725 | 0.4949 |
| Linear regression (v6) | 2.7740 | -0.0869 | 3.7046 | +0.5167 | 0.5902 |
| XGBoost (v6, default) | 3.0462 | +0.1854 | 4.0337 | +0.7617 | 0.5248 |
| XGBoost (v6, tuned) | 2.6349 | -0.2260 | 3.5947 | +0.1033 | 0.6123 |
| (v2 tuned XGBoost, for reference) | 2.6471 | | | | |

## MAE/RMSE/Bias/Pearson r by expected_batters_faced_weight quartile

| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |
|---|---|---|---|---|---|---|
| Q1 (thinnest) | 1228 | Cascade (expected_batters_faced) | 3.3527 | 4.7167 | -0.5160 | 0.5252 |
| Q1 (thinnest) | 1228 | Linear regression (v6) | 3.1107 | 4.0516 | +0.6052 | 0.6843 |
| Q1 (thinnest) | 1228 | XGBoost (v6, default) | 3.1736 | 4.1918 | +0.4417 | 0.6615 |
| Q1 (thinnest) | 1228 | XGBoost (v6, tuned) | 2.9027 | 3.8717 | +0.2442 | 0.7109 |
| Q2 | 1253 | Cascade (expected_batters_faced) | 2.8050 | 3.7522 | +0.5751 | 0.5267 |
| Q2 | 1253 | Linear regression (v6) | 2.6824 | 3.5988 | +0.2852 | 0.5688 |
| Q2 | 1253 | XGBoost (v6, default) | 2.7724 | 3.7015 | +0.3878 | 0.5492 |
| Q2 | 1253 | XGBoost (v6, tuned) | 2.5526 | 3.4808 | +0.0934 | 0.6030 |
| Q3 | 1191 | Cascade (expected_batters_faced) | 2.6076 | 3.5060 | +0.3543 | 0.3021 |
| Q3 | 1191 | Linear regression (v6) | 2.5896 | 3.4846 | +0.5000 | 0.3342 |
| Q3 | 1191 | XGBoost (v6, default) | 2.6761 | 3.5883 | +0.4083 | 0.3285 |
| Q3 | 1191 | XGBoost (v6, tuned) | 2.4847 | 3.4077 | -0.0194 | 0.3596 |
| Q4 (most reliable) | 1114 | Cascade (expected_batters_faced) | 2.6524 | 3.6826 | -0.1452 | 0.2546 |
| Q4 (most reliable) | 1114 | Linear regression (v6) | 2.7032 | 3.6491 | +0.6975 | 0.3347 |
| Q4 (most reliable) | 1114 | XGBoost (v6, default) | 3.6097 | 4.6215 | +1.9128 | 0.1539 |
| Q4 (most reliable) | 1114 | XGBoost (v6, tuned) | 2.5929 | 3.5984 | +0.0902 | 0.3268 |

## bf_gap-quartile floor re-check

Overall: cascade bf_gap MAE 2.8609 | new estimate (XGBoost (v6, tuned)) bf_gap MAE 2.6349

| bf_gap quartile (by new estimate) | n | cascade_bf_gap MAE | new_bf_gap MAE |
|---|---|---|---|
| Q1 (closest) | 1197 | 1.1154 | 0.4693 |
| Q2 | 1196 | 1.7182 | 1.4757 |
| Q3 | 1196 | 2.8530 | 2.7279 |
| Q4 (furthest) | 1197 | 5.7560 | 5.8659 |

## Established-starter (11+ starts) bucket re-check

Same slice v2/v3/v4 check against — this pass does not specifically target
that failure mode (that thread is closed, see ROADMAP.md's v4 entry), so this
is a sanity re-check for regressions, not the headline result.

n=2136 | cascade bf_gap MAE 2.6053 | XGBoost (v6, tuned) bf_gap MAE 2.5238

## Feature importance (XGBoost, v6, tuned)

None of the six new volatility features rank near the top. `opp_team_roll_season_runs_scored_max`
lands at #13 of 33 — the best-placed of the six, but well below v5's `opp_team_roll_season_runs_scored`
(level, unchanged from v5, still #11). `opp_team_roll_season_runs_scored_mean` and `_std`, and
all three `opp_team_roll_last5g_runs_scored_*` columns, rank in the bottom third alongside v5's
weakest last5g/on-base features. The top of the ranking is unchanged from v2/v5:
`pitcher_last3_start_pa_avg_pa_per_start` and `pitcher_this_season_start_pa_avg_pa_per_start`
still dominate by a wide margin. Full ranking: `plots/feature_importance.png`.

## Setup

- Features: ['pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts', 'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n', 'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start', 'expected_batters_faced', 'expected_batters_faced_weight', 'pitcher_throw_hand', 'team_roll_season_walk_rate', 'team_roll_season_on_base_rate', 'pitcher_team_days_since_last_game', 'is_home', 'pitcher_roll_season_pitch_count_avg', 'pitcher_last3_start_pa_avg_pa_per_start', 'pitcher_last3_start_pa_starts_n', 'pa_trend_ratio', 'pa_trend_direction', 'pitcher_days_since_last_start', 'pitcher_last_start_pitches', 'pitcher_workload_density', 'pitcher_workload_density_shrunk', 'opp_team_roll_season_win_pct', 'opp_team_roll_season_runs_scored', 'opp_team_roll_season_run_diff', 'opp_team_roll_last5g_win_pct', 'opp_team_roll_last5g_runs_scored', 'opp_team_roll_last5g_run_diff', 'opp_team_roll_season_runs_scored_mean', 'opp_team_roll_season_runs_scored_std', 'opp_team_roll_season_runs_scored_max', 'opp_team_roll_last5g_runs_scored_mean', 'opp_team_roll_last5g_runs_scored_std', 'opp_team_roll_last5g_runs_scored_max']
- New this pass: opp_team_roll_season_runs_scored_mean/_std/_max and
  opp_team_roll_last5g_runs_scored_mean/_std/_max — all via NEW function
  game_context.build_team_scoring_volatility(schedule, window), TDD'd
  (tests/hit_predictor/test_game_context.py, 4 new tests), called twice
  (window='season' and window=5) and joined on the OPPOSING team for this
  start (same opp_team_id derivation v1/v5 already use).
- Hypothesis: opposing-team scoring VOLATILITY (can this team explode for
  a blowout game) drives quicker pulls independent of scoring LEVEL,
  which v5 already tried (win_pct/runs_scored/run_diff) and found flat.
- v5's level features stay in the feature list unchanged (additive) — this
  experiment's feature set is v5's own (baseline + v1 + v2 + v5) plus the
  six new volatility columns above.
- XGBoost (v6, tuned) hyperparameters and its held-out-season
  early-stopping setup are carried over unchanged from v1-v5.
