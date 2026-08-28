# v5 Results — batters_faced_predictor (opposing-team scoring strength)

**Date:** 2026-08-26  
**Task:** Regression (starting-pitcher-start grain)  
**Target:** realized_batters_faced  
**Primary metric:** MAE (lower = better)  

## Results (evaluated on val)

| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |
|-------|-----|---------------|------|------|-----------|
| Cascade (expected_batters_faced) | 2.8609 | — | 3.9509 | +0.0725 | 0.4949 |
| Linear regression (v5) | 2.7738 | -0.0871 | 3.7043 | +0.5155 | 0.5902 |
| XGBoost (v5, default) | 3.0795 | +0.2187 | 4.0738 | +0.8593 | 0.5238 |
| XGBoost (v5, tuned) | 2.6369 | -0.2240 | 3.5989 | +0.0898 | 0.6113 |
| (v2 tuned XGBoost, for reference) | 2.6471 | | | | |

## MAE/RMSE/Bias/Pearson r by expected_batters_faced_weight quartile

| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |
|---|---|---|---|---|---|---|
| Q1 (thinnest) | 1228 | Cascade (expected_batters_faced) | 3.3527 | 4.7167 | -0.5160 | 0.5252 |
| Q1 (thinnest) | 1228 | Linear regression (v5) | 3.1105 | 4.0509 | +0.6059 | 0.6845 |
| Q1 (thinnest) | 1228 | XGBoost (v5, default) | 3.0973 | 4.1634 | +0.4315 | 0.6691 |
| Q1 (thinnest) | 1228 | XGBoost (v5, tuned) | 2.9076 | 3.8750 | +0.2586 | 0.7105 |
| Q2 | 1253 | Cascade (expected_batters_faced) | 2.8050 | 3.7522 | +0.5751 | 0.5267 |
| Q2 | 1253 | Linear regression (v5) | 2.6831 | 3.5999 | +0.2839 | 0.5684 |
| Q2 | 1253 | XGBoost (v5, default) | 2.8289 | 3.7137 | +0.5471 | 0.5574 |
| Q2 | 1253 | XGBoost (v5, tuned) | 2.5500 | 3.4830 | +0.0836 | 0.6023 |
| Q3 | 1191 | Cascade (expected_batters_faced) | 2.6076 | 3.5060 | +0.3543 | 0.3021 |
| Q3 | 1191 | Linear regression (v5) | 2.5884 | 3.4843 | +0.4972 | 0.3341 |
| Q3 | 1191 | XGBoost (v5, default) | 2.9167 | 3.8558 | +0.7374 | 0.2814 |
| Q3 | 1191 | XGBoost (v5, tuned) | 2.4899 | 3.4135 | -0.0426 | 0.3570 |
| Q4 (most reliable) | 1114 | Cascade (expected_batters_faced) | 2.6524 | 3.6826 | -0.1452 | 0.2546 |
| Q4 (most reliable) | 1114 | Linear regression (v5) | 2.7027 | 3.6478 | +0.6958 | 0.3354 |
| Q4 (most reliable) | 1114 | XGBoost (v5, default) | 3.5160 | 4.5591 | +1.8124 | 0.1496 |
| Q4 (most reliable) | 1114 | XGBoost (v5, tuned) | 2.5934 | 3.6041 | +0.0523 | 0.3228 |

## bf_gap-quartile floor re-check

Overall: cascade bf_gap MAE 2.8609 | new estimate (XGBoost (v5, tuned)) bf_gap MAE 2.6369

| bf_gap quartile (by new estimate) | n | cascade_bf_gap MAE | new_bf_gap MAE |
|---|---|---|---|
| Q1 (closest) | 1197 | 1.1123 | 0.4714 |
| Q2 | 1196 | 1.7313 | 1.4736 |
| Q3 | 1196 | 2.8290 | 2.7273 |
| Q4 (furthest) | 1197 | 5.7699 | 5.8745 |

## Established-starter (11+ starts) bucket re-check

Same slice v2/v3/v4 check against — this pass does not specifically target
that failure mode (that thread is closed, see ROADMAP.md's v4 entry), so this
is a sanity re-check for regressions, not the headline result.

n=2136 | cascade bf_gap MAE 2.6053 | XGBoost (v5, tuned) bf_gap MAE 2.5273

## Feature importance (XGBoost, v5, tuned)

`opp_team_roll_season_runs_scored` — the primary-hypothesis feature — ranks #11
of 27, ahead of `pitcher_days_since_last_start` and `is_home`, modestly better
placed than v1's on-base/walk-rate features. But `opp_team_roll_season_win_pct`
and `opp_team_roll_season_run_diff` both rank in the bottom third, and all
three `opp_team_roll_last5g_*` (trailing-5-game) columns rank near-zero
importance — the short-window secondary hypothesis shows no signal at all.
The top of the ranking is unchanged from v2: `pitcher_last3_start_pa_avg_pa_per_start`
and `pitcher_this_season_start_pa_avg_pa_per_start` still dominate by a wide
margin. Full ranking: `plots/feature_importance.png`.

## Setup

- Features: ['pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts', 'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n', 'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start', 'expected_batters_faced', 'expected_batters_faced_weight', 'pitcher_throw_hand', 'team_roll_season_walk_rate', 'team_roll_season_on_base_rate', 'pitcher_team_days_since_last_game', 'is_home', 'pitcher_roll_season_pitch_count_avg', 'pitcher_last3_start_pa_avg_pa_per_start', 'pitcher_last3_start_pa_starts_n', 'pa_trend_ratio', 'pa_trend_direction', 'pitcher_days_since_last_start', 'pitcher_last_start_pitches', 'pitcher_workload_density', 'pitcher_workload_density_shrunk', 'opp_team_roll_season_win_pct', 'opp_team_roll_season_runs_scored', 'opp_team_roll_season_run_diff', 'opp_team_roll_last5g_win_pct', 'opp_team_roll_last5g_runs_scored', 'opp_team_roll_last5g_run_diff']
- New this pass: opp_team_roll_season_win_pct/_runs_scored/_run_diff and
  opp_team_roll_last5g_win_pct/_runs_scored/_run_diff — both via
  game_context.build_team_win_loss_record(schedule, window), already
  existing and TDD'd (tests/hit_predictor/test_game_context.py), called
  twice (window='season' and window=5) and joined on the OPPOSING team
  for this start (same opp_team_id derivation v1 already uses).
- Hypothesis: a team's general run-scoring strength (not just its
  on-base/walk discipline, which v1 already tried with only a weak
  effect) drives quicker pulls for a struggling starter — more traffic
  of any kind means more total batters faced for the same outs recorded.
- v4's anomaly-count feature is dropped (that thread is closed, see
  ROADMAP.md) — this experiment's feature set is v2's own
  (baseline + v1 + v2) plus the six new columns above.
- XGBoost (v5, tuned) hyperparameters and its held-out-season
  early-stopping setup are carried over unchanged from v1-v4.
