# v7 Results — batters_faced_predictor (pitch-count trend + own-team bullpen strength)

**Date:** 2026-08-27  
**Task:** Regression (starting-pitcher-start grain)  
**Target:** realized_batters_faced  
**Primary metric:** MAE (lower = better)  

## Results (evaluated on val)

| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |
|-------|-----|---------------|------|------|-----------|
| Cascade (expected_batters_faced) | 2.8609 | — | 3.9509 | +0.0725 | 0.4949 |
| Linear regression (v7) | 2.7684 | -0.0924 | 3.7008 | +0.4716 | 0.5895 |
| XGBoost (v7, default) | 2.9498 | +0.0889 | 3.9391 | +0.5658 | 0.5423 |
| XGBoost (v7, tuned) | 2.6405 | -0.2203 | 3.6064 | +0.0328 | 0.6099 |
| (v2 tuned XGBoost, for reference) | 2.6471 | | | | |

## MAE/RMSE/Bias/Pearson r by expected_batters_faced_weight quartile

| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |
|---|---|---|---|---|---|---|
| Q1 (thinnest) | 1228 | Cascade (expected_batters_faced) | 3.3527 | 4.7167 | -0.5160 | 0.5252 |
| Q1 (thinnest) | 1228 | Linear regression (v7) | 3.1067 | 4.0475 | +0.5484 | 0.6835 |
| Q1 (thinnest) | 1228 | XGBoost (v7, default) | 3.1428 | 4.1695 | +0.4488 | 0.6694 |
| Q1 (thinnest) | 1228 | XGBoost (v7, tuned) | 2.9113 | 3.9051 | +0.0148 | 0.7077 |
| Q2 | 1253 | Cascade (expected_batters_faced) | 2.8050 | 3.7522 | +0.5751 | 0.5267 |
| Q2 | 1253 | Linear regression (v7) | 2.6727 | 3.5976 | +0.2473 | 0.5682 |
| Q2 | 1253 | XGBoost (v7, default) | 2.8111 | 3.7335 | +0.4498 | 0.5438 |
| Q2 | 1253 | XGBoost (v7, tuned) | 2.5608 | 3.4966 | +0.0418 | 0.5993 |
| Q3 | 1191 | Cascade (expected_batters_faced) | 2.6076 | 3.5060 | +0.3543 | 0.3021 |
| Q3 | 1191 | Linear regression (v7) | 2.5915 | 3.4803 | +0.4657 | 0.3349 |
| Q3 | 1191 | XGBoost (v7, default) | 2.7278 | 3.6979 | +0.3847 | 0.2821 |
| Q3 | 1191 | XGBoost (v7, tuned) | 2.4922 | 3.4055 | +0.0349 | 0.3550 |
| Q4 (most reliable) | 1114 | Cascade (expected_batters_faced) | 2.6524 | 3.6826 | -0.1452 | 0.2546 |
| Q4 (most reliable) | 1114 | Linear regression (v7) | 2.6925 | 3.6436 | +0.6453 | 0.3315 |
| Q4 (most reliable) | 1114 | XGBoost (v7, default) | 3.1303 | 4.1474 | +1.0190 | 0.1739 |
| Q4 (most reliable) | 1114 | XGBoost (v7, tuned) | 2.5904 | 3.5941 | +0.0401 | 0.3257 |

## bf_gap-quartile floor re-check

Overall: cascade bf_gap MAE 2.8609 | new estimate (XGBoost (v7, tuned)) bf_gap MAE 2.6405

| bf_gap quartile (by new estimate) | n | cascade_bf_gap MAE | new_bf_gap MAE |
|---|---|---|---|
| Q1 (closest) | 1197 | 1.0603 | 0.4746 |
| Q2 | 1196 | 1.7181 | 1.4834 |
| Q3 | 1196 | 2.7931 | 2.7237 |
| Q4 (furthest) | 1197 | 5.8710 | 5.8796 |

## Established-starter (11+ starts) bucket re-check

Same slice v2-v6 check against — this pass does not specifically target
that failure mode (that thread is closed, see ROADMAP.md's v4 entry), so this
is a sanity re-check for regressions, not the headline result.

n=2136 | cascade bf_gap MAE 2.6053 | XGBoost (v7, tuned) bf_gap MAE 2.5233

## Feature importance (XGBoost, v7, tuned)

The two hypotheses split sharply on which of their features actually carried
signal. `pitcher_roll_last3g_pitch_count_avg` (trailing-3-start pitch-count
**LEVEL**, at the new shorter window) is the **#2 most important feature
overall** — behind only `pitcher_last3_start_pa_avg_pa_per_start` and ahead of
`pitcher_this_season_start_pa_avg_pa_per_start`, which had been the dominant
feature in every prior version. The season-level pitch-count feature v1
already had (`pitcher_roll_season_pitch_count_avg`) drops to #11, well behind
its own trailing-3 sibling — recency matters far more than season-to-date
level for this signal. But the TREND transforms built specifically to test
Hypothesis 1 — `pitch_count_trend_ratio` and `pitch_count_trend_direction`
(short vs. season, the actual new idea this pass) — rank near the very
bottom (#30 of 34 and dead last), essentially unused by the model. Same
shape as `interaction_feats.py`'s own documented finding for `pa_trend_*`:
a derived ratio/direction adds nothing a tree can't already reconstruct from
the two raw levels it was computed from — the win here came from adding a
new raw window (trailing-3 pitch count), not from the trend transform.

Bullpen strength (Hypothesis 2) shows a real but modest placement:
`bullpen_roll_last5g_whip` and `bullpen_roll_season_whip` are the strongest
of the ten new bullpen columns (#14-15), ahead of `pa_trend_ratio` and
comparable to v5's `opp_team_roll_season_runs_scored` placement — but well
below the top tier. The other four bullpen rate stats at both windows
(k_rate, bb_rate, hr_rate, strike_rate) all rank in the bottom half,
weaker than WHIP. Own-team bullpen quality is a real but secondary signal
at best, similar in strength to the weaker opposing-team features tried in
v1/v5/v6, not a new top-tier lever.

## Setup

- Features: ['pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts', 'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n', 'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start', 'expected_batters_faced', 'expected_batters_faced_weight', 'pitcher_throw_hand', 'team_roll_season_walk_rate', 'team_roll_season_on_base_rate', 'pitcher_team_days_since_last_game', 'is_home', 'pitcher_roll_season_pitch_count_avg', 'pitcher_last3_start_pa_avg_pa_per_start', 'pitcher_last3_start_pa_starts_n', 'pa_trend_ratio', 'pa_trend_direction', 'pitcher_days_since_last_start', 'pitcher_last_start_pitches', 'pitcher_workload_density', 'pitcher_workload_density_shrunk', 'opp_team_roll_season_win_pct', 'opp_team_roll_season_runs_scored', 'opp_team_roll_season_run_diff', 'opp_team_roll_last5g_win_pct', 'opp_team_roll_last5g_runs_scored', 'opp_team_roll_last5g_run_diff', 'opp_team_roll_season_runs_scored_mean', 'opp_team_roll_season_runs_scored_std', 'opp_team_roll_season_runs_scored_max', 'opp_team_roll_last5g_runs_scored_mean', 'opp_team_roll_last5g_runs_scored_std', 'opp_team_roll_last5g_runs_scored_max', 'pitcher_roll_last3g_pitch_count_avg', 'pitch_count_trend_ratio', 'pitch_count_trend_direction', 'bullpen_roll_season_whip', 'bullpen_roll_season_k_rate', 'bullpen_roll_season_bb_rate', 'bullpen_roll_season_hr_rate', 'bullpen_roll_season_strike_rate', 'bullpen_roll_last5g_whip', 'bullpen_roll_last5g_k_rate', 'bullpen_roll_last5g_bb_rate', 'bullpen_roll_last5g_hr_rate', 'bullpen_roll_last5g_strike_rate']
- New this pass (zero new production code — pure composition of
  already-existing, already-tested functions):
  1. PITCH-COUNT TREND: pitcher_roll_last3g_pitch_count_avg via
     rolling_stats.build_pbp_pitcher_rolling_feats(sp_pbp, window=3, ...)
     — the same function v1's season-level pitch_count_avg already uses,
     called again with an int window. pitch_count_trend_ratio/_direction
     computed inline, same pattern as v2's pa_trend_ratio/_direction.
  2. OWN-TEAM BULLPEN STRENGTH: bullpen_roll_{season,last5g}_{whip,
     k_rate,bb_rate,hr_rate,strike_rate} via
     rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore,
     pbp, window) — the same role-aware pooling function k_predictor's
     v1/v2 and n_pa_predictor's baseline already use — filtered to
     pitcher_role == 'bullpen' and joined on the STARTER's own team
     (pitcher_team_id), not the opposing team.
- Hypothesis 1: a pitcher's recent pitch count trending above/below his
  own season baseline (not just the level, already tried in v1) predicts
  an earlier or later hook.
- Hypothesis 2: a manager who trusts a strong bullpen may pull the
  starter earlier regardless of the starter's own workload/performance —
  the first SELF-team feature tried on this model (v1/v5/v6 all looked
  at the OPPOSING team).
- v1-v6 features stay in the feature list unchanged (additive) — this
  experiment's feature set is v6's own plus the 13 new columns above.
- XGBoost (v7, tuned) hyperparameters and its held-out-season
  early-stopping setup are carried over unchanged from v1-v6.
