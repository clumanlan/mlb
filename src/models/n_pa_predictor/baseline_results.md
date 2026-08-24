# Baseline Results — n_pa_predictor

**Date:** 2026-08-23  
**Task:** Regression (batter-game grain)  
**Target:** How many plate appearances will this batter get in this game?  
**Primary metric:** MAE (lower = better)  
**Diagnostics:** RMSE, predicted-vs-actual scatter  
**Data:** s3://mlbdk  

## Split

| Split | Seasons | Rows | Mean n_pa |
|-------|---------|------|-----------|
| Train | [2018, 2019, 2022, 2023] | 168,951 | 3.988 |
| Val   | 2024 | 43,039 | 4.012 |
| Test  | 2025 | 42,877 | locked — not evaluated here |

## Results (evaluated on val)

Two naive floors are reported: global mean, and the batter's own trailing
rolling avg_n_pa_per_game. The real bar a model must clear is the BETTER of
the two — not just the global mean (see Interpretation below for which one
won this run).

| Model | MAE | Δ vs global naive | RMSE |
|-------|-----|--------------------|------|
| Naive (global mean) | 0.4955 | — | 0.7790 |
| Naive (rolling avg) | 0.5611 | +0.0656 | 0.7399 |
| Linear regression | 0.5419 | +0.0464 | 0.7021 |
| XGBoost | 0.5539 | +0.0584 | 0.7221 |

## Interpretation

No model beats the best naive floor (**Naive (global mean)**, MAE 0.4955). The current feature set (batting_order, home/away, opposing starter's expected_start_innings + season WHIP, batter's own rolling PA rate, team win_pct) has no demonstrated signal beyond the constant-mean prediction — do not carry this feature set into train.py as-is. n_pa's real per-game variance may be dominated by within-game events (extra innings, blowouts, injury subs) that no pre-game feature can see; investigate that before adding more features.

## Setup

- Features: ['batting_order', 'is_home', 'expected_start_innings', 'expected_start_innings_weight', 'opp_starter_whip_season', 'batter_n_pa_roll_season_games_n', 'batter_n_pa_roll_season_avg_n_pa_per_game', 'team_roll_season_win_pct', 'team_roll_season_runs_scored']
- Scope: starters only (has a batting_order) — subs/pinch-hitters excluded
- Label: build_n_pa_label, filtered to PBP.PA_OUTCOMES (not boxscore's ab+bb, which undercounts)

## Plots

- `plots/baseline-model/target_distribution.png`
- `plots/baseline-model/residuals.png` — predicted vs actual, XGBoost
- `plots/baseline-model/feature_importance.png`

## Next steps

- If a model beats the best naive floor: move to train.py, consider more
  opponent-pitcher signal (bullpen quality/rest), team implied run total.
- If not: this feature set isn't ready for train.py. Investigate whether n_pa
  variance is dominated by within-game events pre-game features can't see,
  before adding more features — do not use an unproven model as a
  hit_predictor feature, and prefer the global mean over a losing rolling-avg
  baseline if one must be picked today.
- Final evaluation on test season (2025) only once in train.py.
