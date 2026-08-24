# Baseline Results — bb_predictor

**Date:** 2026-08-23  
**Task:** Binary classification (PA grain)  
**Target:** Will this plate appearance end in a walk?  
**Primary metric:** PR-AUC (higher = better)  
**Diagnostics:** ROC-AUC, confusion matrix (XGBoost @ 0.5)  
**Data:** s3://mlbdk  

## Split

| Split | Seasons | Rows | BB rate |
|-------|---------|------|---------|
| Train | [2018, 2019, 2022, 2023] | 411,861 | 0.076 |
| Val   | 2024 | 105,265 | 0.076 |
| Test  | 2025 | 104,427 | locked — not evaluated here |

## Results (evaluated on val)

Two naive floors are reported: most-frequent-class, and the train-set BB
rate conditioned on expected pitcher role (sp vs bullpen). The real bar a
model must clear is the BETTER of the two.

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.0755 | — | 0.5000 |
| Naive (per-role BB rate) | 0.0757 | — | 0.5013 |
| Logistic regression | 0.0964 | +0.0206 | 0.5767 |
| XGBoost | 0.0896 | +0.0139 | 0.5619 |

## Interpretation

**Logistic regression** beats the best naive floor (**Naive (per-role BB rate)**, PR-AUC 0.0757) with PR-AUC 0.0964 — worth carrying forward into a real experiment.

## Setup

- Features: ['expected_pitcher_role', 'pitcher_last_season_pa_walk_rate', 'batter_last_season_pa_walk_rate', 'pitcher_throw_hand', 'batter_bat_side']
- No new feature engineering — pitcher/batter walk_rate and expected_pitcher_role
  already exist in hit_predictor's season_stats.py / expected_role.py (the same
  _create_pitcher_pa_outcome_stats / _create_batter_pa_outcome_stats aggregations
  k_predictor's strikeout_rate reuses also compute walk_rate in the same pass).
- Scoped to starting pitchers (REALIZED pitcher_role == 'sp') and starting batters
  (inner join on batting_order) only — same population-scoping k_predictor uses,
  and for the same reason: a bullpen reliever's identity isn't pre-game-knowable,
  and a non-starter isn't in the lineup a BB-prop line is written against.
- Label: is_walk (play_result in {'Walk', 'Intent Walk'}),
  bb_predictor.processing.pipeline.create_pa_outcome_walk

## Plots

- `plots/baseline-model/target_distribution.png`
- `plots/baseline-model/confusion_matrix.png` — XGBoost @ 0.5 threshold
- `plots/baseline-model/feature_importance.png`

## Next steps

- If a model beats the best naive floor: move to a real experiment, add rolling-window
  BB rate (recent form) and times_through_order — both already computed by hit_predictor's
  rolling_stats.py, just not wired in here to keep this baseline minimal.
- If not: this feature set isn't ready. Add rolling-window recent-form features before
  concluding there's no signal — season-level rate alone may be too coarse.
- Final evaluation on test season (2025) only once in a real experiment.
