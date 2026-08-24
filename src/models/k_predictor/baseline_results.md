# Baseline Results — k_predictor

**Date:** 2026-08-23  
**Task:** Binary classification (PA grain)  
**Target:** Will this plate appearance end in a strikeout?  
**Primary metric:** PR-AUC (higher = better)  
**Diagnostics:** ROC-AUC, confusion matrix (XGBoost @ 0.5)  
**Data:** s3://mlbdk  

## Split

| Split | Seasons | Rows | K rate |
|-------|---------|------|--------|
| Train | [2018, 2019, 2022, 2023] | 411,861 | 0.218 |
| Val   | 2024 | 105,265 | 0.219 |
| Test  | 2025 | 104,427 | locked — not evaluated here |

## Results (evaluated on val)

Two naive floors are reported: most-frequent-class, and the train-set K
rate conditioned on expected pitcher role (sp vs bullpen). The real bar a
model must clear is the BETTER of the two.

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.2190 | — | 0.5000 |
| Naive (per-role K rate) | 0.2207 | — | 0.5048 |
| Logistic regression | 0.2702 | +0.0495 | 0.5821 |
| XGBoost | 0.2603 | +0.0396 | 0.5698 |

## Interpretation

**Logistic regression** beats the best naive floor (**Naive (per-role K rate)**, PR-AUC 0.2207) with PR-AUC 0.2702 — worth carrying forward into a real experiment.

## Setup

- Features: ['expected_pitcher_role', 'pitcher_last_season_pa_strikeout_rate', 'batter_last_season_pa_strikeout_rate', 'pitcher_throw_hand', 'batter_bat_side']
- No new feature engineering — pitcher/batter strikeout_rate and expected_pitcher_role
  already exist in hit_predictor's season_stats.py / expected_role.py.
- Label: is_strikeout (play_result in {'Strikeout', 'Strikeout Double Play'}),
  k_predictor.processing.pipeline.create_pa_outcome_strikeout

## Plots

- `plots/baseline-model/target_distribution.png`
- `plots/baseline-model/confusion_matrix.png` — XGBoost @ 0.5 threshold
- `plots/baseline-model/feature_importance.png`

## Next steps

- If a model beats the best naive floor: move to a real experiment, add rolling-window
  K rate (recent form) and times_through_order — both already computed by hit_predictor's
  rolling_stats.py, just not wired in here to keep this baseline minimal.
- If not: this feature set isn't ready. Add rolling-window recent-form features before
  concluding there's no signal — season-level rate alone may be too coarse.
- Final evaluation on test season (2025) only once in a real experiment.
