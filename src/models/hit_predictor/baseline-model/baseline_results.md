# Baseline Results — batter_hit_predictor

**Date:** 2026-05-19  
**Task:** Binary classification (per plate appearance)  
**Target:** Did the batter record a hit in this PA?  
**Primary metric:** log_loss (lower = better)  
**Diagnostics:** brier score + calibration plot  
**Data:** s3://mlbdk  

## Split

| Split | Seasons | Rows | Hit rate |
|-------|---------|------|----------|
| Train | [2017, 2018, 2019, 2022, 2023] | 877,964 | 0.226 |
| Val   | 2024 | 177,697 | 0.221 |
| Test  | 2025 | 164,125 | locked — not evaluated here |

## Results (evaluated on val)

Naive baseline predicts the train hit rate for every PA — it sets the floor.
Improvement vs naive is the real signal that the model is learning something.

| Model | LogLoss | Δ vs Naive | % improvement | Brier | Δ vs Naive |
|-------|---------|------------|---------------|-------|------------|
| Naive baseline | 0.5281 | — | — | 0.1721 | — |
| Logistic regression | 0.5276 | -0.0005 | +0.09% | 0.1720 | -0.0002 |
| XGBoost | 0.5290 | +0.0009 | -0.17% | 0.1724 | +0.0003 |

## Setup

- Features: ['batting_order', 'batSide', 'pitcher_hand', 'game_season', 'weather_condition', 'weather_temp', 'weight', 'height_in_inches', 'strikeZoneTop', 'strikeZoneBottom', 'last_season_ba']
- last_season_ba: 2016→2017, 2019→2022 (covid gap bridge), then year-over-year
- Excluded: inning, half_inning, pitch_number (in-game); batter_id/pitcher_id (IDs)

## Plots

- `plots/baseline/calibration_curve.png` — key diagnostic, quantile bins
- `plots/baseline/predicted_proba_distribution.png` — sanity check on output spread
- `plots/baseline/confusion_matrix.png` — informational only at PA level
- `plots/baseline/feature_importance.png`

## Next steps

- Add pitcher ERA / WHIP / K-rate features
- Add rolling batting average windows (7/14/30 day) from feature store
- Add ballpark and batter×pitcher handedness interaction
- Aggregate PA-level predictions to game-level for BTS decision evaluation
- Final evaluation on test season (2025) only once in train.py
