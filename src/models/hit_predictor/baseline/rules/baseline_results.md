# Baseline Results — batter_hit_predictor

> **Superseded 2026-08-22.** Written before the reliability/resolution/game-grain
> evaluation framework existed (BENCHMARKS.md §2) — everything below is PA-grain
> log_loss/Brier only, computed by hand in `run_baseline.py` rather than the
> shared `utils/eval.py` harness every other run in this repo uses. Use
> `baseline/statistical/baseline_results.md` instead: a single, cleaner
> empirical-Bayes formula (vs. this file's 3-way hand-weighted blend),
> evaluated at both PA and game grain with a proper reliability/resolution
> verdict against naive. Kept here for history, not as a current reference
> point.

**Date:** 2026-07-30  
**Task:** Binary classification (per plate appearance)  
**Target:** Did the batter record a hit in this PA?  
**Primary metric:** log_loss (lower = better)  
**Diagnostics:** brier score + calibration plot  
**Data:** s3://mlbdk  

## Split

| Split | Seasons | Rows | Hit rate |
|-------|---------|------|----------|
| Train | [2017, 2018, 2019, 2022, 2023] | 837,119 | 0.227 |
| Val   | 2024 | 171,017 | 0.222 |
| Test  | 2025 | 158,099 | locked — not evaluated here |

## Results (evaluated on val)

Naive baseline predicts the train hit rate for every PA — it sets the floor.
Rules-based baseline is a hand-weighted blend of prev-season BA, this-season
shrunk rolling BA (k=100), and previous-season BA-by-batting-order-slot.

| Model | LogLoss | Δ vs Naive | % improvement | Brier | Δ vs Naive |
|-------|---------|------------|---------------|-------|------------|
| Naive baseline | 0.5294 | — | — | 0.1727 | — |
| Rules-based baseline | 0.5295 | +0.0001 | -0.03% | 0.1727 | +0.0000 |

## Rules-based baseline — component breakdown (val)

| Component | LogLoss | Brier |
|-----------|---------|-------|
| Rules: prev season only | 0.5450 | 0.1744 |
| Rules: rolling shrunk only | 0.5292 | 0.1726 |
| Rules: order slot only | 0.5292 | 0.1726 |

## Setup

- last_season_ba: 2016→2017, 2019→2022 (covid gap bridge), then year-over-year
- Rules-based baseline weights: w_prev=0.4, w_roll=0.4, w_order=0.2, k=100

## Plots

- `plots/baseline-model/calibration_curve.png` — key diagnostic, quantile bins
- `plots/baseline-model/predicted_proba_distribution.png`
- `plots/baseline-model/target_distribution.png`

## Next steps

- Look at where the component breakdown is weakest and revisit that rule
- Tune w_prev/w_roll/w_order and k against val, not by hand-guessing
- Check bucket-level calibration at the tails specifically
- Final evaluation on test season (2025) only once, in train.py
