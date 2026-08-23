# Statistical Shrinkage Baseline Results — batter_hit_predictor

**Date:** 2026-08-22  
**Estimator:** this-season BA-so-far shrunk toward last-season BA (k=100), falling back to season league-average hit rate for batters with no prior-season data.  
**Data:** s3://mlbdk  

## Split

| Split | Seasons | Rows | Hit rate |
|-------|---------|------|----------|
| Train | [2017, 2018, 2019, 2022, 2023] | 847,354 | 0.225 |
| Val   | 2024 | 173,067 | 0.219 |

## Results (val season, evaluated at both grains)

| Metric | PA grain | Game grain |
|--------|----------|------------|
| LogLoss | 0.5279 | 0.6375 |
| Brier | 0.1717 | 0.2229 |
| ROC-AUC | 0.5206 | 0.6489 |
| ECE | 0.0185 | 0.0296 |
| Reliability | 0.0005 | 0.0015 |
| Resolution | 0.0002 | 0.0159 |

## Verdict vs naive (game grain)

Per BENCHMARKS.md §2, the decision metrics are reliability + resolution together, not log_loss/Brier eyeballed against the docs — computed here via `utils/eval.py::summarize_verdict()` against a naive baseline run through the identical harness in this same script run (not a cached number).

| | Naive | Shrinkage baseline | Delta |
|---|---|---|---|
| Reliability | 0.0041 | 0.0015 | -0.0026 |
| Resolution | 0.0182 | 0.0159 | -0.0023 |

**Verdict: `calibration_only`** — trustworthy=True, differentiated=False.

## Setup

- shrinkage k = 100
- last_season_ba: from season_stats.build_batter_stats (2016->2017, 2019->2022 covid gap bridge, then year-over-year)
