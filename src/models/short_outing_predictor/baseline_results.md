# Baseline Results — short_outing_predictor

**Date:** 2026-08-24  
**Task:** Binary classification (starting-pitcher-start grain)  
**Target:** Will this starting pitcher have a short outing (<=4 IP)?  
**Primary metric:** PR-AUC (higher = better)  
**Diagnostics:** ROC-AUC, confusion matrix (XGBoost @ 0.5)  
**Data:** s3://mlbdk  

## Split

| Split | Seasons | Rows | Short-outing rate |
|-------|---------|------|--------------------|
| Train | [2018, 2019, 2022, 2023] | 18,830 | 0.212 |
| Val   | 2024 | 4,786 | 0.198 |
| Test  | 2025 | 4,768 | locked — not evaluated here |

## Results (evaluated on val)

Two naive floors are reported: most-frequent-class, and the train-set
short-outing rate conditioned on the pre-game expected_start_innings blend
(rounded to the nearest whole inning). The real bar a model must clear is
the BETTER of the two.

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.1977 | — | 0.5000 |
| Naive (per-innings-bucket rate) | 0.3094 | — | 0.6307 |
| Logistic regression | 0.4199 | +0.1104 | 0.6772 |
| XGBoost | 0.3912 | +0.0818 | 0.6352 |

## Interpretation

**Logistic regression** beats the best naive floor (**Naive (per-innings-bucket rate)**, PR-AUC 0.3094) with PR-AUC 0.4199 — worth carrying forward into a real experiment.

## Setup

- Features: ['pitcher_last_season_start_ip_avg_ip_per_start', 'pitcher_last_season_start_ip_n_starts', 'pitcher_this_season_start_ip_avg_ip_per_start', 'pitcher_this_season_start_ip_starts_n', 'league_last_season_avg_ip_per_start', 'expected_start_innings', 'expected_start_innings_weight', 'pitcher_throw_hand']
- No new feature engineering — pitcher_last_season_start_ip_*, 
  pitcher_this_season_start_ip_*, league_last_season_avg_ip_per_start, and
  expected_start_innings/_weight already exist in hit_predictor's
  season_stats.py / game_context.py (built for n_pa_predictor's OPPOSING-
  starter feature — reused here for the pitcher's OWN start instead).
- Grain: one row per (personId, gamepk) starting-pitcher start — different
  from every sibling model's PA or batter-game grain. Scoped to REALIZED
  pitcher_role == 'sp' by construction (a bullpen boxscore row isn't a
  'start' at all), not as a population-scoping choice.
- Label: is_short_outing (realized ip <= 4.0),
  short_outing_predictor.processing.pipeline.create_start_outcome. README's
  mechanism also includes an explicit planned-opener flag not yet wired up
  here — see processing/schema.py's SHORT_OUTING_IP_THRESHOLD docstring.

## Plots

- `plots/baseline-model/target_distribution.png`
- `plots/baseline-model/confusion_matrix.png` — XGBoost @ 0.5 threshold
- `plots/baseline-model/feature_importance.png`

## Next steps

- If a model beats the best naive floor: move to a real experiment, add
  opponent platoon-advantage depth and bullpen rest state/day-after-
  doubleheader flags — both named in README's mechanism but not yet wired
  in here to keep this baseline minimal.
- If not: this feature set isn't ready. The pre-game innings estimate alone
  may be too coarse — recent-start pitch-count trend (not just IP) could
  carry more signal about an imminent short outing.
- Add explicit-opener detection to the label (see schema.py) before calling
  this experiment-ready — currently undercounts planned bullpen days that
  happen to log >4 IP via a long relief follow.
- Final evaluation on test season (2025) only once in a real experiment.
