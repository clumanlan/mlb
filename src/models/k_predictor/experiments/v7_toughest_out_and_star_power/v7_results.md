# v7 Results — k_predictor (toughest-out + best-batter star power)

**Date:** 2026-08-29
**Task:** Binary classification (PA grain), aggregated to batter-game grain for the decision verdict
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC (PA grain, triage filter) → reliability/resolution (game grain, decision metric)
**Data:** s3://mlbdk

## What this pass adds

Two new opposing-lineup features, both built on a shared "shrink each starting-lineup
batter's own stat toward last season, then take a cross-sectional extremum across the
lineup" pattern (new `k_predictor/processing/features/batter_workload.py`, direct port
of `pitcher_workload.build_pitcher_shrunk_whip`'s shrinkage recipe):

1. **PRIMARY — `opp_team_toughest_out_shrunk_k_rate`**: MIN(shrunk batter K rate)
   across the opposing starting lineup. A single elite-contact/low-K batter caps a
   pitcher's total-K ceiling regardless of how weak the rest of the lineup is — the
   existing team-average K rate (`opp_team_roll_season_pa_strikeout_rate`, from v2)
   can't see this.
2. **EXPLORATORY — `opp_team_best_batter_shrunk_obp` / `_slg`**: MAX(shrunk OBP) /
   MAX(shrunk SLG) across the same lineup, independently. Built to see what it looks
   like, per user request, after discussion reframed K-rate/contact as the
   better-motivated mechanism for a strikeout target specifically (OBP/SLG measure
   overall offensive value, not the bat-to-ball skill a strikeout is the direct
   inverse of).

Carries forward v6's exact winning hyperparameters (LR: C=0.1/L1/no class weighting;
XGBoost: max_depth=2, learning_rate=0.03) rather than re-grid-searching — same
"carry tuned hyperparameters forward unchanged" convention `batters_faced_predictor`'s
v2-v7 established.

## A real bug was caught and fixed mid-pass

The first run produced impossible feature values: `opp_team_toughest_out_shrunk_k_rate`
had a mean of exactly **0.0000** (a real K rate is never 0), and
`opp_team_best_batter_shrunk_obp` / `_slg` had means of **0.8834** / **1.7160**
(physically impossible — no real batter's OBP/SLG gets anywhere near that).

**Root cause:** `build_opposing_lineup_extremum`'s join dropped `gamepk` when
selecting columns from the shrunk table, then joined on `batter_id` alone. Since the
shrunk tables (`build_batter_shrunk_k_rate`/`build_batter_shrunk_obp_slg`) are
per-game rolling tables — one row per batter per game across the whole 2017-2025
dataset, not a static per-batter value — this fanned every lineup slot out against
**every game that batter ever played**, then took the MIN/MAX across his entire
multi-season history instead of just that night's value. MIN collapses toward 0
(some thin-sample early-season game is always near-zero somewhere in 7 seasons of
history); MAX inflates past 1 (some thin-sample game spikes to 1.000+).

**Fix:** scoped the join to `(batter_id, gamepk)`. Caught with a new regression test,
`test_build_opposing_lineup_extremum_uses_this_games_shrunk_value_not_other_games`
(`tests/k_predictor/test_batter_workload.py`) — constructs a batter with a real
value in today's game and a near-zero value in an unrelated earlier game, and
asserts only today's value survives. All 546 tests pass after the fix. The numbers
below are from the corrected re-run.

Sanity check on the corrected run — all three new columns now land in a plausible
range:

| Feature | Non-null | Mean | Sanity |
|---|---|---|---|
| `opp_team_toughest_out_shrunk_k_rate` | 99.8% | 0.1273 | below league K rate (~0.22), as expected for a MIN across 9 batters |
| `opp_team_best_batter_shrunk_obp` | 99.9% | 0.3854 | above league OBP (~0.32), as expected for a MAX across 9 batters |
| `opp_team_best_batter_shrunk_slg` | 99.9% | 0.5530 | above league SLG (~0.40), as expected for a MAX across 9 batters |

## Results (evaluated on val season 2024, 105,265 PAs)

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.2190 | — | 0.5000 |
| Naive (per-role K rate) | 0.2207 | — | 0.5048 |
| Logistic regression (v6 config) | 0.2822 | +0.0615 | 0.5979 |
| XGBoost (v6 config) | 0.2836 | +0.0629 | 0.5997 |
| (v6 tuned XGBoost, for reference) | 0.2838 | | |

**PA-grain: flat vs v6** (0.2836 vs 0.2838, -0.0002 — well inside this project's
~0.005 "real" threshold).

## Game-grain check (batter-game "1+ strikeout")

| Metric | Naive (per-role) | XGBoost (v6 config) | v6 (for reference) |
|---|---|---|---|
| reliability | 0.0005 | 0.0002 | 0.0001 |
| resolution | 0.0045 | 0.0136 | 0.0137 |
| roc_auc | 0.5626 | 0.6365 | — |

**Verdict: `real_improvement`** vs naive (same as every version since v2) — but not
a deeper one than v6's own result. Reliability and resolution are both within noise
of v6's numbers.

## Feature importance — the three new columns

| Feature | Rank (of 45) | Importance |
|---|---|---|
| `opp_team_toughest_out_shrunk_k_rate` | #33 | 0.0076 |
| `opp_team_best_batter_shrunk_obp` | #31 | 0.0081 |
| `opp_team_best_batter_shrunk_slg` | #41 | 0.0066 |

All three land in the bottom half of 45 features — modest but not dead-last
placement (the pre-fix buggy run had ranked the toughest-out feature dead last at
#44 with 0.0000 importance; post-fix it has real, if small, importance). No clean
separation between the well-motivated primary feature (toughest-out K-rate, #33)
and the exploratory one (#31/#41 for OBP/SLG) — OBP actually edges out K-rate
slightly.

## Interpretation

**Flat result, same "additive feature, no aggregate movement" pattern seen
throughout most of this project's feature-hunt history** (v3, v4, v5 here; v3-v7 on
`batters_faced_predictor`). This reads as a genuine negative result now that the
wiring bug is fixed and confirmed sane — not an artifact of thin data or a scale
error. The toughest-out K-rate mechanism (a single tough out capping the ceiling)
didn't clear the bar this pass, though it wasn't disproven either — a small,
consistent, correctly-signed contribution (real importance, right sign of effect)
that just doesn't move the aggregate metric, the same shape most individual feature
additions in this repo have taken.

**v6's tuned XGBoost (PR-AUC 0.2838, `real_improvement`, max_depth=2,
learning_rate=0.03) remains the strongest k_predictor version and standing
production candidate.**

## Next steps

- Don't pursue further variants of this specific mechanism (toughest-out K-rate,
  best-batter OBP/SLG) without a new hypothesis — this thread is closed on the
  evidence here, similar to how opposing-team K-rate volatility and pitch
  efficiency closed after v3.
- The genuinely new, untried thread is league-wide rolling context (does the model
  know what season/era it's in — rule changes, juiced/dead-ball eras) — see
  `ROADMAP.md`'s 2026-08-29 entry for the full brief; a separate session/pass has
  already started planning this.
- `batter_workload.py`'s shrinkage + lineup-extremum primitives are real, tested,
  reusable infrastructure regardless of this specific result — any future model
  needing a per-batter shrunk rate or a cross-sectional lineup extremum can reuse
  them directly.
