# v11 Results — k_predictor (CSW% — called-strike rate + swinging-strike rate)

**Date:** 2026-08-30
**Task:** Binary classification (PA grain), aggregated to batter-game grain for the decision verdict
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC (PA grain, triage filter) → reliability/resolution (game grain, decision metric)
**Data:** s3://mlbdk

## What this pass adds

A residual-correlation screen (throwaway analysis, not a TDD build — see the
published artifact "The CSW% Screen") tested 8 candidate features against v6's
held-out val-season residuals before committing any to a full build: get
out-of-fold residuals from the current best model, hold each candidate OUT of
training, correlate it against the residual (Spearman + mutual info), then
check whether the bottom-decile-to-top-decile residual movement is monotonic
(real signal) or flat (likely spurious despite a nonzero correlation).

**CSW% (`command_called_strike_rate` + `command_swinging_strike_rate`) came
back the clear standout of the 8** — Spearman r = -0.1705 against the
residual, ~4x the next-strongest candidate, with a clean monotonic decile
shape (bottom-decile residual +0.0020 → top-decile -0.0113). The called-strike
rate component ALONE is weak in isolation (r = 0.0341) — the signal lives in
the combination, matching sabermetrics literature's finding that CSW% (not
either half alone) is the strongest single predictor of K% (R² ≈ 0.59). Six
pitch-tunneling/arsenal candidates from the same screen were parked (weak
signal, two showed a flat decile shape despite decent correlation — likely
spurious, plus a 58.7% coverage gap) — not part of this experiment.

Both components already existed — CSW% is their SUM, not new feature
engineering. New TDD'd column, `command_csw_rate`, at two grains (4 new tests
total, `tests/hit_predictor/test_rolling_stats.py` and
`tests/hit_predictor/test_season_stats.py`):
- **Rolling** (what the screen actually tested): `build_pbp_pitcher_rolling_feats`
  already computed both components from already-rolled counts.
- **Season/last-season** (added for symmetry per user's scope decision — NOT
  independently screened before this run): `season_stats.py`'s
  `_create_pitcher_stuff_command_stats` has the same two components at season
  grain.

Carries forward v6's exact winning hyperparameters and v10's full 114-column
feature set unchanged, per this project's established convention.

## Sanity check on the new columns

| Feature | Non-null | Mean | Sanity |
|---|---|---|---|
| `pitcher_roll_season_command_csw_rate` | 93.8% | 0.2736 | equals the already-verified sum of `command_called_strike_rate` + `command_swinging_strike_rate` at this same grain (TDD'd invariant, not just a plausibility check) |
| `pitcher_last_season_command_csw_rate` | 58.7% | 0.2752 | same identity at season grain; lower non-null matches every other "last season" column's shift-based coverage gap for rookies/first-year-in-data pitchers |

Both means sit in the expected ~0.27-0.28 range for a real MLB CSW% (per the
sabermetrics literature cited above, league-average CSW% is typically in the
high-20s to low-30s percent range) — a plausibility check, not just an
internal-consistency one.

## Results (evaluated on val season 2024, 105,265 PAs)

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.2190 | — | 0.5000 |
| Naive (per-role K rate) | 0.2207 | — | 0.5048 |
| Logistic regression (v6 config) | 0.2834 | +0.0627 | 0.5990 |
| XGBoost (v6 config) | 0.2839 | +0.0632 | 0.6005 |
| (v6 tuned XGBoost, standing best, for reference) | 0.2838 | | |

**PA-grain: flat vs v6** (0.2839 vs 0.2838, +0.0001 — well inside this
project's ~0.005 "real" threshold), same pattern as every version since v7.

## Game-grain check (batter-game "1+ strikeout")

| Metric | Naive (per-role) | XGBoost (v6 config) | v6 (standing best, for reference) |
|---|---|---|---|
| reliability | 0.0005 | 0.0002 | 0.0001 |
| resolution | 0.0045 | 0.0139 | 0.0137 |
| roc_auc | 0.5626 | 0.6376 | — |

**Verdict: `real_improvement`** vs naive (same as every version since v2) —
but not deeper than v6's own result: resolution is +0.0002 above v6's own
0.0137, reliability is -0.0001 (marginally worse), both within noise.

## Feature importance — the 2 new columns

| Feature | Rank (of 116) | Importance |
|---|---|---|
| `pitcher_last_season_command_csw_rate` | #22 | 0.0098 |
| `pitcher_roll_season_command_csw_rate` | #23 | 0.0071 |

**Real, respectable middle-tier usage — not top-ranked, not dead weight.**
Both CSW% columns land comfortably inside the top quintile of 116 features,
a meaningfully stronger showing than most single-pass additions in this
project's recent history (v10's pitcher-side put-away rate ranked #112/114;
several of v8's league-context and v9's vs-league columns ranked in the
bottom half). The season and rolling versions rank almost identically to
each other, consistent with them being the same underlying signal at two
different windows rather than two independent pieces of information.

## Interpretation

**A genuinely informative result about the screening methodology itself, not
just about CSW%.** The residual-correlation screen correctly predicted CSW%
would be the strongest of the 8 candidates tested, and the build confirms
that ranking — CSW% is real, used non-trivially by the model, and ranks well
above the typical "dead weight" columns this project has accumulated. But the
aggregate PA-grain and game-grain metrics moved by essentially nothing
(+0.0001 PR-AUC, +0.0002 resolution), the same "flat" verdict shape as every
single-pass feature addition since v7. **A Spearman r of -0.17 corresponds to
only ~3% of residual variance explained in a linear sense** — a real,
non-spurious relationship, but a modest one, and this project's "flat vs.
real" threshold (~0.005 on already-compressed metrics like PR-AUC/resolution)
turns out to require more magnitude than a correlation of that size reliably
delivers once folded into an already-116-feature shallow-tree (`max_depth=2`)
model with substantial pre-existing overlap in called-strike/command-adjacent
signal (`command_swinging_strike_rate` and several K-rate features were
already present).

**The practical takeaway for this project's feature-hunting process:** the
residual-correlation screen is a genuinely useful cheap filter for ruling out
candidates that are almost certainly redundant (it correctly flagged 6 of 8
candidates as weak/spurious before any of them cost a full session to build),
but passing the screen is evidence a candidate is *worth building*, not a
guarantee it will clear this project's aggregate-improvement bar. That
distinction is worth carrying into future screens — a real, useful cheap
triage step, not a way to skip the "flat until proven otherwise" prior this
project has earned through v3-v10's own history.

**v6's tuned XGBoost (PR-AUC 0.2838, `real_improvement`, max_depth=2,
learning_rate=0.03) remains the strongest k_predictor version and standing
production candidate**, seven follow-up passes (v5-v10 plus this one) in.

## Next steps

- `command_csw_rate` is real, tested, reusable infrastructure regardless of
  this flat result — kept, not reverted, same convention as every prior
  version's new columns.
- Both CSW% columns' respectable-but-not-dominant importance is consistent
  with them capturing real but partially-redundant signal (some overlap with
  `command_swinging_strike_rate`, already in the feature set since v3) — a
  trailing-N-game window (this pass only tried season-level for both grains)
  is the more likely next lever than re-trying at the same window.
- This result is the second data point (after v9's vs-league finding) arguing
  for the pre-production redundancy audit already queued in `ROADMAP.md`
  (correlation-cluster the K-rate-adjacent feature family, swap-ablate to see
  what's load-bearing) before finalizing the production feature set — CSW%
  should be included in that audit, not assumed necessary just because it
  ranks respectably.
