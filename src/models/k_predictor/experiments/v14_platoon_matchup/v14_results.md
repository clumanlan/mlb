# v14 Results — k_predictor (platoon_matchup interaction feature)

**Date:** 2026-09-03
**Task:** Binary classification (PA grain), aggregated to batter-game grain for the decision verdict
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC (PA grain, triage filter) -> reliability/resolution (game grain, decision metric), plus a slice-level bootstrap check (the real decision metric for this specific feature)
**Data:** s3://mlbdk

## What this pass adds

A same-session bootstrap-CI'd slice diagnostic (ROADMAP.md item 7, reusing
`hit_predictor`'s already-validated slice-diagnostic tooling against
k_predictor's real PA predictions for the first time) found a real,
statistically significant gap: same-hand batter/pitcher matchups are
significantly worse-calibrated than the model overall, despite
`pitcher_throw_hand` and `batter_bat_side` already being raw model inputs
since v6. Root cause: both columns are ordinally encoded, and "same code on
two separately-encoded columns" (`throw_hand_code == bat_side_code`) isn't a
split a `max_depth=2` tree naturally discovers — it can approximate the
interaction given enough boosting rounds, but not express it directly.
`hit_predictor` already solved this for its own model via a precomputed
`platoon_matchup` column (`baseline/statistical/slice_diagnostic.py`), which
k_predictor never got.

New derived column, `platoon_matchup` (`same_hand`/`opposite_hand`/
`switch_hitter`), added to `create_pa_outcome_strikeout`
(`processing/pipeline.py`) so every downstream experiment inherits it for
free — same derivation hit_predictor's diagnostic already uses:
`is_switch = batter_bat_side == 'S'`, `is_same_hand = batter_bat_side ==
pitcher_throw_hand`. 4 new tests (`tests/k_predictor/test_pipeline.py`)
covering same-hand, opposite-hand, switch-hitter, and null-propagation.
Added ADDITIVELY to v6's `FEATURE_COLS` (43 total) — the two raw hand
columns stay, `platoon_matchup` is a third, explicit interaction column.
Everything else (data, tuning grids, eval) identical to `v6_tuned/train.py`.

## Results (evaluated on val season 2024, 105,265 PAs)

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.2190 | — | 0.5000 |
| Naive (per-role K rate) | 0.2207 | — | 0.5048 |
| Logistic regression (tuned) | 0.2823 | +0.0616 | 0.5981 |
| XGBoost (tuned) | 0.2839 | +0.0632 | 0.5999 |
| (v6 config, standing best, for reference) | 0.2837 | | 0.600 |

**PA-grain: flat vs v6** (+0.0002 PR-AUC) — same pattern as every version
since v7.

## Game-grain check (batter-game "1+ strikeout")

| Metric | Naive (per-role) | v14 XGBoost (tuned) | v6 config (for reference) |
|---|---|---|---|
| reliability | 0.0005 | 0.00016 | ~0.0002 |
| resolution | 0.0045 | 0.01376 | ~0.0137 |
| roc_auc | 0.5626 | 0.6367 | ~0.6367 |

**Verdict: `real_improvement`** vs naive (same as every version since v2) —
flat vs v6, same as every version since v7.

## The real test: slice-level verification, not aggregate metrics

Aggregate PA-grain and game-grain metrics being flat does not settle whether
`platoon_matchup` fixed the specific problem it was built for — same-hand
PAs are ~42% of the population and the reported gap was small, easily
diluted to invisible in a whole-population number. `slice_verification.py`
fits the v6 feature set (42 features, no `platoon_matchup`) and the v14
feature set (43, + `platoon_matchup`) side by side on the IDENTICAL
train/val split and IDENTICAL fixed XGBoost config (`max_depth=2`,
`learning_rate=0.03` — the winning config both grid searches independently
picked), so any difference is attributable to the one feature.

| Slice | n | Reliability (v6->v14, lower=better) | Resolution (v6->v14, higher=better) |
|---|---|---|---|
| ALL PAs | 105,265 | 0.00005 -> 0.00005 (flat) | 0.00330 -> 0.00324 (flat/-) |
| **same_hand** | **43,769** | **0.00011 -> 0.00008 (-0.00004)** | **0.00275 -> 0.00284 (+0.00009)** |
| opposite_hand | 50,596 | 0.00004 -> 0.00003 (-0.00002) | 0.00371 -> 0.00372 (flat) |
| switch_hitter | 10,859 | 0.00008 -> 0.00007 (-0.00001) | 0.00278 -> 0.00278 (flat) |

Both calibration and discrimination improve specifically on the `same_hand`
slice — exactly the slice the diagnostic flagged — while the other two
slices barely move.

**Paired bootstrap, same_hand slice, Brier score (v6 minus v14, 1000
resamples, same rows both models):** delta = +0.00004, 95% CI **[+0.00001,
+0.00009]** — entirely positive, excludes zero. **Real improvement, not
noise**, on 43,769 PAs.

## Interpretation

**`platoon_matchup` is a confirmed, real, bootstrap-significant fix — small
in absolute magnitude, invisible in aggregate metrics, but reproducible on
the specific slice it was built for.** This is a genuinely different
failure mode than every prior "flat" feature addition in this project's
history (v7-v11): those were candidates for NEW information the model
didn't have access to; `platoon_matchup` re-exposes information the model
already had (raw `pitcher_throw_hand`/`batter_bat_side`) in a form a shallow
tree can actually use. The lesson generalizes: an aggregate PR-AUC/game-grain
"flat" verdict is the right read for "does this candidate add new
information," but the wrong lens for "did this fix a known, localized
calibration bug" — that question needs the slice-level bootstrap check this
version introduces as a verification step.

**v6's tuned XGBoost (PR-AUC 0.2838-0.2839, `real_improvement`) remains the
standing production candidate.** `platoon_matchup` should be adopted into
the production feature set regardless of the flat aggregate result — it's a
verified fix, not a speculative addition.

## Case studies (val season 2024, v14 model)

Most-confident predicted strikeouts are dominated by extreme swing-and-miss
hitters (Bobby Dalbec, Joey Gallo, K rates 0.43-0.53+) against high-K
pitchers regardless of platoon direction — `platoon_matchup` nudges at the
margin rather than driving the top of the confidence ranking. Biggest
false-negative misses (predicted low, struck out anyway) are dominated by a
single low-K contact hitter (Miguel Rojas, 4 of the top 10) against four
different low-K pitchers — not explained by any available leading stat,
flagged as a candidate for future investigation rather than concluded on.

## Next steps

- `platoon_matchup` is real, tested, reusable infrastructure — kept, added
  to the shared pipeline (`create_pa_outcome_strikeout`), available to every
  future k_predictor experiment.
- The broader systematic sweep across OTHER undiscovered 2-way categorical
  interactions (`expected_pitcher_role` x `tto_bucket`, `weather_condition`
  x handedness, the `Dome` weather slice flagged in the original diagnostic)
  is still open — this version only closed the specific platoon thread.
- Same slice-plus-bootstrap verification methodology introduced here was
  reused for v15's in-game workload features (see `v15_results.md`) — a
  reusable pattern for this project's future single-feature additions
  going forward, not a one-off for platoon specifically.
