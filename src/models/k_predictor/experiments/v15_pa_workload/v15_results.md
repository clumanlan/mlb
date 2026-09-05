# v15 Results — k_predictor (in-game workload features)

**Date:** 2026-09-04
**Task:** Binary classification (PA grain), aggregated to batter-game grain for the decision verdict
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC (PA grain, triage filter) -> reliability/resolution (game grain, decision metric), plus a slice-level bootstrap check (same verification pattern v14 introduced)
**Data:** s3://mlbdk

## What this pass adds

Two features from the same "already have the ingredients, never combined
them" bucket v14's `platoon_matchup` came from:

1. **`estimated_team_pa_position`** (zero new pipeline code) — the uncapped,
   continuous version of the PA-position signal `FEATURE_COLS` already had
   capped at 3 as `expected_times_through_order`. Already computed by
   `create_pa_outcome_strikeout`, already tested — just never added to any
   experiment's `FEATURE_COLS` before. Capping at 3 throws away granularity
   the fatigue hypothesis needs: a batter's 4th PA against a starter still
   cruising isn't the same as his 4th PA against one who's clearly gassed,
   but both read as `TTO=3`.
2. **`pitcher_projected_pitches_before_pa`** (new TDD'd function,
   `build_pitcher_projected_workload` in `processing/features/pitcher_workload.py`,
   3 new tests in `tests/k_predictor/test_pitcher_workload.py`) —
   `(estimated_team_pa_position - 1) * pace`, a projected cumulative pitch
   count heading into this PA. Rationale: two batters at the same PA-position
   slot can represent very different real pitcher workload depending on how
   many pitches he's been burning per batter (deep counts, foul balls,
   working around traffic) — pitch count is a sharper fatigue proxy than
   PA-position or TTO alone.

Both point-in-time-safe the same way `expected_times_through_order` already
is: `estimated_team_pa_position` is a deterministic function of
`batting_order` (known before first pitch), not a leaked realized outcome
(same reasoning `expected_role.py` already relies on); the pace input is
already `shift(1)`'d to exclude the current game.

## A real bug found and fixed mid-build, not part of the original plan

The pace input was initially joined onto `pa_outcome` via
`expected_pitcher_key_id`/`expected_pitcher_role` — the same pre-game-estimate
join every other pitcher-level rolling stat in `FEATURE_COLS` uses, which
deliberately falls back to team-pooled bullpen stats once a PA is projected
past the starter's own expected depth (correct for those OTHER features,
where at serving time you genuinely don't know which reliever it'll be). But
`create_pa_outcome_strikeout` is already scoped to REALIZED
`pitcher_role=='sp'` rows only — every row here really is the starter — so
reusing that estimate-gated join for a feature meant to describe THIS
pitcher's own workload silently swapped in his team's bullpen pace once his
outing ran long, exactly the innings where a fatigue signal should be
sharpest.

**Caught via a manual case study** (`case_study.py`, Davis Schneider's 5 PAs
vs. Nick Pivetta, 2024-06-17): the pace column visibly dropped from 4.18 to
4.07 pitches/PA between PA 3 and PA 4 — precisely where
`expected_times_through_order` went to `NaN` (the pre-game estimate decided
Pivetta was probably out of the game by then; he wasn't, this game's rows
are realized). Fixed by joining the pace on `starting_pitcher_id` (the
realized identity) instead. Post-fix, the same case study shows pace holding
constant at 4.18 across all 5 PAs and the projected pitch count climbing
cleanly: 0 -> 37.6 -> 75.3 -> 112.9 -> 150.5.

## Results (evaluated on val season 2024, 105,265 PAs)

| Model | PR-AUC | ROC-AUC |
|-------|--------|---------|
| v14 (43 features, for reference) | 0.2839 | 0.5999 |
| v15, buggy pace join | 0.2837 | 0.5997 |
| **v15, fixed pace join** | **0.2839** | **0.6002** |

## Game-grain check (batter-game "1+ strikeout")

| Metric | v14 (reference) | v15 buggy | v15 fixed |
|---|---|---|---|
| reliability | 0.00016 | 0.0002 | 0.0002 |
| resolution | 0.01376 | 0.0137 | 0.0138 |
| roc_auc | 0.6367 | 0.6365 | 0.6370 |

The fix nudges every number slightly favorably and edges past v14 on both
PA-grain and game-grain ROC-AUC, but the movement is small enough to be
within normal run-to-run noise — same aggregate-metric ceiling this
project's feature additions have hit since v7.

## Slice-level verification (the decisive check, per v14's own precedent)

`slice_verification.py` fits the v14 feature set (43 features) and the v15
feature set (v14 + the 2 workload features, corrected pace join) on the
IDENTICAL train/val split and fixed XGBoost config, sliced by
`expected_times_through_order` (TTO) — the sharpest cut for this hypothesis,
since `TTO=NaN` rows have **zero** information about in-game depth under the
old feature set (a NaN silently imputed to the training median), and
`TTO=3` rows are compressed (a 19th PA and a 40th PA both read as `TTO=3`).

| Slice | n | Reliability (v14->v15) | Resolution (v14->v15) |
|---|---|---|---|
| ALL PAs | 105,265 | 0.00005 -> 0.00005 (flat) | 0.00328 -> 0.00327 (flat) |
| TTO=1 | 42,505 | 0.00007 -> 0.00007 (flat) | 0.00340 -> 0.00337 (flat/-) |
| TTO=2 | 40,096 | 0.00003 -> 0.00004 (flat) | 0.00299 -> 0.00296 (flat/-) |
| TTO=3 | 12,291 | 0.00005 -> 0.00008 (worse) | 0.00355 -> 0.00366 (better) |
| **TTO=NaN (expected bullpen)** | **10,373** | **0.00024 -> 0.00016 (better)** | **0.00182 -> 0.00195 (better)** |

`TTO=NaN` moves favorably on both metrics — the direction the hypothesis
predicted — but:

**Paired bootstrap, TTO=NaN slice, Brier score (v14 minus v15):** delta =
-0.00001, 95% CI **[-0.00016, +0.00014]** — includes zero.
**Paired bootstrap, TTO=3 slice, Brier score (v14 minus v15):** delta =
+0.00003, 95% CI **[-0.00003, +0.00010]** — also includes zero.

`TTO=3`'s two metrics also point in opposite directions (reliability worse,
resolution better) — the signature of noise being decomposed into
components, not a real effect. Compare v14's own same-hand result last
version: CI [+0.00001, +0.00009], entirely positive, no zero-crossing, on a
comparable sample size (43,769). Here, even on the two slices specifically
chosen because the old features are most information-starved there, neither
CI clears zero.

## Interpretation

**Unlike `platoon_matchup`, this is a genuine negative result, not a hidden
localized win.** The workload features (in-game PA position and a
projected-pitch-count proxy) do not appear to carry real incremental signal
for PA-level strikeout probability once the model already has the pitcher's
season/rolling rate stats and TTO-based role gating — even in the slices
built specifically to give them the best chance. Two candidate explanations,
neither tested here: (1) fatigue/deep-count effects may already be
substantially captured by the rolling K-rate/WHIP features, which implicitly
reflect a pitcher's stuff degrading in general as a start wears on; (2) the
effect may be real but expressed at a different outcome grain (walk rate,
exit velocity) rather than strikeout probability specifically.

**v6's tuned XGBoost remains the standing production candidate.** Given how
clean this negative result is (flat everywhere, including the two slices
purpose-built to favor the hypothesis), this feature thread is closed here
rather than continuing to slice-hunt for a positive result.

## Next steps

- `estimated_team_pa_position`, `pitcher_projected_pitches_before_pa`
  (`build_pitcher_projected_workload`), and the pace-join bug fix are kept as
  real, tested, reusable infrastructure regardless of this flat result — same
  convention as every prior version's new columns (v9's vs-league ratios,
  v11's CSW%, etc.).
- **The pace-join bug pattern (an `expected_pitcher_key_id`/
  `expected_pitcher_role`-gated merge silently substituting bullpen-pooled
  stats for a starter's own, inside a dataset already scoped to REALIZED
  `pitcher_role=='sp'` rows) is worth auditing elsewhere** — any other
  future feature that wants "this specific starter's own recent numbers"
  rather than "the pre-game-safe estimate of whoever's pitching" should join
  on `starting_pitcher_id` directly, not reuse the expected-role-gated merge
  pattern most of `FEATURE_COLS` correctly relies on for other purposes.
- In-game workload as a feature FAMILY is not necessarily dead — this
  version tested position-count and a pitch-count proxy specifically; a
  different operationalization (e.g. this pitcher's OWN this-game running
  K-rate vs. his pre-game baseline) is closer to what `v16_in_game_context`
  (a separate concurrent session) tested, with its own separate flat result
  — see that version's `v16_results.md` for a related but distinct finding.
