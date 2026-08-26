# Implementation Plan — batters_faced_predictor

**Goal:** stop treating `expected_batters_faced` as a shrinkage average; build it as
its own predictive regression model, following this repo's baseline-first pattern.
See ROADMAP.md's k_predictor Mid-term entries (2026-08-25/26) for the full
background on why (Story-0 cascade floor numbers, the shelved uncertainty-modeling
attempt, and the bf_gap-quartile isolation showing this is an ROI bet, not a
ceiling-driven necessity).

**New sibling model:** `src/models/batters_faced_predictor/` — same shape as
`n_pa_predictor`/`bb_predictor`/`k_predictor`/`short_outing_predictor`, grain
matches `short_outing_predictor` (one row per (personId, gamepk) SP start).

**Frozen, not touched:** `game_context.py`'s `build_expected_batters_faced`,
`build_pitcher_start_pa_this_season` stay exactly as-is — other models depend on
their current behavior (same reasoning already documented for
`build_expected_start_innings`). The new model's own label/feature-assembly code
lives entirely in the new sibling directory + one new shared function.

---

## Epic 1 — Start-grain label: realized_batters_faced

Why: every sibling model's first step is building its own grain's label from raw
pbp/boxscore. This one reuses hit_predictor's existing
`rolling_stats._pitcher_pa_outcome_per_game` (already the source k_predictor's
count-distribution scripts use for the same number) rather than recomputing it.

STORY 1.1 — `create_start_pa_outcome()` in `batters_faced_predictor/processing/pipeline.py`
Acceptance: one row per (personId, gamepk) REALIZED sp start, with
`realized_batters_faced`, `game_date`, `game_season`. Bullpen appearances excluded
by construction (same as short_outing_predictor's `create_start_outcome` — a
bullpen boxscore/pbp row isn't a "start").
Layer: processing

TASK 1.1.1 — label a start's realized batters faced
  RED
    - File: tests/batters_faced_predictor/test_pipeline.py
    - Test: test_create_start_pa_outcome_labels_realized_batters_faced
    - Assert: 3 distinct PAs (by play_id) for one sp start -> realized_batters_faced == 3
    - Run: pytest tests/batters_faced_predictor/test_pipeline.py -> FAILS (ModuleNotFoundError / ImportError, function doesn't exist yet)
  GREEN
    - File: src/models/batters_faced_predictor/processing/pipeline.py
    - Filter pbp to pitcher_role=='sp', reuse `rolling_stats._pitcher_pa_outcome_per_game(sp_pbp, entity_col='pitcher_id')`, rename pitcher_id->personId, pa_total->realized_batters_faced
  REFACTOR
    - none expected — thin wrapper

TASK 1.1.2 — excludes bullpen role PAs
  RED: test_create_start_pa_outcome_excludes_bullpen_role_pas — mixed sp/bullpen pbp, assert only the sp pitcher_id appears in result
  GREEN: covered by the pitcher_role=='sp' filter already added in 1.1.1 — should pass immediately; if not, fix filter
  REFACTOR: n/a

TASK 1.1.3 — returns expected columns
  RED: test_create_start_pa_outcome_returns_expected_columns — assert exact column set
  GREEN: trim the returned frame to the documented columns
  REFACTOR: n/a

---

## Epic 2 — New shared feature: opposing lineup on-base/walk rate

Why: candidate feature named in the task brief. No existing team-level on-base/walk
rate function — `build_team_batter_strikeout_rolling_feats` (rolling_stats.py) is
the closest analog (built for k_predictor v2) but strikeout-only. Mirrors it
exactly with pa_walk_n/pa_hit_n/pa_hbp_n instead of pa_strikeout_n — both already
computed by `_batter_pa_outcome_per_game`, no new aggregation needed there.
Lives in hit_predictor's shared `rolling_stats.py` (target-agnostic infra), not
duplicated locally — same precedent as k_predictor's own addition.

STORY 2.1 — `build_team_batter_onbase_rolling_feats()` in
`hit_predictor/processing/features/rolling_stats.py`
Acceptance: one row per (batter_team_id, gamepk): rolled pa_total, pa_walk_n,
pa_hit_n, pa_hbp_n, walk_rate, on_base_rate (hit+walk+hbp / pa_total) — pooled
starting-lineup-only, point-in-time safe, same as the strikeout version.
Layer: processing (shared feature)

TASK 2.1.1 — pools starting lineup by team, computes walk_rate/on_base_rate from rolled sums
  RED
    - File: tests/hit_predictor/test_rolling_stats.py
    - Test: test_build_team_batter_onbase_rolling_feats_rate_divides_rolled_sums_not_per_game_avg
      (mirrors the strikeout version's own test at line ~1190 — same "roll sums,
      divide once" contract, first game NaN not inf/zero-division)
    - Run: pytest tests/hit_predictor/test_rolling_stats.py -> FAILS (function doesn't exist)
  GREEN
    - Copy build_team_batter_strikeout_rolling_feats's shape, swap in
      pa_walk_n/pa_hit_n/pa_hbp_n, add walk_rate + on_base_rate columns
  REFACTOR
    - Only if real duplication emerges between the two functions worth a shared
      helper (e.g. a `_build_team_batter_rolling_feats(pbp, batter_boxscore, window, count_cols, rate_defs)` factored out) — do this ONLY if both versions
      still work after refactor (tests green for both), otherwise leave the
      duplication (matches this file's existing convention of near-duplicate
      per-target aggregation blocks rather than premature generalization)

TASK 2.1.2 — excludes non-starters (pinch-hit appearances) from the team pool
  RED: test_build_team_batter_onbase_rolling_feats_pools_starting_lineup_by_team
    (mirrors strikeout version's own test)
  GREEN: reuse `_create_batting_order` inner-join, same as strikeout version
  REFACTOR: n/a

---

## Epic 3 — Baseline: does a regressor beat the cascade using the SAME inputs?

Why: the real bar here is the cascade's own accuracy (MAE 2.861, RMSE 3.951,
Bias +0.073, Pearson r 0.495 on val 2024), not a global-mean naive. Per this
repo's established two-stage pattern (baseline = reuse existing signals only,
v1 = add real new engineering), the baseline model gets exactly the inputs the
cascade formula already has access to (pitcher/team/league last-season avg PA,
this-season rolling avg PA + starts_n, expected_batters_faced/_weight itself) and
asks: does letting XGBoost/LR combine them non-linearly already beat the
hand-tuned shrinkage formula, before any new feature engineering?

STORY 3.1 — `baseline/model/run.py`, scaffolded via the baseline-model skill
Acceptance: `baseline_results.md` with MAE/RMSE/Bias/Pearson r for
(a) the cascade itself as a "model" row, (b) LR, (c) XGBoost — evaluated on val
season 2024, plus the same expected_batters_faced_weight-quartile stratification
already used for the cascade's Story-0 capture (for direct comparability).
Layer: experiments

No RED/GREEN here — matches every sibling model's own convention:
`baseline/model/run.py` (short_outing_predictor, k_predictor, bb_predictor,
n_pa_predictor) is an S3-driven analysis script, not unit tested; only
`processing/pipeline.py` and shared feature functions get TDD. `config.yaml`
target_column = realized_batters_faced, task_type: regression (first regression
config in this repo — n_pa_predictor's `low_pa` is classification even though
n_pa itself is a count, this is genuinely the first regression target/model_name).

---

## Epic 4 — v1: add real feature engineering, re-run bf_gap-quartile isolation

Why: baseline (Epic 3) tests "same info, better combiner." v1 tests the actual
hypothesis — new information beyond what the cascade sees.

STORY 4.1 — v1 experiment: opposing on-base/walk rate (Epic 2), team rest days
(`game_context.build_team_rest_days`, already exists), home/away (derived from
schedule's home_id/away_id vs pitcher_team_id), handedness (`pitcher_throw_hand`,
already exists), pitch-efficiency proxy (`pitch_count_avg` from
`build_pbp_pitcher_rolling_feats`, season-to-date — a trailing-N trend delta is a
further refinement, not attempted here, flagged as a "next" if this doesn't move
the floor).
Layer: experiments — same "no unit tests, S3-driven script" convention as Epic 3.

STORY 4.2 — bf_gap-quartile floor re-check with the NEW estimate
Acceptance: rerun `k_predictor/experiments/count_distribution_check/run.py`'s
bf_gap-quartile isolation method (BENCHMARKS.md's documented general method) with
`expected_batters_faced` replaced by the new model's prediction, to see whether
the closest-quartile floor MAE (currently 1.724) itself moves — per the user's
explicit caveat, it may not (that would still be a real, worthwhile win if
bf_gap-driven error specifically shrinks).

STORY 4.3 — record result in ROADMAP.md
Append a new dated entry to the k_predictor Mid-term section (do not rewrite
history) with the baseline + v1 numbers, verdict, and next steps — same
convention as every other entry there.

---

## Epic 5 — v2: recent-workload/fatigue features (post-launch, driven by residual error analysis)

Why: `experiments/residual_error_analysis/run.py` (run against the tuned XGBoost from
v1_opposing_traffic_and_rest, val MAE 2.7347) found two systematic failure modes, not
just noise:
  1. UNDER-prediction: the 0-2-starts-this-season bucket is the worst segment in val
     (MAE 3.246 vs 2.578 for 11+ starts, +0.694 bias) — cold-start pitchers who go
     deep, efficient outings the cascade's near-zero-weight fallback can't see coming.
  2. OVER-prediction: established starters (cascade weight 0.4-0.86 — high confidence)
     pulled after 0-2 IP despite clean box lines (H/R/ER near 0) — the signature of an
     in-game injury exit, rain delay, or planned short/piggyback outing. Nothing in
     the current feature set carries recent-start TREND, only season-to-date averages,
     which react slowly.
Full worked examples and bucket breakdown in `experiments/residual_error_analysis/error_analysis.md`.

Two features target failure mode #2 directly (#1 is a separate, harder cold-start
problem — deferred, see error_analysis.md's candidate #3):
  (a) trailing-3-start PA/start trend — recent form vs season baseline
  (b) pitcher's own rest days (not team rest days, which is ~always 1) + a
      pitches-thrown workload-density signal, shrunk by this-season sample size

Column-naming note: `interaction_feats.py`'s `find_rolling_trend_pairs`/
`build_trend_features`/`build_shrinkage_weight_features` are regex-driven around
`rolling_stats.py`'s `*_roll_last{N}g_*`/`*_roll_season_*` naming — `game_context.py`
uses a different prefix convention (`pitcher_last{N}_start_pa_*` /
`pitcher_this_season_start_pa_*`) that the regex won't match. Renaming
`game_context.py`'s columns to fit would touch every existing caller across all
sibling models — out of scope. Trend ratio/direction and the shrinkage weight are
computed as plain arithmetic in `train.py` instead (same formulas, no forced reuse).
`build_expected_batters_faced` already sets this precedent — its own
`shrinkage_weight = starts_n / (starts_n + k)` is computed inline, not factored into
a shared helper.

STORY 5.1 — `build_pitcher_rest_days()` in
`hit_predictor/processing/features/game_context.py`
Acceptance: one row per (personId, gamepk) SP start with
`pitcher_days_since_last_start` — calendar days since that pitcher's own previous
**SP** start only (a relief appearance in between doesn't count as "his last start"),
NaN for his first SP start in the data. Mirrors `build_team_rest_days`'s shape
(sort by `game_datetime`, `.diff()` per entity) but keyed on `personId` instead of
`team_id`, and scoped to `pitcher_role == 'sp'` like every other start-grain builder
in this file.
Layer: processing (shared feature)

TASK 5.1.1 — computes calendar days since pitcher's own last SP start
  RED
    - File: tests/hit_predictor/test_game_context.py
    - Test: test_build_pitcher_rest_days_computes_calendar_days_since_last_start
    - Fixture: two sp pbp rows for the same pitcher_id, game_date 4 days apart
      (mirror `_start_pa_this_season_pbp_row`'s shape)
    - Assert: second start's `pitcher_days_since_last_start == 4`
    - Run: pytest tests/hit_predictor/test_game_context.py -> FAILS (function doesn't exist)
  GREEN
    - File: src/models/hit_predictor/processing/features/game_context.py
    - Filter pbp to `pitcher_role == 'sp'`, build one row per (personId, gamepk,
      game_date, game_datetime) via the same last-pitch-per-PA groupby
      `build_pitcher_start_pa_this_season` already uses, sort by
      ['personId', 'game_datetime'], `.groupby('personId')['game_date'].diff().dt.days`
  REFACTOR
    - none expected — thin, mirrors build_team_rest_days

TASK 5.1.2 — first SP start is NaN
  RED: test_build_pitcher_rest_days_first_start_is_nan — one pitcher, one start ->
    pd.isna(result['pitcher_days_since_last_start'])
  GREEN: covered by `.diff()`'s natural NaN on a group's first row — confirm, no new code
  REFACTOR: n/a

TASK 5.1.3 — only counts SP starts, ignores a relief appearance by the same pitcher in between
  RED: test_build_pitcher_rest_days_ignores_bullpen_appearances_between_starts —
    pitcher starts g1 (day 0), relieves in g2 (day 2, pitcher_role='bullpen'),
    starts g3 (day 5) -> g3's `pitcher_days_since_last_start == 5` (from g1), not 3
  GREEN: covered by the `pitcher_role == 'sp'` filter in 5.1.1 — should pass
    immediately; if not, fix the filter
  REFACTOR: n/a

STORY 5.2 — `build_pitcher_workload_density()` in
`hit_predictor/processing/features/game_context.py`
Acceptance: one row per (personId, gamepk) SP start with `pitcher_last_start_pitches`
(shift(1) of that pitcher's previous SP start's pitch count, from
`pitcher_boxscore`'s `p` column) and `pitcher_workload_density` =
`pitcher_last_start_pitches / pitcher_days_since_last_start` (from Story 5.1's
output, passed in — same composition style as `build_expected_batters_faced` taking
pre-built pieces), NaN-guarded (not inf) when rest days is 0 or NaN. Pitches thrown,
not batters faced — pitch count is the actual lever managers pull on for a pull
decision, and batters-faced recency is already covered by Story 5.3's trend feature.
The shrunk version (`* starts_n/(starts_n+k)`) is glue in `train.py`, not part of
this function — same precedent as `build_expected_batters_faced`'s own inline
shrinkage weight.
Layer: processing (shared feature)

TASK 5.2.1 — computes last-start pitch count via shift(1)
  RED
    - Test: test_build_pitcher_workload_density_carries_forward_last_start_pitch_count
    - Fixture: pitcher_boxscore with two SP starts for the same personId, p=90 then p=100
    - Assert: second start's `pitcher_last_start_pitches == 90`
    - Run: FAILS (function doesn't exist)
  GREEN: sp-scoped pitcher_boxscore rows, sort by ['personId','game_datetime'],
    `.groupby('personId')['p'].shift(1)`
  REFACTOR: none expected

TASK 5.2.2 — first start's last_start_pitches is NaN
  RED: test_build_pitcher_workload_density_first_start_is_nan
  GREEN: covered by shift(1)'s natural NaN on a group's first row — confirm
  REFACTOR: n/a

TASK 5.2.3 — workload_density divides pitches by rest days, guards zero/NaN rest days
  RED: test_build_pitcher_workload_density_divides_by_rest_days_and_guards_zero
    - pitches=90, rest_days=3 -> density == 30.0
    - pitches=90, rest_days=0 (same-day doubleheader edge case — cannot happen for
      the SAME pitcher starting twice, but a relief-appearance-adjacent start could
      still land rest_days==0 if data is malformed; guard defensively anyway,
      matching `.replace(0, np.nan)`'s existing convention elsewhere in this file)
      -> pd.isna(density), not inf
  GREEN: `density = last_start_pitches / rest_days.replace(0, np.nan)`
  REFACTOR: n/a

STORY 5.3 — v2 experiment: wire trend + rest + workload density into
`experiments/v2_workload_and_rest/train.py`
Acceptance: same tuned-XGBoost hyperparameters as v1 (`n_estimators=2000,
learning_rate=0.02, max_depth=3, min_child_weight=10, subsample=0.8,
colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5, early_stopping_rounds=50`,
carried over unchanged), v1's full feature set plus:
  - `pitcher_last3_start_pa_avg_pa_per_start` /`_starts_n` via
    `build_pitcher_start_pa_this_season(pbp, window=3)` (already exists, no new
    production code)
  - `pa_trend_ratio = pitcher_last3_start_pa_avg_pa_per_start / pitcher_this_season_start_pa_avg_pa_per_start`,
    `pa_trend_direction = sign(last3 - this_season)` — plain arithmetic in train.py,
    see the naming note above
  - `pitcher_days_since_last_start` (Story 5.1)
  - `pitcher_workload_density`, `pitcher_workload_density_shrunk = pitcher_workload_density * (pitcher_this_season_start_pa_starts_n / (pitcher_this_season_start_pa_starts_n + k))`,
    k=5.0 matching `PA_SHRINKAGE_K` already used elsewhere in this pipeline (Story 5.2)
Evaluated the same way as v1 (MAE primary vs. v1's own tuned-XGBoost floor 2.7347,
RMSE/Bias/Pearson r secondary, same expected_batters_faced_weight-quartile
stratification, same bf_gap-quartile floor re-check). Writes `v2_results.md`.
Layer: experiments — same "S3-driven analysis script, not unit tested" convention as
Epics 3/4 (`baseline/model/run.py`, `v1_opposing_traffic_and_rest/train.py`): only
`processing/pipeline.py` and shared feature functions (Stories 5.1/5.2 above) get TDD.

No RED/GREEN here — matches Epic 3/4's stated convention.

STORY 5.4 — record result in ROADMAP.md
Append a new dated entry to the k_predictor/batters_faced_predictor Mid-term section
(do not rewrite history) with the v2 numbers, verdict, and whether the trend/workload
features actually moved the over-prediction bucket specifically (re-run the
worked-examples/bucket cut from `residual_error_analysis/run.py` against v2's
predictions, not just the aggregate MAE) — same convention as Story 4.3.

---

## Epic 6 — v3: cold-start pitchers (multi-season lookback), then wrap this thread

Why: v2 (Epic 5) targeted the OVER-prediction failure mode and won (MAE 2.6471,
-0.0876 vs v1). The UNDER-prediction failure mode from
`residual_error_analysis/error_analysis.md` — the 0-2-starts-this-season bucket,
MAE 3.246 vs 2.578 for 11+ starts, +0.694 bias — is still open. Roughly half of
those worst under-prediction worked examples DO have a valid
`pitcher_last_season_start_pa_avg_pa_per_start` (not NaN) yet still miss badly at
low starts_n, so this isn't purely a missing-data problem — but the NaN subset is
real and directly fixable: `season_stats.build_pitcher_start_pa_stats` (frozen, not
touched — other models depend on its current behavior) only looks exactly one
season back via `_shift_to_last_season`, so a pitcher who missed a full season
(injury, demotion, rehab) gets NaN there and `build_expected_batters_faced` falls
straight to team/league average, even if he has a perfectly good track record from
two or three seasons ago.

STORY 6.1 — `build_pitcher_last_known_season_start_pa()` in
`hit_predictor/processing/features/season_stats.py`
Acceptance: one row per (personId, game_season) with
`pitcher_last_known_season_start_pa_avg_pa_per_start` /`_n_starts` — that pitcher's
stats from the most recent PRIOR season in which he actually started, not
necessarily the immediately-prior one (skips over a gap season with zero starts).
NaN if he has never started in any prior season in the data. Built via
`pd.merge_asof(..., direction='backward', allow_exact_matches=False)` against
`_create_pitcher_start_pa_stats(pbp)`'s raw (un-shifted) per-season table, matched
per personId against every season that appears anywhere in the data — a strictly
more correct lookback than `_shift_to_last_season`'s fixed +1 for this specific use
case, so implemented as a new standalone function rather than a parameter on the
frozen one.
Layer: processing (shared feature)

TASK 6.1.1 — looks back across a gap season to find the last season with starts
  RED
    - File: tests/hit_predictor/test_season_stats.py
    - Test: test_build_pitcher_last_known_season_start_pa_looks_back_across_a_gap_season
    - Fixture: pitcher '10' starts in 2021 only (no 2022, no 2023 starts of his own);
      a different pitcher's pbp row in 2023 is needed so game_season=2023 appears in
      the global season list at all
    - Assert: result row for (personId='10', game_season=2023) carries his 2021
      avg_pa_per_start/n_starts, not NaN — this is exactly what
      build_pitcher_start_pa_stats (fixed +1 shift) CANNOT do, since he has no 2022
      row to shift from
    - Run: pytest tests/hit_predictor/test_season_stats.py -> FAILS (function doesn't exist)
  GREEN
    - File: src/models/hit_predictor/processing/features/season_stats.py
    - Cross-join every personId in `_create_pitcher_start_pa_stats(pbp)` against
      every game_season present anywhere in pbp, `pd.merge_asof` backward,
      `allow_exact_matches=False`, rename via `_prefix_stat_cols`
  REFACTOR
    - none expected

TASK 6.1.2 — no prior starts at all is NaN (true rookie's first-ever start)
  RED: test_build_pitcher_last_known_season_start_pa_no_prior_starts_is_nan —
    pitcher's only pbp row is in game_season=2023 -> result row for
    (personId, game_season=2023) is NaN (nothing strictly before 2023 for him)
  GREEN: covered by merge_asof's natural no-match -> NaN behavior — confirm
  REFACTOR: n/a

STORY 6.2 — v3 experiment: wire it into
`experiments/v3_cold_start_lookback/train.py`
Acceptance: same tuned-XGBoost hyperparameters as v1/v2 (unchanged), v2's full
feature set plus `pitcher_last_known_season_start_pa_avg_pa_per_start` /`_n_starts`.
Evaluated the same way as v1/v2 (MAE primary vs. v2's own tuned-XGBoost floor
2.6471, same quartile stratification, same bf_gap-quartile floor re-check), PLUS a
targeted re-check of the SPECIFIC bucket this pass targets: the 0-2-starts-this-season
bf_gap, cascade vs. new best model (mirrors Story 5.2's established-starter
re-check, same method, opposite bucket). Writes `v3_results.md`.
Layer: experiments — same "S3-driven analysis script, not unit tested" convention
as every prior experiments Epic — only Story 6.1's shared feature function gets TDD.

No RED/GREEN here — matches Epic 3/4/5's stated convention.

STORY 6.3 — record result in ROADMAP.md and close out this session's thread
Append a new dated entry (do not rewrite history) with the v3 numbers, verdict, and
whether the cold-start bucket specifically moved — same convention as Story 5.4.
This is the last planned story for this batters_faced_predictor error-analysis
thread (residual analysis -> v2 -> v3) — note in the entry that a production
switch-over decision for `build_expected_batters_faced` is still open and deferred
to a future session, along with anything v3 doesn't fully resolve.

---

## Before marking any task complete:
- [ ] Watched the test FAIL before writing any code
- [ ] Test failed for the right reason (feature missing, not a typo)
- [ ] Wrote minimal code — nothing extra
- [ ] All tests pass after GREEN
- [ ] No new warnings or errors in pytest output
- [ ] Tests use real code — mocks only where I/O is unavoidable (none needed here;
      pure pandas transforms throughout, same as every sibling model's processing
      layer)
