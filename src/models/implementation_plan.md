# Implementation Plan — k_predictor v2: opposing-batting-side features

## Context

k_predictor is currently the strongest of five model experiments in this repo — the
only one with a confirmed `real_improvement` verdict (via `summarize_verdict()`) at
game grain, the grain a DK strikeout prop actually resolves on. v1
(`experiments/v1_pitcher_workload/train.py`) added pitcher-side workload features and
came back flat vs. baseline at PA grain. Per `ROADMAP.md`'s own "Next" note on that
result, the untried levers are batter-side signal and `times_through_order`. This plan
scopes v2: add the opposing batter's own rolling K rate, the opposing *team's* rolling
K rate (the whole lineup's recent tendency — new), and `times_through_order`, then
re-run the same PR-AUC + game-grain check v1 used.

Audit gate (2026-08-24): `pytest tests/ -q` — 468 passed, 0 failed. Baseline clean.

## Epic: k_predictor v2 — opposing-batting-side features

Depends on: none (v1 is frozen, per this project's experiment-versioning convention).

---

### Story 1 — Team-level opposing-lineup rolling strikeout rate (new production code)

**Acceptance:** a new function in
`src/models/hit_predictor/processing/features/rolling_stats.py`,
`build_team_batter_strikeout_rolling_feats(pbp, batter_boxscore, window)`, returns one
row per `(batter_team_id, gamepk, game_date, game_season)` with a point-in-time-safe
rolling strikeout rate computed across **starting-lineup batters only**
(`batting_order`-having rows — confirmed decision, excludes pinch hitters/late subs).
Mirrors the existing bullpen-pooling pattern in
`build_pitcher_rolling_stats_all_roles` (pool per-entity-per-game rows into one
team-game row before rolling, since `_rolling_sum`'s `.transform()` needs exactly one
row per entity per game).

Reuses, unmodified: `_batter_pa_outcome_per_game` (already computes `pa_total`/
`pa_strikeout_n` per batter-game), `_create_batting_order` (cross-module import from
`hit_predictor.processing.pipeline` — already precedented, `k_predictor/processing/
pipeline.py` does the same), `_rolling_sum`, `_prefix_stat_cols`, `_rolling_prefix`.
No new per-PA aggregation logic — this is a pooling/re-keying wrapper, same shape as
the bullpen case.

**Layer:** processing / feature engineering.

#### Task 1.1 — Pool starting-lineup batters into one team-game row, then roll

**RED**
- File: `tests/hit_predictor/test_rolling_stats.py`
- Test: `test_build_team_batter_strikeout_rolling_feats_pools_starting_lineup_by_team`
- Setup: synthetic `pbp` — team `T1` has two batters across games `g1`, `g2`;
  matching `batter_boxscore` gives batter A a `batting_order` in both games, batter B
  a `batting_order` only in `g1` (simulates a bench/pinch appearance in `g2` with no
  lineup slot).
- Assert: `g2`'s rolled `team_roll_season_pa_total` reflects only `g1`'s
  starting-lineup batters' summed PA count — batter B's `g2` PAs are excluded from
  the pool (no batting_order that game), and batter B's `g1` PAs (where he *did* have
  a slot) are included.
- Run `pytest tests/hit_predictor/test_rolling_stats.py -k team_batter_strikeout` and
  confirm it fails — `build_team_batter_strikeout_rolling_feats` doesn't exist yet
  (`AttributeError`/`ImportError`).

**GREEN**
- File: `src/models/hit_predictor/processing/features/rolling_stats.py`
- Implement: filter `_batter_pa_outcome_per_game(pbp)` to
  `(gamepk, batter_id)` pairs present in `_create_batting_order(batter_boxscore)`,
  join in `batter_team_id` from `pbp`, `groupby(['batter_team_id', 'gamepk',
  'game_date', 'game_season'])[['pa_total', 'pa_strikeout_n']].sum()`, then
  `_rolling_sum(..., entity_col='batter_team_id', window=window)`, prefix via
  `_rolling_prefix('team', window)`.
- Run the test, confirm it passes.

**REFACTOR**
- Check against the bullpen-pooling code for consistency in naming/shape; no new
  duplication expected since this reuses existing per-game/rolling helpers directly.

#### Task 1.2 — Point-in-time safety (current game excluded from its own rolled value)

**RED**
- Same test file.
- Test: `test_build_team_batter_strikeout_rolling_feats_excludes_current_game`
- Setup: team `T1`, one starting-lineup batter, three games `g1`, `g2`, `g3` with
  known PA/K counts.
- Assert: `g3`'s rolled `pa_total`/`pa_strikeout_n` equal the sum of `g1`+`g2` only —
  `g3`'s own PAs are not included in its own rolled value (mirrors the existing
  bullpen test `test_build_pitcher_rolling_stats_all_roles_pools_bullpen_by_team`'s
  g3 assertion).
- Run and confirm it fails before 1.1's implementation exists; once 1.1's GREEN
  lands, this should pass immediately since `_rolling_sum` already shifts by
  construction — if it does not, that's a real bug in the pooling/grouping step to
  fix here, not a new mechanism to add.

**GREEN**
- Only if 1.2's test fails after 1.1 — fix the grouping/join logic. Expected: no
  code change needed, this task exists to *verify* point-in-time safety, not add it.

**REFACTOR**
- N/A unless a fix was needed above.

#### Task 1.3 — Rate computed from rolled counts, with divide-by-zero guarded to NaN

**RED**
- Same test file.
- Test:
  `test_build_team_batter_strikeout_rolling_feats_rate_divides_rolled_sums_not_per_game_avg`
- Setup: two games with different PA volume (e.g. 3 PA/1 K vs. 6 PA/1 K) for the same
  team, plus a first game with zero prior rolled PA.
- Assert: (a) the rolled `pa_strikeout_rate` for the third game equals
  `(rolled pa_strikeout_n) / (rolled pa_total)` — not an average of the two games'
  individual per-game rates (this project's stated convention throughout
  `rolling_stats.py`: "roll counts, not rates"); (b) the first game's rate is `NaN`,
  not a `ZeroDivisionError` or `inf`.
- Run, confirm it fails (function doesn't compute a rate column yet, or computes it
  wrong).

**GREEN**
- Add `pa_strikeout_rate = pa_strikeout_n / pa_total.replace(0, np.nan)` to the
  rolled frame before prefixing.
- Run, confirm it passes.

**REFACTOR**
- Confirm output column naming matches convention (`team_roll_season_pa_strikeout_rate`,
  `team_roll_season_pa_total`, `team_roll_season_pa_strikeout_n`).

---

### Story 2 — Wire opposing-batter, opposing-team, pitching-team, weather, and TTO
features into k_predictor v2

**Acceptance:** `experiments/v2_batter_and_team_features/train.py` runs end-to-end,
mirroring `v1_pitcher_workload/train.py`'s structure (season split, PA-grain PR-AUC
vs. best naive floor, game-grain aggregation check via `run_pa_vs_game_grain_check` +
`summarize_verdict`), with `FEATURE_COLS` extended by:

- `batter_roll_season_pa_strikeout_rate` — via `rolling_stats.build_pbp_batter_rolling_feats`
  (pure reuse, same function `bb_predictor`'s v1 already used for the batter side).
- `opp_team_roll_season_pa_strikeout_rate` — the opposing (batting) team's rolling K
  rate, via the new `build_team_batter_strikeout_rolling_feats` from Story 1, merged
  on `batter_team_id`.
- `pitching_team_roll_season_pa_strikeout_rate` — the pitcher's own team's rolling K
  rate (their own hitters), from the **same** rolled table Story 1 produces — no
  second function needed, since that table is one row per `(team_id, gamepk)`
  regardless of role (every team bats every game it plays). Merged a second time on
  `pitcher_team_id` instead of `batter_team_id`, with distinct output column names
  to avoid a collision on the double merge. Flagged as more speculative than the
  opposing-team feature above — it's an indirect proxy for park/weather/umpire
  "game environment" effects (and also picks up team-quality confounds unrelated to
  today's specific matchup) rather than a direct signal about who the pitcher is
  facing. Track its feature importance separately in the results writeup rather than
  assuming it behaves like the opposing-team feature.
- `weather_condition`, `weather_temp` — already merged into `pa_outcome` via
  `game_info` in `create_pa_outcome_strikeout` (confirmed by reading
  `k_predictor/processing/pipeline.py`), just not previously added to
  `FEATURE_COLS`. Zero new merge work — the more direct test of the same
  "game environment" hypothesis the pitching-team-K-rate feature above only proxies
  for.
- `expected_times_through_order` — already a byproduct of
  `expected_role.assign_expected_pitcher_role`, just not previously added to
  `FEATURE_COLS`; confirmed present on `pa_outcome` already, no new merge needed.

No new filtering logic needed for "starters only" — `create_pa_outcome_strikeout`
already inner-joins `batting_order` (confirmed by reading
`k_predictor/processing/pipeline.py`).

**Layer:** experiments. No dedicated test file — this project's established
convention (confirmed: no test file exists for `v1_pitcher_workload/train.py` or any
other `experiments/v{N}_*/train.py` in this repo) treats these as reproducible,
runnable scripts evaluated by their own printed PR-AUC/game-grain output, not by
pytest. The building blocks they wire together (Story 1's new function, and the
already-tested `build_pbp_batter_rolling_feats`) carry the real test coverage.

#### Task 2.1 — Build `experiments/v2_batter_and_team_features/train.py`

- Copy `v1_pitcher_workload/train.py`'s structure (sections 1–9: config, load, build
  PA-grain df, feature merges, split, train/eval, print results, game-grain check,
  plots, MLflow logging).
- Add the new feature merges (batter rolling K rate, opposing-team rolling K rate,
  pitching-team's-own rolling K rate via a second merge of the same table, weather
  columns, TTO) alongside v1's existing pitcher-side merges — do not remove v1's
  pitcher workload features, this experiment is additive (season/rolling/shrunk WHIP
  + the new ones), per the versioned-experiment convention (each version is a
  frozen, standalone snapshot).
- Run: `python experiments/v2_batter_and_team_features/train.py` from
  `src/models/k_predictor/`. Requires AWS credentials with S3 read access
  (`s3://mlbdk`, us-east-2).
- Report: PA-grain PR-AUC vs. v1's 0.2732 and baseline's 0.2702, plus the game-grain
  `summarize_verdict()` result vs. the naive floor — same reporting shape as v1's
  Section 6/7 printouts.

---

## Verification — DONE 2026-08-24

1. `pytest tests/hit_predictor/test_rolling_stats.py -k team_batter_strikeout -v` —
   all three new tests passed on first GREEN attempt.
2. `pytest tests/ -q` — 471 passed, 0 failed. (Caught and fixed one real bug along
   the way: a fixture name collision — a new `_batter_box_row` test helper appended
   at the bottom of the file silently shadowed an unrelated, pre-existing helper of
   the same name used by `build_batter_rolling_stats`'s own tests, since Python
   resolves a module-level `def` to whichever definition ran last. Renamed the new
   one to `_lineup_slot_row`. This is exactly why "run the full suite, not just the
   new test" matters — task-scoped testing alone would have missed it.)
3. `python experiments/v2_batter_and_team_features/train.py` — ran to completion
   against real S3 data (2018/2019/2022/2023 train, 2024 val, 105,265 PAs).
4. Result vs. v1 (PR-AUC 0.2732, game-grain `real_improvement`, reliability 0.0002,
   resolution 0.0115): **PA-grain LR PR-AUC 0.2816 (+0.0084), ROC-AUC 0.5981.
   Game-grain verdict `real_improvement` again — resolution 0.0131 (+0.0016 vs.
   v1), reliability flat at 0.0002.** Feature importance: `batter_roll_season_pa_
   strikeout_rate` is the #2 feature overall, `expected_times_through_order` is #4
   — both real. The two team-level features and weather rank low — present but not
   load-bearing. Logged to `ROADMAP.md`'s Mid-term k_predictor entry.

## Before marking any task complete:
- [x] Watched the test FAIL before writing any code
- [x] Test failed for the right reason (function/feature missing, not a typo)
- [x] Wrote minimal code — nothing extra
- [x] All tests pass after GREEN
- [x] No new warnings or errors in pytest output (one pre-existing regression
      surfaced and fixed — see Verification #2 above)
- [x] Tests use real code — mocks only where I/O is unavoidable (none needed here —
      all synthetic in-memory DataFrames, no S3/AWS calls in the unit tests)
