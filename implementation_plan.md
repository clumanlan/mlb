# Implementation Plan — Team-pooled bullpen features (fix serving-time leakage)

## Context

The prior fix (join on `pitcher_id` + `pitcher_role`, see git history) made bullpen-role PAs pick
up the correct role's stats instead of always the SP's — but still keyed those stats by
INDIVIDUAL `pitcher_id`. User caught a deeper issue: this is a pre-game daily-batch prop-betting
system (CLAUDE.md). The starting pitcher is known pre-game; which specific reliever will face a
given batter is not. Any feature conditioned on an individual `pitcher_id` is fine for SP-role
PAs but relies on information unavailable at serving time for bullpen-role PAs — real, even
though it's not classic train/val leakage (both splits are historical completed games where the
true pitcher_id is always known after the fact, so validation metrics look fine without
reflecting real deployment accuracy for bullpen PAs).

Confirmed precedent: `src/features/transforms/bullpen_pitcher_base.py` (a separate, older
Feast-layer feature store, not part of `hit_predictor`) already pools bullpen stats
`PARTITION BY team_id` in its SQL. Correct pattern, never applied in this experiment pipeline.

Scope, confirmed with user: thread an `entity_col` parameter through the pitcher-stat builders
in `season_stats.py`/`rolling_stats.py` so bullpen rows can be pooled by `pitcher_team_id`
instead of `pitcher_id`, covering BOTH the pbp-derived stats (already role-tagged from the prior
fix) and the boxscore-derived stats (`pitcher_season_stats`/`pitcher_rolling_*_stats` — never
role-aware at all until now, same bug, different code path). Also null `pitcher_throw_hand` for
bullpen rows (a specific reliever's hand isn't knowable pre-game either).

Key design finding from reading the actual source (not just the sketch): `_rolling_sum`/
`_rolling_max`/`_rolling_pooled_std` in rolling_stats.py are already entity-agnostic — only the
per-game aggregation layer needs threading. And `_pitcher_last_inning_per_game`'s hardcoded
`.groupby(['pitcher_id', 'gamepk'])['play_id'].idxmax()` generalizes to `entity_col` with no
semantic redesign: at `entity_col='pitcher_team_id'` it naturally means "the last pitch thrown by
any reliever on that team that game" — exactly the right team-level meaning, for free.

Testing strategy: `entity_col` defaults to `'pitcher_id'`/`'personId'` everywhere and must
reproduce current behavior exactly — the existing 246-test suite is the regression safety net
for that, no new tests needed to prove "unchanged default behavior." New tests only need to
prove `entity_col='pitcher_team_id'`/`'team_id'` pools correctly, tested through the top-level
public functions (not every private helper individually) — those exercise the full chain.

---

## EPIC 1: `season_stats.py` — entity_col for pbp-derived pitcher stats
Why: lets the bullpen half of `build_pbp_pitcher_feats_all_roles` pool by team instead of
individual pitcher.
Depends on: none

### STORY 1.1: `entity_col` threaded through the pbp-derived builder chain
Acceptance: `build_pbp_pitcher_feats(pbp, pitcher_role='bullpen', entity_col='pitcher_team_id')`
produces one row per (team, season) pooling ALL bullpen pitchers' pbp rows for that team,
computed with the same formulas as the pitcher_id-keyed version. Default calls (no entity_col)
behave identically to today — covered by the existing test suite, not re-tested here.
Layer: processing
Files: `src/models/hit_predictor/processing/features/season_stats.py`
Tests: `tests/hit_predictor/test_season_stats.py`

TASK 1.1.1: Thread `entity_col` through the 5 `_create_pitcher_*` helpers + `build_pbp_pitcher_feats`
  RED:
    - Test: `test_build_pbp_pitcher_feats_pools_bullpen_by_team_when_entity_col_given`
    - Fixture: extend `_make_full_pitcher_pbp()` (or a variant) with a SECOND bullpen pitcher
      (pitcher_id='2') on the SAME team_id and game as pitcher_id='1's existing bullpen PA (both
      `pitcher_team_id='T1'`), so pooling has something real to prove — if entity_col isn't
      wired through, the two pitchers stay separate rows instead of one team row.
    - `result = build_pbp_pitcher_feats(pbp, pitcher_role='bullpen', entity_col='pitcher_team_id')`
    - Assert: `len(result) == 1` (one row for team_id='T1', not two for pitcher_id='1'/'2');
      `'pitcher_team_id' in result.columns` and `'pitcher_id' not in result.columns`;
      `pitcher_last_season_pa_total == 2` (both bullpen PAs counted into the pooled team row).
    - Run: `pytest tests/hit_predictor/test_season_stats.py -k pools_bullpen_by_team` → confirm
      FAILS (currently hardcodes 'pitcher_id', so entity_col isn't even an accepted kwarg yet —
      TypeError).
  GREEN:
    - In each of `_create_pitcher_stuff_command_stats`, `_create_pitcher_pa_outcome_stats`,
      `_create_pitcher_last_inning_stats`, `_create_pitcher_pitch_count_stats`,
      `_create_pitcher_contact_quality_stats`: add `entity_col: str = 'pitcher_id'` param,
      change `group_cols = ['pitcher_id', 'game_season']` → `group_cols = [entity_col, 'game_season']`
      (keep `+ (extra_group_cols or [])` unchanged). `_create_pitcher_pitch_count_stats`'s inner
      `game_pitch_totals` groupby also hardcodes `['pitcher_id', 'gamepk', 'game_season']` —
      change to `[entity_col, 'gamepk', 'game_season']`.
    - `build_pbp_pitcher_feats(pbp, pitcher_role=None, entity_col='pitcher_id')`: pass
      `entity_col` to each of the 5 helper calls; change the 4 `.merge(..., on=['pitcher_id','game_season'])`
      calls to `on=[entity_col, 'game_season']`.
  REFACTOR: none expected — mechanical parameter threading.

### STORY 1.2: `build_pbp_pitcher_feats_all_roles` pools bullpen by team
Acceptance: bullpen half now team-pooled; both halves share a common `pitcher_key_id` column so
they can be concatenated and later joined onto PA-level data uniformly.
Layer: processing
Files: same as above.

TASK 1.2.1: Update `build_pbp_pitcher_feats_all_roles`
  RED:
    - Test: `test_build_pbp_pitcher_feats_all_roles_pools_bullpen_by_team`
    - Same 2-bullpen-pitcher-same-team fixture as 1.1.1.
    - Assert: result has a `pitcher_key_id` column (not separate `pitcher_id`/`pitcher_team_id`
      columns); the 'sp' row's `pitcher_key_id` equals the sp pitcher's `pitcher_id`; the
      'bullpen' row's `pitcher_key_id` equals `'T1'` (the team_id), and there's exactly ONE
      bullpen row (pooled), not one per individual reliever.
    - Run pytest, confirm FAILS (current implementation still outputs a `pitcher_id` column for
      both halves, no pooling).
  GREEN:
    - `sp = build_pbp_pitcher_feats(pbp, pitcher_role='sp').rename(columns={'pitcher_id': 'pitcher_key_id'}).assign(pitcher_role='sp')`
    - `bullpen = build_pbp_pitcher_feats(pbp, pitcher_role='bullpen', entity_col='pitcher_team_id').rename(columns={'pitcher_team_id': 'pitcher_key_id'}).assign(pitcher_role='bullpen')`
    - `return pd.concat([sp, bullpen], ignore_index=True)`
  REFACTOR: check the OLD test from the prior fix
    (`test_build_pbp_pitcher_feats_all_roles_tags_and_stacks_both_roles`) still passes — it
    asserted on `pitcher_role` values and stat values, not on a `pitcher_id` column name, so it
    should be unaffected, but confirm by running the full file, not just the new test.

---

## EPIC 2: `season_stats.py` — team-pooled boxscore-derived pitcher stats
Why: `pitcher_season_stats` has NEVER been role-aware — same serving-time bug as the pbp-derived
stats had, on a different code path (`pitcher_boxscore` has no `pitcher_role` column of its own).
Depends on: none (independent of Epic 1, but both land before Epic 5/train.py)

### STORY 2.1: Shared role lookup
Acceptance: one row per (gamepk, pitcher_id) giving that pitcher's role and team that game,
derived from pbp, reusable by both season_stats.py and rolling_stats.py.
Layer: processing
Tests: `tests/hit_predictor/test_season_stats.py`

TASK 2.1.1: `_pitcher_role_lookup(pbp)`
  RED:
    - Test: `test_pitcher_role_lookup_returns_one_row_per_pitcher_per_game`
    - Use `_make_full_pitcher_pbp()` (pitcher_id='1': 2 SP rows in gamepk='1', 1 bullpen row in
      gamepk='2' — multiple pbp rows per game already, good for proving dedup).
    - Assert: `len(result) == 2` (one row per (gamepk, pitcher_id), not one per pbp row);
      columns include `gamepk`, `pitcher_id`, `pitcher_team_id`, `pitcher_role`; the gamepk='2'
      row has `pitcher_role == 'bullpen'`.
    - Run pytest, confirm FAILS (`ImportError`/`AttributeError`, function doesn't exist). NOTE:
      `_make_full_pitcher_pbp()` doesn't currently set `pitcher_team_id` — add it to the fixture
      (e.g. `'pitcher_team_id': 'T1'` on every row) as part of this task's RED step, since this
      is the first test that needs it.
  GREEN:
    - `def _pitcher_role_lookup(pbp): return pbp[['gamepk', 'pitcher_id', 'pitcher_team_id', 'pitcher_role']].drop_duplicates()`
  REFACTOR: none expected.

### STORY 2.2: `build_pitcher_stats_all_roles`
Acceptance: sp rows aggregated by pitcher_id as today; bullpen rows aggregated by
pitcher_team_id; both tagged and concatenated under a common `pitcher_key_id` column, same
`_shift_to_last_season` treatment as the existing `build_pitcher_stats`.
Layer: processing
Tests: `tests/hit_predictor/test_season_stats.py`

TASK 2.2.1: Extract entity_col into `build_pitcher_stats`, add `build_pitcher_stats_all_roles`
  RED:
    - Test: `test_build_pitcher_stats_all_roles_pools_bullpen_by_team`
    - Fixture: a small `pitcher_boxscore` DataFrame (personId, gamepk, team_id, game_season, h,
      r, er, bb, hr, k, p, s, ip) with two pitchers on the same team_id in the same game, both
      NOT the starter that game, plus a matching minimal `pbp` fixture (via
      `_pitcher_role_lookup`) tagging both as `pitcher_role='bullpen'`, `pitcher_team_id`
      matching the boxscore's team_id — check dtype alignment carefully (pitcher_boxscore's
      `personId`/`team_id` vs pbp's `pitcher_id`/`pitcher_team_id`: cast both to str explicitly
      in the fixture, matching the existing dtype-alignment pattern in
      `test_add_pbp_handedness_joins_despite_int_vs_str_id_dtype_mismatch` in
      test_pipeline.py).
    - Assert: exactly one row with `pitcher_role='bullpen'` for that team/season (pooled, not
      two rows for the two individual pitchers); that row's summed `h`/`ip`/etc. reflect BOTH
      pitchers combined, not just one.
    - Run pytest, confirm FAILS (function doesn't exist).
  GREEN:
    - Give `build_pitcher_stats` an `entity_col: str = 'personId'` parameter; change its
      `.groupby(['personId', 'game_season'])` → `.groupby([entity_col, 'game_season'])` and its
      `_prefix_stat_cols(df, 'pitcher_season_', key_cols=['personId', 'game_season'])` →
      `key_cols=[entity_col, 'game_season']`.
    - `build_pitcher_stats_all_roles(pitcher_boxscore, pbp)`:
      - `role_lookup = _pitcher_role_lookup(pbp)` (cast `pitcher_boxscore['personId']` to str
        and rename to `pitcher_id` before merging with `role_lookup` on `['gamepk', 'pitcher_id']`,
        `how='left'` — rows with no pbp role match, e.g. a pitcher who threw 0 pitches somehow,
        stay unclassified; decide during implementation whether to drop or keep them, note the
        choice in a comment).
      - `sp_box = tagged[tagged['pitcher_role']=='sp']`; `bullpen_box = tagged[tagged['pitcher_role']=='bullpen']`
      - `sp = build_pitcher_stats(sp_box, entity_col='pitcher_id').rename(columns={'pitcher_id':'pitcher_key_id'}).assign(pitcher_role='sp')`
      - `bullpen = build_pitcher_stats(bullpen_box, entity_col='pitcher_team_id').rename(columns={'pitcher_team_id':'pitcher_key_id'}).assign(pitcher_role='bullpen')`
      - `return pd.concat([sp, bullpen], ignore_index=True)`
  REFACTOR: confirm the existing `build_pitcher_stats` call site/tests (used directly elsewhere,
    e.g. any current caller outside this new wrapper) still pass unchanged with the new
    `entity_col` param defaulted.

---

## EPIC 3: `rolling_stats.py` — entity_col for pbp-derived rolling pitcher stats
Why: rolling equivalent of Epic 1.
Depends on: Epic 1 (imports `_prefix_stat_cols`-style helpers from season_stats.py; same
constants/pattern)

### STORY 3.1: `entity_col` threaded through the per-game layer + `build_pbp_pitcher_rolling_feats`
Acceptance: `build_pbp_pitcher_rolling_feats(pbp, window=..., pitcher_role='bullpen', entity_col='pitcher_team_id')`
rolls pooled team-bullpen per-game totals forward, same formulas as the pitcher_id-keyed
version. The rolling-math layer (`_rolling_sum`/`_rolling_max`/`_rolling_pooled_std`) is already
entity-agnostic — only the per-game aggregation functions need changes.
Layer: processing
Files: `src/models/hit_predictor/processing/features/rolling_stats.py`
Tests: `tests/hit_predictor/test_rolling_stats.py`

TASK 3.1.1: Thread `entity_col` through `_pitcher_stuff_command_per_game`,
`_pitcher_pa_outcome_per_game`, `_pitcher_last_inning_per_game`, `_pitcher_contact_quality_per_game`,
`_pitcher_pbp_per_game`, `build_pbp_pitcher_rolling_feats`
  RED:
    - Test: `test_build_pbp_pitcher_rolling_feats_pools_bullpen_by_team_when_entity_col_given`
    - Fixture: extend the existing `_pitch_row()`-based bullpen fixture with a SECOND bullpen
      pitcher on the same `pitcher_team_id`, overlapping games, so pooling has something to
      prove (mirroring 1.1.1's approach).
    - `result = build_pbp_pitcher_rolling_feats(df, window=10, pitcher_role='bullpen', entity_col='pitcher_team_id')`
    - Assert: rows are keyed by `pitcher_team_id` (one row per team+game, not per individual
      pitcher+game); a later game's rolled stat reflects BOTH pitchers' prior games pooled
      together, not just one pitcher's history — pick a concrete rolled value the same way the
      existing `test_build_pbp_pitcher_rolling_feats_pitcher_role_filter` test does (exact
      `pytest.approx` on a specific rolled mean).
    - Run pytest, confirm FAILS (`entity_col` not an accepted kwarg — TypeError — or wrong
      grouping if partially wired).
  GREEN:
    - Replace `group_cols = PBP_PITCHER_KEY_COLS` in each of the 4 per-game helpers with
      `group_cols = _pbp_pitcher_key_cols(entity_col)` — new small helper:
      `def _pbp_pitcher_key_cols(entity_col): return [entity_col, 'gamepk', 'game_date', 'game_season']`,
      replacing the module constant `PBP_PITCHER_KEY_COLS` usage (keep the constant itself for
      any other reference, or replace it entirely with `_pbp_pitcher_key_cols('pitcher_id')` if
      nothing else depends on the literal name — check usages before deciding).
    - `_pitcher_last_inning_per_game`: change `.groupby(['pitcher_id', 'gamepk'])['play_id'].idxmax()`
      → `.groupby([entity_col, 'gamepk'])['play_id'].idxmax()` (see design note above — this is
      the one line that gives the team-pooled version its correct "last pitch thrown by any
      reliever that game" meaning for free).
    - `_pitcher_pbp_per_game(pbp, pitcher_role=None, entity_col='pitcher_id')`: pass `entity_col`
      to all 4 per-game helper calls and to the `n_pitches` groupby/merge.
    - `build_pbp_pitcher_rolling_feats(pbp, window, pitcher_role=None, entity_col='pitcher_id')`:
      replace the hardcoded local `entity_col = 'pitcher_id'` with the new parameter; pass
      `entity_col` into `_pitcher_pbp_per_game`; replace `key_cols = PBP_PITCHER_KEY_COLS` with
      `key_cols = _pbp_pitcher_key_cols(entity_col)`.
  REFACTOR: none expected — mechanical threading, same shape as Epic 1.

### STORY 3.2: `build_pbp_pitcher_rolling_feats_all_roles` pools bullpen by team
Acceptance: same `pitcher_key_id` unification as Epic 1's Story 1.2, rolling version.
Layer: processing
Tests: `tests/hit_predictor/test_rolling_stats.py`

TASK 3.2.1: Update `build_pbp_pitcher_rolling_feats_all_roles`
  RED:
    - Test: `test_build_pbp_pitcher_rolling_feats_all_roles_pools_bullpen_by_team`
    - Same 2-bullpen-pitcher-same-team fixture as 3.1.1.
    - Assert: result has `pitcher_key_id` (not separate `pitcher_id`/`pitcher_team_id` cols);
      bullpen rows keyed by team_id, pooled across both pitchers.
    - Run pytest, confirm FAILS.
  GREEN: same rename+concat pattern as Task 1.2.1.
  REFACTOR: confirm the prior fix's existing test
    (`test_build_pbp_pitcher_rolling_feats_all_roles_tags_and_stacks_both_roles`) still passes.

---

## EPIC 4: `rolling_stats.py` — team-pooled boxscore-derived rolling pitcher stats
Why: rolling equivalent of Epic 2 (`pitcher_rolling_season_stats`/`pitcher_rolling_short_stats`
are the rolling half of the same never-role-aware bug).
Depends on: Epic 2 (`_pitcher_role_lookup`)

### STORY 4.1: `build_pitcher_rolling_stats_all_roles`
Acceptance: same split-tag-concat pattern as Epic 2, rolling version — `_rolling_sum` is already
entity-agnostic, so this is mostly plumbing `team_id` through the column selection and role join.
Layer: processing
Tests: `tests/hit_predictor/test_rolling_stats.py`

TASK 4.1.1: Extract entity_col into `build_pitcher_rolling_stats`, add `build_pitcher_rolling_stats_all_roles`
  RED:
    - Test: `test_build_pitcher_rolling_stats_all_roles_pools_bullpen_by_team`
    - Same shape as Task 2.2.1's fixture, extended to 2+ games so the rolling window has
      something to roll forward (mirroring existing rolling-stat test patterns in this file —
      first game's rolled value is NaN, second game picks up the first).
    - Assert: bullpen rows keyed by `pitcher_team_id` (renamed `pitcher_key_id`), pooled across
      the team's relievers, rolled value on game 2 reflects BOTH relievers' game-1 totals
      combined.
    - Run pytest, confirm FAILS.
  GREEN:
    - Give `build_pitcher_rolling_stats` an `entity_col: str = 'personId'` param; bring `team_id`
      into its `BOX_KEY_COLS + stat_cols` column selection (currently only selects `BOX_KEY_COLS
      + stat_cols` where `BOX_KEY_COLS` hardcodes `'personId'` first — needs the same
      `_box_key_cols(entity_col)` treatment as Story 3.1's `_pbp_pitcher_key_cols`); pass
      `entity_col` into the `_rolling_sum(df, entity_col=entity_col, ...)` call (already accepts
      it — just currently hardcoded to `'personId'` at the call site); `_prefix_stat_cols(...,
      key_cols=_box_key_cols(entity_col))`.
    - `build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window)`: same role-tag-join
      (via `_pitcher_role_lookup`), split, rename-to-`pitcher_key_id`, concat pattern as Task
      2.2.1.
  REFACTOR: none expected.

---

## EPIC 5: `pipeline.py` — `pa_outcome` exposes `pitcher_team_id`
Why: `train.py` needs `pitcher_team_id` on every PA row to compute `pitcher_key_id` (sp → use
`pitcher_id`, bullpen → use `pitcher_team_id`).
Depends on: none

### STORY 5.1: `create_pa_outcome` selects `pitcher_team_id`
Layer: processing
File: `src/models/hit_predictor/processing/pipeline.py`
Tests: `tests/hit_predictor/test_pipeline.py`

TASK 5.1.1: Add `pitcher_team_id` to `create_pa_outcome`'s column selection
  RED:
    - Test: `test_create_pa_outcome_includes_pitcher_team_id`
    - Same fixture pattern as `test_create_pa_outcome_includes_pitcher_role`, with
      `'pitcher_team_id': 'T1'` added to the `pbp` fixture row.
    - Assert: `result.loc[0, 'pitcher_team_id'] == 'T1'`.
    - Run pytest, confirm FAILS with `KeyError`.
  GREEN: add `'pitcher_team_id'` to the `pbp[[...]]` column list in `create_pa_outcome`.
  REFACTOR: none — one-line change, third time this exact pattern has been applied this session.

---

## EPIC 6: Wire team-pooled bullpen features into `v2_rolling_features/train.py`
Why: the actual fix, end to end.
Depends on: Epics 1-5

### STORY 6.1: `pitcher_key_id` + team-pooled merges + hand nulling
Acceptance: script runs end-to-end against real S3 data; `pitcher_key_id` resolves to
`pitcher_id` for sp rows and `pitcher_team_id` for bullpen rows; all pitcher-conditioned feature
merges use the new `_all_roles` builders joined on `pitcher_key_id`; `pitcher_throw_hand` is null
for bullpen rows; a known swingman's bullpen-role stats are now team-pooled values, genuinely
different from his own individual SP-role stats.
Layer: experiments
File: `src/models/hit_predictor/experiments/v2_rolling_features/train.py`

TASK 6.1.1: Rewire the pitcher-feature assembly block
  No RED/GREEN — same trust boundary as the rest of `train.py`. Every piece it calls is covered
  by its own RED/GREEN task above. Verified by manual smoke run.

  Implementation notes:
  - After `pa_outcome` is available: `model_df['pitcher_key_id'] = model_df['pitcher_id'].where(model_df['pitcher_role'] == 'sp', model_df['pitcher_team_id'])`
    (pandas `.where()` avoids adding a numpy import purely for this one line — `Series.where(cond, other)`
    keeps values where `cond` is True, replaces with `other` where False, which is exactly "sp → pitcher_id, else → pitcher_team_id").
  - `pitcher_season_stats = season_stats.build_pitcher_stats(...)` → `season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)`,
    merged on `['game_season', 'pitcher_key_id', 'pitcher_role']`.
  - `pitcher_rolling_season_stats`/`pitcher_rolling_short_stats` → `rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window=...)`,
    pulled out of the generic batter/pitcher rolling loop (needs the extra `pitcher_role` key
    column the batter tables don't have), merged on `['gamepk', 'pitcher_key_id', 'pitcher_role']`.
  - `pitcher_role_season_stats`/`pitcher_role_rolling_*_stats` (from the prior fix): their merges
    change from `pitcher_id` to `pitcher_key_id` (since `build_pbp_pitcher_feats_all_roles`/
    `build_pbp_pitcher_rolling_feats_all_roles` now output `pitcher_key_id` per Epics 1/3).
  - Null `pitcher_throw_hand` where `pitcher_role == 'bullpen'`:
    `model_df.loc[model_df['pitcher_role'] == 'bullpen', 'pitcher_throw_hand'] = None` — after
    `model_df` is fully assembled, before the missing-indicator/impute step.
  - `CAT_FEATS` unchanged (`pitcher_role` already added by the prior fix).
  - Smoke-test via the established scratch-copy pattern (3-season window, 10-tree RF, BASE_DIR
    hardcoded). Sanity checks: `pitcher_key_id` populated for every row; a known swingman's
    bullpen-role `pitcher_last_season_stuff_start_speed_mean` (or similar) now differs from his
    SP-role value AND from what it was before this fix (team-pooled, not his own individual
    numbers); row count still preserved through the new merges.

---

## Before marking any task complete:
- [x] Watched the test FAIL before writing any code
- [x] Test failed for the right reason (feature missing, not a typo)
- [x] Wrote minimal code — nothing extra
- [x] All tests pass after GREEN
- [x] No new warnings or errors in pytest output
- [x] Tests use real code — mocks only where I/O is unavoidable (none needed in Epics 1-5; Epic 6
      verified by manual smoke run, not pytest, per its own note above)
- [x] Full suite still 246+ passing after each epic, not just the new tests (255 passing at the end)
- [x] Smoke-run `v2_rolling_features/train.py`: `pitcher_key_id` populated for every row, and a
      known swingman's bullpen-role stats are genuinely team-pooled (different from his own
      individual numbers), not just role-separated from his SP stats

## Deviation found during implementation
The scoped smoke test caught a serious bug the unit tests missed: `build_pitcher_rolling_stats_all_roles`'s
bullpen path produced a 6x row fan-out in `model_df` (504,604 -> 3,165,976 rows). Root cause:
`pitcher_boxscore` has one row per INDIVIDUAL pitcher per game, but `_rolling_sum` (used inside
`build_pitcher_rolling_stats`) operates via `.transform()`, which preserves one output row per
INPUT row rather than collapsing to one per (team, game) — so any game with 2+ relievers on the
same team produced duplicate rows at the same `(team_id, gamepk)` join key. The pbp-derived
rolling stats (Epics 1/3) never had this bug because `_pitcher_pbp_per_game` already collapses to
one row per (entity_col, gamepk) via `.groupby().agg()` before any rolling is applied — the
boxscore path was missing that same collapse step. My Epic 4 unit test didn't catch it because
its fixture never had two relievers in the same game (each bullpen appearance was on a different
gamepk). Fixed by summing `bullpen_box`'s stat columns to one row per (team_id, gamepk) before
calling `build_pitcher_rolling_stats`, plus a new regression test
(`test_build_pitcher_rolling_stats_all_roles_collapses_multiple_relievers_per_game_before_rolling`)
built specifically around two relievers sharing a game. This is the second time in this session a
scoped smoke run caught a real bug unit tests missed purely because the unit fixtures were too
narrow (see also the earlier `SimpleImputer`/`keep_empty_features` finding) — worth remembering
when writing fixtures for aggregation code: always include the "multiple real-world entities
collapsing to one group" case, not just the "one entity across time" case.

## Smoke-run results (scoped 3-season window, 2023-2025)
- `len(pa_outcome) == len(model_df)` — 504,604 == 504,604 after the fan-out fix.
- `pitcher_key_id` null rate: 0.0000 — resolves correctly for every row.
- `pitcher_throw_hand` null rate by role: `bullpen` 100% null (correctly nulled — a specific
  reliever's hand isn't knowable pre-game), `sp` ~1% null (pre-existing missingness, unrelated to
  this fix).
- Swingman `pitcher_id=425844`: sp `pitcher_key_id='425844'` (his own id) vs. bullpen
  `pitcher_key_id='118'` (a team id, not his own) — confirms role-conditional key resolution and
  team pooling both work end to end.
