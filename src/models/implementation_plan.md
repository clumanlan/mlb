# Implementation Plan: `build_team_bullpen_pa_share` — manager quick-hook tendency feature

**Audit gate:** `pytest -q` -> 602 passed, 0 failed (2026-09-01). `pylint src/models/hit_predictor/processing/features/game_context.py --fail-under=7.0` -> 7.82/10 (pre-existing line-length/docstring/import nits, none touching code this plan changes). Baseline is clean; planning proceeds.

**Why this is next:** came out of a design discussion on reframing k_predictor (pitcher strikeout prediction) away from both its current grains — v1-v11's per-PA grain (real PAs within a start are correlated, and treats a batter's 1st PA the same as his 3rd/4th against the same starter) and v12's start-grain NB2 (collapses away all batter identity, and today tests significantly worse-calibrated against real 2026 DK odds than v6 — 36x vs. the market vs. v6's 13x; see `ROADMAP.md` item 6(d) / `experiments/v12_start_grain_negbin/v12_results.md`). The agreed next design is a third grain — one row per batter-in-lineup × their expected PA count for that start, using `game_context.build_batter_slot_expansion`'s existing cycling logic collapsed to per-batter, with `expected_pa` as a count-regression exposure offset. That full redesign (a k_predictor v13 experiment) is follow-on work, NOT in scope here.

**Scope of this plan:** one new shared feature function only — `build_team_bullpen_pa_share` in `game_context.py`. During the design conversation, the user specifically asked for "expected starter times through order / plate appearances" plus a "bullpen plate appearance count" as a proxy for manager quick-hook tendency. The quick-hook proxy is genuinely reusable (agreed: "could be helpful for other models as well" — directly relevant to `short_outing_predictor` and `batters_faced_predictor` too, arguably more central to those), so it's being built as its own independent, TDD'd unit first, ahead of and separate from the v13 batter-grain experiment that will eventually consume it.

**Scope decisions (confirmed with user):**
- **Output is a rolling historical average, not the current game's own realized value.** Point-in-time safety: this is a proxy for a TEAM's standing tendency (how early that team's manager typically pulls starters), computed from games strictly before the one being predicted — same `.shift(1)`-via-`_rolling_sum` convention every other rolling feature in this file already follows. Using today's own game's realized bullpen share would both leak the outcome and defeat the point (today's game's own quick-hook decision is exactly what a pre-game feature can't know yet).
- **Stat name: `bullpen_pa_share`**, not `starter_pa_share`. Higher value = more of the game handed to the bullpen = quicker hook — the polarity matches "quick-hook tendency" directly, no `1 -` flip needed downstream. Confirmed with user.
- **Roll counts, then divide once — never average per-game ratios.** Same correctness rule this whole file already documents (`rolling_stats.py`'s header comment) and the same shape `build_team_win_loss_record` uses for `win_pct`: roll `bullpen_pa` and `team_total_pa` as separate counts via `_rolling_sum`, then `_finalize_rates` divides once at the end. Averaging per-game shares directly would incorrectly weight a start with 5 PAs the same as one with 30.
- **PA counted via last-pitch-per-play_id, not raw pbp row count.** pbp is pitch-level (multiple rows per PA); `build_pitcher_start_pa_this_season` already establishes the correct pattern (`pitch_number == max` per `(gamepk, play_id)`) for isolating one row per real PA before counting. Reused verbatim, not reinvented.
- **Keyed and merged like every other team-level feature here**: output uses `team_id` (renamed from pbp's `pitcher_team_id` at build time) plus `TEAM_GAME_KEY_COLS`-shaped keys, so downstream consumers merge on `(team_id, gamepk)` the same way they already merge `build_team_win_loss_record`'s output — callers rename to `pitcher_team_id`/whatever they need at merge time, same as `v12_start_grain_negbin/train.py` already does for `pitching_team_rolling`.
- **No new test file** — appended to the existing `tests/hit_predictor/test_game_context.py`, reusing the existing `_start_pa_this_season_pbp_row` fixture helper (already supports `pitcher_role='sp'|'bullpen'`), matching how every other `game_context.py` function's tests live in that one file.

---

## EPIC 1: `build_team_bullpen_pa_share` — new shared team-level context feature
Why: no existing feature anywhere in this codebase captures manager quick-hook tendency — confirmed absent from `game_context.py`, `short_outing_predictor` (which has no features file of its own), and `batters_faced_predictor`. It's a real gap the design conversation surfaced, and it's reusable infrastructure, not k_predictor-specific.
Depends on: none

### STORY 1.1 — rolling, point-in-time-safe bullpen PA share per (team_id, gamepk)
Layer: processing / shared feature (`src/models/hit_predictor/processing/features/game_context.py`)
Acceptance: one row per `(team_id, gamepk)`; `window='season'` gives an expanding season-to-date share (resets each year), `window=<int>` gives a trailing-N-game share (carries across season boundaries) — same two-mode contract as `build_team_win_loss_record`. Output column named `team_roll_season_bullpen_pa_share` / `team_roll_last{N}g_bullpen_pa_share` via the existing `_rolling_prefix`/`_prefix_stat_cols` helpers. A team's first game in a window has NaN (no prior data), not 0 or an error. A team-game with zero bullpen PAs (a complete-game start) contributes a real 0, not a NaN, to the rolled numerator.

#### TASK 1.1.1 — season-window rolling share, basic two-role case
RED:
- File: `tests/hit_predictor/test_game_context.py`
- Test name: `test_build_team_bullpen_pa_share_season_window_rolls_forward_share`
- Fixture: reuse `_start_pa_this_season_pbp_row`. Team T1, g1 (2024-04-01): 3 `pitcher_role='sp'` PAs (play_id 1-3) + 2 `pitcher_role='bullpen'` PAs (play_id 4-5) -> 5 total PAs, share = 2/5 = 0.4. Team T1, g2 (2024-04-06): any PAs (content doesn't matter for this assertion).
- Assert: `import build_team_bullpen_pa_share` from `game_context`; call with `window='season'`; on g2's T1 row, `team_roll_season_bullpen_pa_share == pytest.approx(0.4)` — g1's own share rolled forward, not g2's own (which isn't computed into its own row).
- Run `pytest tests/hit_predictor/test_game_context.py -k bullpen_pa_share -v` and confirm it FAILS.
- Expected failure message: `ImportError: cannot import name 'build_team_bullpen_pa_share'` (function doesn't exist yet).

GREEN:
- File: `src/models/hit_predictor/processing/features/game_context.py`
- Implement `build_team_bullpen_pa_share(pbp: pd.DataFrame, window: str | int) -> pd.DataFrame`:
  1. Isolate one row per real PA: `last_pitch = pbp[pbp['pitch_number'] == pbp.groupby(['gamepk','play_id'])['pitch_number'].transform('max')]` (same pattern as `build_pitcher_start_pa_this_season`).
  2. `groupby(['pitcher_team_id','gamepk','game_date','game_datetime','game_season','pitcher_role']).agg(pa_n=('play_result','count')).reset_index()`.
  3. `pivot_table(index=[...team-game keys...], columns='pitcher_role', values='pa_n', fill_value=0)` to get `sp`/`bullpen` columns side by side; `.reset_index()`.
  4. `bullpen_pa = wide.get('bullpen', 0)`; `team_total_pa = wide.get('sp', 0) + wide.get('bullpen', 0)`; rename `pitcher_team_id` -> `team_id`.
  5. `_rolling_sum(wide, entity_col='team_id', cols=['bullpen_pa','team_total_pa'], window=window, sort_col='game_datetime')`.
  6. `_finalize_rates(rolled, {'bullpen_pa_share': ('bullpen_pa','team_total_pa')})`.
  7. Slice to key cols + `bullpen_pa_share`, prefix via `_prefix_stat_cols(..., prefix=_rolling_prefix('team', window), key_cols=key_cols)`.
- Run: `pytest tests/hit_predictor/test_game_context.py -k bullpen_pa_share -v` and confirm it PASSES.

REFACTOR:
- Compare against `build_team_win_loss_record`'s structure line-by-line for consistency (key col ordering, docstring shape). No behavior change expected.
- Run: same pytest command, must stay green.

#### TASK 1.1.2 — first game of the season is NaN, not 0
RED:
- Test name: `test_build_team_bullpen_pa_share_first_game_of_season_is_nan`
- Fixture: single game for T1 (any sp/bullpen mix).
- Assert: `pd.isna(result.iloc[0]['team_roll_season_bullpen_pa_share'])`.
- Run and confirm it FAILS if Task 1.1.1's implementation doesn't naturally produce NaN here (sanity check on `_rolling_sum`'s own `.shift(1)`/first-row NaN behavior propagating through `_finalize_rates`'s division) — if it already passes because Task 1.1.1's GREEN step already covered this mechanically, note that explicitly rather than forcing an artificial failure.

GREEN:
- No new code anticipated — `_rolling_sum`'s existing shift(1) behavior should already produce NaN counts for a team's first game, which `_finalize_rates` correctly divides through to NaN (not a `ZeroDivisionError`, since `.replace(0, np.nan)` guards the denominator — confirm NaN/NaN also correctly yields NaN, not an error).
- Run and confirm PASS.

#### TASK 1.1.3 — trailing-N-game window
RED:
- Test name: `test_build_team_bullpen_pa_share_short_window_is_trailing_n_games`
- Fixture: 3 games for T1 with distinct sp/bullpen splits, `window=2`.
- Assert: g3's `team_roll_last2g_bullpen_pa_share` reflects only g1+g2's pooled counts (not g3's own, not a 3-game expanding value); column name uses the `last2g` prefix.
- Run and confirm it FAILS only if the int-window branch of `_rolling_sum`/`_rolling_prefix` isn't already wired correctly — expect this to mostly prove existing composition works, per Task 1.1.1's design already passing `window` straight through.

GREEN:
- Fix wiring only if the RED step actually failed; otherwise this task confirms existing behavior via a new passing test.
- Run and confirm PASS.

#### TASK 1.1.4 — complete-game start (no bullpen PAs at all) contributes a real 0, not NaN
RED:
- Test name: `test_build_team_bullpen_pa_share_complete_game_start_gives_zero_not_nan`
- Fixture: T1 g1 has ONLY `pitcher_role='sp'` PAs (no bullpen rows at all that game — a real complete game), rolled into g2.
- Assert: g2's rolled `team_roll_season_bullpen_pa_share == pytest.approx(0.0)` — a genuine "this team's bullpen threw 0% of PAs in the window" data point, not NaN ("no data yet").
- Run and confirm it FAILS: `pivot_table` on data with zero `'bullpen'` rows anywhere in the whole input frame won't produce a `'bullpen'` column at all, so `wide.get('bullpen', 0)` returns the scalar `0` for the `.get()` fallback but this needs verifying against the pivot's actual shape — expected failure is a `KeyError` or the share coming back NaN instead of 0.0 if the fallback isn't handled correctly.

GREEN:
- Ensure `wide.get('bullpen', 0)` (or equivalent explicit column-existence check) correctly back-fills an all-zero `bullpen_pa` column when the pivot produces no `'bullpen'` column at all, so the rolled sum is a real 0 count rather than missing/NaN.
- Run and confirm PASS.

---

---

**EPIC 1 status: DONE, 2026-09-01.** `build_team_bullpen_pa_share` implemented and tested (4/4 tests green on first GREEN pass, including the complete-game-start zero-vs-NaN edge case). Full suite 602 -> 606 passed. Not yet wired into any consumer.

## EPIC 2: `build_batter_expected_pa` — batter-slot collapse for v13's batter grain
Why: v13's whole design rests on one row per real lineup batter, sized by his expected PA count, rather than v6's ~9-45 synthetic per-slot rows or v12's single start-level row. `build_batter_slot_expansion` already produces the per-slot data; this collapses it to the batter grain v13 actually fits against.
Depends on: none (composes with `build_batter_slot_expansion`'s existing output, doesn't modify it)

### STORY 2.1 — `build_batter_expected_pa(slot_expansion)` returns one row per real batter with expected_pa and capped TTO-max
Layer: processing / shared feature (`game_context.py`, same file as EPIC 1 and `build_batter_slot_expansion`)
Acceptance: one row per (gamepk, expected_pitcher_key_id, batter_id); `expected_pa` = count of synthetic slots assigned to that batter (a raw, never-capped count); `expected_times_through_order_max` = his highest reached cycle, which IS capped at 3 (inherited from `build_batter_slot_expansion`'s own cap). The raw-count-vs-capped-cycle distinction is real and was specifically tested (a batter with 4 real slots across a long outing gets `expected_pa=4` but `expected_times_through_order_max=3`).

**STATUS: DONE, 2026-09-01.** 3/3 new tests green on first GREEN pass (`test_build_batter_expected_pa_counts_slots_per_real_batter`, `test_build_batter_expected_pa_one_row_per_real_lineup_batter`, `test_build_batter_expected_pa_counts_raw_pa_even_when_times_through_order_is_capped`). Full suite 606 -> 609 passed. `pylint` 7.73/10, gate 7.0. Not yet wired into any consumer.

---

## EPIC 3: k_predictor v13 experiment — batter-grain Poisson GLM (NOT YET STARTED)
Why: consumes EPIC 1 + EPIC 2 to actually test the batter-grain hypothesis against real 2026 DK odds, same evaluation framework v6/v12 already used.
Depends on: EPIC 1, EPIC 2 (both done)

Scope (per design discussion, not yet broken into tasks):
- Target: `pa_outcome[pitcher_role=='sp'].groupby(['gamepk','starting_pitcher_id','batter_id'])['is_strikeout'].sum()` — realized strikeouts a real batter recorded against the starter specifically (existing columns, no new schema).
- Features per batter-row: existing batter identity (`batter_last_season_pa_strikeout_rate`, `batter_roll_season_pa_strikeout_rate`), the starter's own pitching features (v12's already-assembled set), `team_roll_season_bullpen_pa_share` (EPIC 1, merged on the pitcher's team).
- Model: Poisson GLM (`statsmodels`), `exposure=expected_pa` (EPIC 2's output) rather than a plain feature — deliberately not XGBoost yet, to isolate the grain/exposure hypothesis from a model-family confound, matching v12's own reasoning; see the published design artifact for full reasoning (Poisson-additivity, GLM-vs-XGBoost).
- Total start-K distribution: sum of the 9 batters' independent Poisson(mean) predictions is exactly Poisson(sum of means) — no new pmf-combination utility needed, unlike v12's `negative_binomial_pmf`.
- Evaluation: same real-2026-odds backtest framework (`edge_report_2026.py`/`significance_check_2026.py`) v6 and v12 were both scored against, for a clean 3-way comparison.
- Per this repo's established convention (v12's plan, scope decision #5), the experiment script itself (`experiments/v13_.../train.py`) does not get its own tests — only genuinely new reusable methodology does. Target construction is data wiring in the script.
- Not yet broken into RED/GREEN tasks — no new reusable methodology has been scoped beyond EPICs 1-2 yet. Revisit this section into real tasks before writing `train.py`.

**STATUS: `train.py` written and running, 2026-09-01.** One more genuinely new reusable unit surfaced while wiring the script: `poisson_pmf(mean, max_k)` in `count_distribution.py`, TDD'd (2 new tests: matches `scipy.stats.poisson` at a known mean; is a valid distribution across several means — full suite 609 -> 615). This is intentionally thin compared to v12's `negative_binomial_pmf` — the whole point of the design (see the published artifact) is that summing several batters' independent Poisson means needs no new combination algorithm, only a pmf builder for the already-summed total.

A real bug was caught and fixed before running: `build_batter_expected_pa` (EPIC 2) originally grouped on `expected_pitcher_key_id`, matching its own test fixture's column name — but every real start-grain frame in this codebase (v6, v12, `score_2026_test_dates.py`) carries the pitcher key as `personId`. The tests passed only because the fixture happened to match the hardcoded string, not because the design was correct. Fixed to group on `personId`, with a new test (`test_build_batter_expected_pa_keys_on_personid_not_shared_lineup_position`) covering the specific failure mode this would have caused: two different starts sharing a `gamepk` (e.g. both starters of one game) getting silently collapsed together.

`experiments/v13_batter_grain_poisson/train.py` written, mirroring v12's structure (S3 load, shared feature tables, pitcher-side start frame) with a new Section 4 (batter-grain frame via `build_batter_slot_expansion` -> `build_batter_expected_pa`, batter-identity features, EPIC 1's `bullpen_pa_share`), Section 5 (target: realized batter-vs-starter K count via `create_pa_outcome_strikeout`, already scoped to `pitcher_role=='sp'`), Section 6 (Poisson GLM fit, `exposure=expected_pa`), and the same val-season coverage/threshold check + real-2026-odds backtest machinery v6/v12 were both scored against.

**A real, severe bug found and fixed, 2026-09-02, via a cheap single-season diagnostic (not the full 8-season run):** the first two full runs produced an implausible ~10% implied per-PA strikeout rate (this repo's documented baseline is ~22%) and a model that predicted roughly half the realized mean K per start. Root cause: `build_batter_slot_expansion` merges `batting_order` onto `(gamepk, lineup_position)` with no team-awareness — genuinely ambiguous whenever a `gamepk` has two starts (home SP, away SP), each needing a *different* team's 9 batters. Passing it the full unscoped `batting_order` — which every existing caller of this function in the codebase does, including v6's own `score_2026_test_dates.py` real-odds backtest script — lets roughly half of all synthetic slots collide against the WRONG team (a pitcher's own teammates, whom he never actually faces). A single-season diagnostic confirmed this precisely: match rate against the real target was 49.9% (an exact home/away-collision signature), and after fixing it (team-scoping `batting_order` per start — call the expansion once for home-team starters against the away lineup, once for away-team starters against the home lineup, concatenate) match rate went to 99.7% and the summed target matched the true season total exactly (21,423 = 21,423), implied per-PA rate 0.2198 vs. true 0.2211.

**This is a latent bug in existing production backtest code, not just a new one introduced here** — all 6 existing callers of `build_batter_slot_expansion` (`score_2026_test_dates.py`, `score_2025_test_dates.py`, `run_xgboost_uncertainty.py`, `run_naive_batter_uncertainty.py`, `run_batters_faced_distribution.py`, `run.py`) pass it an unscoped `batting_order`. v6's own trained PA classifier is unaffected (it fits on real, correctly-matched `pa_outcome` rows, never on this synthetic construct) — but any script using this function to COMBINE/AGGREGATE predictions at scoring time, including the one that produced v6's published 13x real-odds reliability gap, may have corrupted batter-identity and opposing-team features on roughly half its synthetic slots. Not yet audited or fixed in those other scripts — flagged here, not in scope for this v13 plan.

**EPIC 3 STATUS: DONE, 2026-09-02.** Post-fix full run: val-season predicted mean K (4.784) closely tracks realized (4.836), batter-identity features used non-trivially (p<0.001) — a real, working model, unlike v12. But on the real 2026 odds backtest, **hypothesis rejected**: reliability gap 70x the market's (worse than v12's 36x and v6's 13x), disagreement win rate 46.7% significantly BELOW the ~52.4% break-even bar (p=0.007) — no edge, worse than the market. Full detail in `experiments/v13_batter_grain_poisson/v13_results.md` and `ROADMAP.md` item 6(e). v6's tuned XGBoost remains the standing production candidate; do not pursue a fourth grain variation on this hypothesis — see v13_results.md's "Next steps" for why.

## Before marking any task complete:
- [ ] Watched the test FAIL before writing any code
- [ ] Test failed for the right reason (feature missing, not a typo)
- [ ] Wrote minimal code — nothing extra
- [ ] All tests pass after GREEN
- [ ] No new warnings or errors in pytest output
- [ ] Tests use real code — mocks only where I/O is unavoidable
- [ ] `pylint src/models/hit_predictor/processing/features/game_context.py --fail-under=7.0` still passes after the new function is added
