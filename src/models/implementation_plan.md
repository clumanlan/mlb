# Implementation Plan: k_predictor v12 — start-grain Negative Binomial total-K model

**Audit gate:** `pytest -q` -> 591 passed, 0 failed (2026-09-01). `pylint` is not installed in this venv, so the repo's own lint gate (CLAUDE.md's `pylint src/ --fail-under=7.0`) could not be run here — flake8 is available (7.3.0) but is not this repo's configured linter, so it wasn't run broadly to avoid false-positive noise against an unfamiliar config. Tests are the load-bearing hard blocker per the planning gate, and they're clean. Planning proceeds; re-run the real lint command in whichever environment has `pylint` installed before merging v12's code.

**Why this is next:** `ROADMAP.md`'s Near-term backlog (k_predictor) item 6(c) (2026-09-01) found v6's aggregated `P(total K > line)` significantly miscalibrated on a real 1,160-start 2026 DK-odds backtest — reliability 0.0145 vs. the market's 0.0011 (13x, bootstrap CI excludes zero). Isotonic recalibration fixes reliability 24x on honest held-out data but produces **no betting edge** (post-calibration disagreement win rate ~49.6%, still short of the ~52.4% break-even bar); resolution (discrimination) vs. the market isn't significantly different either way at this sample size. Eight prior feature-engineering passes on the per-PA classifier (v3–v11) already came back flat on PR-AUC/resolution, and `ROADMAP.md` already concluded to stop hunting for more per-PA features — so v12 is deliberately a different lever: change what's being modeled, not add another feature to what's already there.

**Working hypothesis:** v6's Poisson-binomial combination step treats a pitcher's ~22 synthetic batter-slots as *independent* Bernoulli trials. They aren't — same game, same park, same weather, same bullpen/fatigue trajectory all correlate outcomes across slots within a start — and under-modeling that correlation is a textbook mechanism for the aggregate overconfidence actually measured (bottom probability bucket predicted 25.1%, actual outcome rate was 49.0%). v12 tests this hypothesis directly by modeling total strikeouts per start as its own count-distributed target, with no per-slot independence assumption anywhere in the chain.

**Scope decisions (confirmed with user):**
- **Model family: Negative Binomial GLM** (`statsmodels.discrete.discrete_model.NegativeBinomial`, NB2 parameterization), not Poisson regression or XGBoost `count:poisson`. Reasoning: strikeout counts per start are overdispersed (variance > mean is the standard finding for this kind of count data, and v6's own coverage check already found non-zero interval-coverage gaps consistent with underestimated variance) — a plain Poisson model's mean-equals-variance constraint would likely reproduce the same overconfidence problem this experiment exists to fix. NB2's single fitted dispersion parameter α gives a properly-widened predictive distribution for free, estimated jointly with the regression coefficients via one MLE fit — no separate slot-expansion, no separate combination step, no independence assumption to get wrong. Tradeoff being accepted: a GLM is linear in the log-mean (`log(μ) = Xβ`), so it can't capture the nonlinear feature interactions XGBoost's shallow trees do — if v12 comes back weak, that's the first thing to revisit (e.g. a tree-based distributional model), not immediately assumed to mean the hypothesis is wrong.
- **Grain: one row per pitcher-start**, not per PA. This is the core architectural change from v6 — total K is modeled directly as the target, not derived by aggregating ~22 independent per-PA predictions.
- **Fit discipline stays identical to v6's**, even though the mechanism differs: `CORE_FIT_SEASONS` = `[2018, 2019, 2022]` and `EARLY_STOP_SEASON` = `2023` (both derived from `config.yaml` exactly as `score_2026_test_dates.py` does — see its `assert SCORE_SEASON not in FIT_SEASONS` guard, reused verbatim). v6 uses `EARLY_STOP_SEASON` for XGBoost's early-stopping mechanism; an MLE-fit GLM has no equivalent iterative-overfitting risk, so v12 fits on `CORE_FIT_SEASONS ∪ {EARLY_STOP_SEASON}` combined ([2018, 2019, 2022, 2023]) — same total fit-data footprint as v6 (core + early-stop rows), same untouched 2024/2025/2026, just no separate held-out subset during fitting since none is needed. This is a real difference from the v3–v11 convention worth stating plainly rather than silently copying a boosting-specific pattern onto a model that doesn't use it.
- **Features: reuse the already-assembled start-grain frame**, not new feature engineering. `score_2026_test_dates.py`/`run_xgboost_uncertainty.py`'s own Section 4 (`pitcher_starts_test`/`pitcher_starts_2024`) already builds a one-row-per-start frame carrying `pitcher_last_season_pa_strikeout_rate`, `pitcher_last_season_whip`, the season/3-game rolling pitcher stats, `pitcher_shrunk_whip`, weather, `expected_batters_faced`/`expected_batters_faced_weight`, and CSW/put-away/park-factor columns where available — this is exactly v12's starting feature set, reused wholesale rather than rebuilt. One real gap: that frame currently has no opponent-*batting*-side strikeout-susceptibility feature at start grain (only the pitcher's own team's batting K rate gets merged there today, via `pitching_team_rolling` keyed on `pitcher_team_id` — the wrong side of the matchup for this purpose). `build_team_batter_strikeout_rolling_feats` (already built, already tested, already used elsewhere in the same script keyed on `batter_team_id`) just needs an additional merge onto `opp_team_id`/`gamepk` — this is wiring an existing, already-tested table onto a new grain, not new feature engineering, consistent with the "stop hunting for new features" decision already made.
- **Evaluation reuses existing infrastructure, does not rebuild it.** Compare v12 against (a) v6 raw and (b) v6+isotonic-calibrated using the SAME 2026 real-odds backtest data already on disk (`backtest/pred_df_test2026.parquet`'s matched real-odds rows via `edge_report_2026.py`/`significance_check_2026.py`'s reliability/resolution/disagreement-win-rate framework), and separately against the 2024 val-season coverage-check framework (`run_xgboost_uncertainty.py`'s fixed-line threshold check + 50/80/95% interval coverage), for calibration/coverage continuity with every prior k_predictor version's evaluation. No new eval machinery gets built for this.
- **New production code gets tests first, per this repo's TDD requirement; the experiment script itself does not** — this repo's own established convention is that no `experiments/v{N}_*/train.py` has its own tests, only the shared building blocks it calls do (see `count_distribution.py`'s existing `poisson_binomial_pmf`/`prob_exceeds_line`, both TDD'd, both called untested from experiment scripts). The one genuinely new piece of reusable methodology here is a Negative-Binomial-pmf builder — that gets the same TDD treatment `poisson_binomial_pmf` got. The opponent-K-rate merge is data wiring in the experiment script, not a new module — no new test, matching how `score_2026_test_dates.py`'s own merges aren't unit tested.

---

## EPIC 1: `negative_binomial_pmf` — new count-distribution utility
Why: `count_distribution.py` already has `poisson_binomial_pmf` (independent-Bernoulli-sum total-K distribution) and `prob_exceeds_line` (distribution-agnostic — works on any array indexed 0..N by count, confirmed by reading its implementation). It has no NB-parameterized pmf builder — v12's entire predictive-distribution output depends on one existing.
Depends on: none

### STORY 1.1 — `negative_binomial_pmf(mean, alpha, max_k)` returns an exact NB2 pmf array
Layer: processing / shared utility (`src/models/hit_predictor/utils/count_distribution.py` — same file `poisson_binomial_pmf` lives in; this is target-agnostic combinatorial math, same "shared infrastructure, not k_predictor-local" precedent every prior count-distribution addition here has followed)
Acceptance: given a predicted mean μ and a fitted dispersion α (NB2 parameterization: `variance = μ + α·μ²`), returns a numpy array of `P(K=k)` for `k = 0..max_k`, summing to ~1.0 (up to truncation at `max_k`), matching `scipy.stats.nbinom`'s own pmf at spot-checked values — usable as a drop-in replacement for `poisson_binomial_pmf`'s output anywhere downstream (`prob_exceeds_line`, coverage-check interval logic, edge-report matching) since the whole point is that nothing downstream needs to know or care which pmf builder produced the array.

#### TASK 1.1.1 — converts (mean, alpha) to the `scipy.stats.nbinom(n, p)` parameterization correctly
RED:
- File: `tests/hit_predictor/test_count_distribution.py`
- Test name: `TestNegativeBinomialPmf::test_negative_binomial_pmf_matches_scipy_nbinom_at_known_params`
- What to assert: for a hand-picked `(mean=5.0, alpha=0.3, max_k=20)`, compute the expected `n = 1/alpha`, `p = n / (n + mean)` by hand (or via `scipy.stats.nbinom` directly in the test as the oracle), and assert `negative_binomial_pmf(5.0, 0.3, 20)` matches `scipy.stats.nbinom.pmf(np.arange(21), n, p)` elementwise to `np.allclose` tolerance. This pins the parameterization convention (NB2, not the alternative `size`/`prob` conventions some libraries use) so a future reader can't silently flip mean/variance.
- Run: `pytest tests/hit_predictor/test_count_distribution.py -k negative_binomial -v` and confirm it FAILS
- Expected failure message: `AttributeError: module 'count_distribution' has no attribute 'negative_binomial_pmf'`

GREEN:
- File: `src/models/hit_predictor/utils/count_distribution.py`
- Add `negative_binomial_pmf(mean: float, alpha: float, max_k: int) -> np.ndarray`, converting `(mean, alpha)` to `scipy.stats.nbinom`'s `(n, p)` via the NB2 formulas above, returning `nbinom.pmf(np.arange(max_k + 1), n, p)`. Import `scipy.stats.nbinom` at module top (new dependency for this file — confirm `scipy` is already in `requirements.txt`/the venv before assuming it's free; it is used elsewhere in this repo's stats work per `significance_check.py`, so almost certainly already available).
- Run: confirm PASSES

REFACTOR:
- If the mean/alpha -> n/p conversion ends up needed in more than one place (e.g. also wanted directly by train.py for diagnostics), extract it as a small private helper (`_nb2_to_scipy_params`) rather than duplicate the two-line formula. Only if actually duplicated — not preemptively.

### STORY 1.2 — degenerate/edge-case behavior
Layer: same file
Acceptance: the function doesn't silently produce garbage (NaNs, negative probabilities, a non-normalized array) on inputs at the edges of what a real fitted model could hand it.

#### TASK 1.2.1 — near-zero alpha behaves like a Poisson limit, not a crash
RED:
- File: `tests/hit_predictor/test_count_distribution.py`
- Test name: `TestNegativeBinomialPmf::test_negative_binomial_pmf_small_alpha_approximates_poisson`
- What to assert: as `alpha -> 0`, NB2 converges to Poisson(μ) (a real, well-known limiting property, not an implementation detail) — assert `negative_binomial_pmf(4.0, 1e-6, 20)` is close (`atol=1e-3`) to `scipy.stats.poisson.pmf(np.arange(21), 4.0)`. This also implicitly guards against a division-by-zero on `n = 1/alpha` blowing up numerically for small-but-nonzero alpha.
- Run and confirm FAILS (function doesn't exist yet if this task is done before 1.1.1's GREEN, or passes trivially/wrongly if done after without this specific case covered — write it fresh regardless and confirm it fails for the right reason: either `AttributeError` if run standalone-first, or an assertion mismatch if the naive implementation from 1.1.1 already happens to handle this correctly, in which case this test still documents the guarantee even though it was "accidentally" green).

GREEN:
- Only add code if Task 1.1.1's implementation doesn't already satisfy this (likely it does, since the scipy formulas are exact) — if the test passes immediately after 1.1.1's GREEN with no changes, that's fine and expected here; record that in the task rather than force an unnecessary code change.
- Run: confirm PASSES

REFACTOR: none expected.

#### TASK 1.2.2 — pmf sums to ~1 and every entry is a valid probability, across a spread of realistic (mean, alpha) pairs
RED:
- File: `tests/hit_predictor/test_count_distribution.py`
- Test name: `TestNegativeBinomialPmf::test_negative_binomial_pmf_is_a_valid_distribution`
- What to assert: parametrize over a few `(mean, alpha)` pairs spanning what a real fitted start-level model could plausibly output (e.g. `mean` in `[2.0, 5.0, 9.0]`, `alpha` in `[0.05, 0.3, 1.0]`), assert every returned pmf has all entries `>= 0`, sums to `>= 0.99` (allowing for truncation mass beyond `max_k`), and has length `max_k + 1`.
- Run and confirm FAILS pre-implementation (or passes if run after 1.1.1 — same note as Task 1.2.1: still worth having as an explicit, permanent regression guard even if it's green on first run).

GREEN: same as 1.2.1 — likely no new code needed; confirm and document.

REFACTOR: none expected.

---

## EPIC 2: Start-grain feature frame — opponent batting K-rate merge
Why: the only real gap in the already-assembled `pitcher_starts_test`/`pitcher_starts_2024` frame (built by `score_2026_test_dates.py`/`run_xgboost_uncertainty.py`'s Section 4) for a start-grain model is an opponent-*batting*-side strikeout-susceptibility feature — today that frame only carries the pitcher's own team's batting K rate (irrelevant; merged via `pitching_team_rolling` keyed on `pitcher_team_id`), not the actual opposing lineup's.
Depends on: none (independent of EPIC 1)

### STORY 2.1 — merge `opp_team_roll_season_pa_strikeout_rate` onto the start-grain frame
Layer: experiment script wiring (`src/models/k_predictor/experiments/v12_start_grain_negbin/train.py`), NOT a new shared-module function — `build_team_batter_strikeout_rolling_feats` already exists and is already tested; this is purely an additional `.merge()` in the new experiment script, following the exact same pattern the PA-grain code already uses (`opp_team_rolling` merged onto `batter_team_id`/`gamepk` in Section 2) but keyed on `opp_team_id`/`gamepk` instead — `opp_team_id` itself is already computed in Section 4 (`np.where(pitcher_team_id == home_id, away_id, home_id)`), so this is a one-line merge addition, not new logic.
Acceptance: `pitcher_starts_test`/`pitcher_starts_2026` (v12's copy of this frame) gains a column, `opp_team_roll_season_pa_strikeout_rate`, non-null for the same ~93-98% of rows every other rolling-season feature in this pipeline already achieves (matching the established non-null-rate sanity-check convention every prior version's results doc uses — see v10/v11's "Sanity check on the new columns" sections).

No RED/GREEN/REFACTOR here — per this repo's own established convention (confirmed in this plan's Scope Decisions above), experiment-script data wiring that only re-merges an already-tested table onto a new key doesn't get a new unit test, matching how none of `score_2026_test_dates.py`'s dozens of existing merges have tests either. Verify by manual sanity check in `train.py` itself (print non-null rate + mean, same pattern v10/v11's results docs already use) — not a pytest task.

---

## EPIC 3: `experiments/v12_start_grain_negbin/train.py`
Why: this is where EPIC 1 and EPIC 2's pieces actually get assembled into a fitted model and compared against v6 — the experiment itself, following this repo's `experiments/v{N}_*/train.py` convention (own directory, own `mlruns/`, results doc at the end). Structurally different from every prior version's `train.py` since this is start-grain, not PA-grain — no PA-vs-game-grain aggregation check applies here (there's no PA grain to aggregate from), and the model-fit step is `statsmodels` MLE, not `xgboost.XGBClassifier`.
Depends on: EPIC 1, EPIC 2

### STORY 3.1 — assemble start-grain training frame, reusing Section 1–4 of `score_2026_test_dates.py` byte-for-byte where possible
Layer: experiments
Acceptance: one row per pitcher-start across `CORE_FIT_SEASONS ∪ {EARLY_STOP_SEASON} = [2018, 2019, 2022, 2023]`, target = realized total K (same `pitcher_boxscore`'s `k` column, `pitcher_role == 'sp'` filter, already used identically in every prior scoring script's Section 7), features = the existing `pitcher_starts_*` frame's columns + EPIC 2's new opponent-K-rate merge. No PA-grain / slot-expansion code path — v12 never builds synthetic batter slots at all, that machinery is v6-specific and doesn't apply here.

No RED/GREEN/REFACTOR — data assembly in an experiment script, same convention as STORY 2.1 above and as every prior `train.py` in this repo.

### STORY 3.2 — fit the NB2 GLM
Layer: experiments
Acceptance: `statsmodels.discrete.discrete_model.NegativeBinomial(y_train, X_train).fit()` converges (check `.mle_retvals['converged']`) on the assembled training frame, yields a fitted `alpha` (`.params['alpha']` or the model's own dispersion attribute — confirm exact API against the installed statsmodels version before finalizing) and per-row predicted means via `.predict(X)`.

No RED/GREEN/REFACTOR — model fitting in an experiment script, same convention as XGBoost fitting in every prior `train.py` (untested; only the utilities it calls, per EPIC 1, are tested).

### STORY 3.3 — per-start pmf + evaluation against existing frameworks
Layer: experiments
Acceptance:
1. For each scored start, build `negative_binomial_pmf(predicted_mean, fitted_alpha, max_k)` (EPIC 1's function; `max_k` should be generous enough to hold negligible tail mass — 30 is a safe default given v6's own observed strikeout range).
2. Run the SAME 2024 val-season coverage-check (50/80/95% nominal-vs-empirical coverage, fixed-line threshold check via `evaluate_hit_predictor`) `run_xgboost_uncertainty.py` already runs on v6, applied to v12's pmfs instead — direct like-for-like comparison, no new eval code.
3. Score v12 on 2026 using the SAME `SCORE_SEASON`/`BACKTEST_DATES` as `score_2026_test_dates.py`, match to the SAME real DK lines already in `edge_report_2026.parquet`'s underlying odds data (reuse `edge_report_2026.py`'s matching/devig/edge logic — point it at v12's `pred_df` instead of v6's), and run `significance_check_2026.py`'s reliability/resolution/disagreement-win-rate checks on v12's results the same way they were run on v6's and v6+isotonic's.

No RED/GREEN/REFACTOR — evaluation orchestration in an experiment script, calling only already-tested utilities (`negative_binomial_pmf`, `prob_exceeds_line`, `evaluate_hit_predictor`, `get_calibration_df`/`murphy_decomposition`).

---

## EPIC 4: `v12_results.md`
Why: every version since baseline has a structured results doc (see `v10_results.md`/`v11_results.md`'s format) — v6 itself is the one exception currently flagged as still owing one (`ROADMAP.md` item 3). v12 should not repeat that gap.
Depends on: EPIC 3

### STORY 4.1 — write the results doc
Layer: docs
Acceptance: same sections every prior results doc uses (What this pass adds / Sanity checks / Results table / Interpretation / Next steps), reporting: 2024 val-season coverage numbers side-by-side with v6's already-published numbers (`coverage_by_level: {"50": 0.5612, "80": 0.7992, "95": 0.9373}`, threshold-check `reliability=0.0018, resolution=0.0239`), and 2026 real-odds numbers side-by-side with v6-raw and v6+isotonic's already-published numbers (disagreement win rate, mean edge, reliability, resolution — all already on record in `ROADMAP.md` item 6(c) and this session's memory). The comparison table is the actual deliverable — a v12 number with nothing to compare it against doesn't answer the question this experiment exists to answer.

No RED/GREEN/REFACTOR — documentation.

---

## Before marking any task complete:
- [ ] Watched the test FAIL before writing any code
- [ ] Test failed for the right reason (feature missing, not a typo)
- [ ] Wrote minimal code — nothing extra
- [ ] All tests pass after GREEN
- [ ] No new warnings or errors in pytest output
- [ ] Tests use real code — mocks only where I/O is unavoidable (none expected in EPIC 1 — pure math, no I/O)
- [ ] Re-run `pylint src/ --fail-under=7.0` in an environment that has it installed (this planning session's audit gate couldn't) before merging
