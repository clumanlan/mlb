# Roadmap

**Last updated:** 2026-08-22
**Purpose:** the one living doc for where this project is going and what's next. Read this first when picking up work after a gap. Update it at the end of any session that changes priorities or ships something on this list — don't let it go stale the way `implementation_plan.md`/`mlb_dashboard_plan.md` did (both deleted 2026-08-19 for exactly this reason).

---

## Vision

**The actual goal (clarified 2026-08-19): get models into production and build a shared feature store.** Not, right now, to make money betting — there's no confidence yet that this has a real edge or the leverage to act on one. The bar for what ships is "not trash" (real, defensible signal), not "beats the market." That means the economic/CLV layer (§ below) is explicitly low-priority — it matters before any real staking, not before production.

An end-to-end MLB ML system for DraftKings prop research: a shared feature store feeding **multiple models** (batter hit/no-hit, pitcher K/no-K, others as they prove out).

**Current reality:** one model (`hit_predictor`), one bespoke feature pipeline (not yet on the shared store). Feature-store convergence is deliberately deferred until model #2 exists (see Mid-term).

---

## Current state (2026-08-22)

- `hit_predictor` v1–v4 are all statistically flat vs. the naive baseline **at the per-PA grain** (log_loss within ~0.001, ROC-AUC 0.51–0.52).
- **Per-game aggregation check: done 2026-08-19, corrected 2026-08-22.** Same model/features aggregated to player-game grain (the grain a DK prop actually resolves on): resolution ~127x higher than PA grain, ROC-AUC 0.516→0.636 — that framing insight still holds. **But the original "model beats naive Brier at game grain" claim was wrong** — it compared the model against the wrong naive floor (Brier's `uncertainty` term, not an actual naive model aggregated the same way the real model is). Corrected via a proper head-to-head (`utils/eval.py::run_pa_vs_game_grain_check`, run through v4): **a properly-aggregated naive baseline beats the model on every game-grain metric** (ROC-AUC 0.6448 vs 0.6361, Brier 0.2230 vs 0.2259). Full writeup: `BENCHMARKS.md` §4.5. **Game grain is still the primary evaluation target**, but "beat naive" is not yet a demonstrated result — see near-term backlog item 1 below (new).
- Root cause of the naive baseline's game-grain edge: it's constant per PA, but a batter's `n_pa` (plate appearances) varies by game, and `1-∏(1-p)` with more PAs mechanically raises P(1+ hit) — real signal the naive model picks up for free that the model isn't clearly beating.
- Separately, aggregated probabilities are currently overconfident (the `1-∏(1-p)` independence assumption doesn't account for correlation across a batter's same-game PAs) — needs recalibration, but note recalibration fixes calibration, not the discrimination gap above. See near-term backlog.
- **v4 (inning-based pitcher-role gating + game context): run 2026-08-22.** Result: statistically flat vs. v2 at both grains (PA ROC-AUC 0.516 vs 0.517, game ROC-AUC 0.636 vs 0.636) — the new features made no measurable difference. `expected_pitcher_role` (inning-based) agreed with the realized role 88.9% of PAs, but v4 changed the role-gating logic and added game-context features in the same experiment, so which change (if either) mattered isn't separable from this run.
- No economic/betting eval layer exists (devig, calibration-in-the-tails, CLV). Not urgent — see Vision above.
- Full experiment history + metrics table lives in `src/models/hit_predictor/BENCHMARKS.md` §1 — append new rows there per run, don't duplicate the table here.
- `research/feature_glossary_gap_analysis.md` has a sourced, tiered backlog of candidate features (26 candidates, external-literature-backed) — Tier 1 is build-now, no new ingestion required.

---

## Near-term backlog (hit_predictor)

Priority order, revised 2026-08-22 (v4 ran, naive-baseline correction surfaced):

1. **[ ] Explain/close the naive-baseline gap at game grain** — the corrected naive floor beats the model on ROC-AUC, resolution, and Brier (see Current state above, `BENCHMARKS.md` §4.5 correction + §5 item 9). Start with: is `n_pa` (or a pre-game estimate of it) already an explicit model feature? If not, that's the most likely quick win. This is now the top priority — every other item below is secondary until the model demonstrably beats a fair baseline.
   - **Caveat surfaced 2026-08-22:** realized `n_pa` (what the naive baseline's aggregation implicitly benefits from) is not known at prediction time in production — the model only ever scores starters, so `create_pa_outcome`'s inner join dropping non-starter PAs (`pipeline.py:283`, via `_create_batting_order`'s null-`batting_order` filter) is fine as-is, not a bug. But it means closing this gap requires *predicting* `n_pa` pre-game (lineup slot, team scoring environment, opposing pitcher's expected innings, etc. — not reading it off box score data the way training/backtesting does), not just adding realized `n_pa` as a feature, which would leak.
2. **[ ] Recalibrate aggregated game-grain predictions** — isotonic regression (or similar) fit on `game_pred_prob` at the game grain, to fix the overconfidence the aggregation check surfaced. Note: this fixes calibration, not item 1's discrimination gap — do both, but item 1 matters more for "is there a real model here."
3. **[ ] Build Tier 1 features from `research/feature_glossary_gap_analysis.md`**, in its suggested order — extended log5 matchup, spray angle/pull%, proper Barrel% (EV/LA curve), count-state splits, then further down the list.
4. **[ ] Isolate v4's two changes** — v4 bundled inning-based role gating and new game-context features into one experiment; both together were flat vs. v2. If there's reason to believe one half is worth keeping, a v5 that isolates them (one change at a time) would show which, if either, is pulling weight — not scheduled, just noted as the open thread v4 left.
5. **[ ] Economic eval layer** (devig, edge-vs-vig report, calibration-focused model selection, eventually CLV) — deferred, not blocking. Revisit before this model is ever used to size a real bet, not before production/feature-store work.

---

## Mid-term

- **Model #2 — pitcher K props.** `BENCHMARKS.md` §4.4 flags K-rate as structurally stickier than hit outcomes (DIPS-era "three true outcomes" reasoning) — the natural next model, and untested hypothesis worth checking once hit_predictor's open questions above settle.
- **Feature-store convergence** — `hit_predictor/processing/features/` (bespoke pandas, PA-grain, point-in-time-shift) and `src/features/transforms/` (Feast-oriented, DuckDB/SQL) stay separate **until model #2 exists** (decided 2026-08-19: don't generalize a shared store from a single example — let a second model's real needs reveal what's actually shared).

---

## Parked / explicitly deferred decisions

Revisit these, don't act on them yet:

- **Legacy `src/models/` siblings** (`experimentation/sequence_modeling.py`, `starting_pitcher_pa_hits/train.py`, `team_runs/train.py`) — single flat files, no tests, unclear if dead or early drafts of a future model. Decided 2026-08-19: leave alone for now, audit/decide when model #2 actually starts (one of them may already be the right shape for a K-prop model).
- **`dashboard/frontend/node_modules/` has ~1320 files already tracked in git** — found during the `.gitignore` fix (2026-08-19 commit `cfd88d8`), out of scope for that fix. Needs its own cleanup pass (add `node_modules/` to `.gitignore`, `git rm -r --cached`).
- **Experiment file duplication** — `experiments/v1–v4/train.py` are ~50% copy-pasted between consecutive versions (deliberate, for reproducible frozen snapshots). Fine at 4 versions; reconsider extracting a shared `run_experiment()` core if a v5+ makes the duplication cost outweigh the reproducibility benefit.

---

## Working conventions

- **Per-task plans stay ephemeral.** Use the `/impl-planning` skill (per `CLAUDE.md`'s TDD Requirement) for one task at a time; delete the plan doc once the work merges, same as this session did for `implementation_plan.md`. This file (`ROADMAP.md`) is for standing priorities and direction, not task-level implementation detail.
- **`BENCHMARKS.md`** is the running experiment log (results table) and the place model-quality reasoning lives — update its table after every run, including v4's. **After every experiment (yours or Claude-built), fill in ECE/Reliability/Resolution alongside LogLoss/Brier/ROC-AUC** — reliability (calibration error, lower better) and resolution (discrimination, higher better) together are the real "did this help" signal, not log_loss alone; both are already computed by `evaluate_hit_predictor()` and logged to MLflow automatically (decided 2026-08-19, see `BENCHMARKS.md` §1).
- **`FEATURE_GLOSSARY.md`** is the feature reference — check before adding a feature or wondering if one exists.
- Update this file's "Current state" and "Near-term backlog" sections whenever a backlog item ships or priorities change — that's what keeps it worth reading cold.
