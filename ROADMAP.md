# Roadmap

**Last updated:** 2026-08-19
**Purpose:** the one living doc for where this project is going and what's next. Read this first when picking up work after a gap. Update it at the end of any session that changes priorities or ships something on this list — don't let it go stale the way `implementation_plan.md`/`mlb_dashboard_plan.md` did (both deleted 2026-08-19 for exactly this reason).

---

## Vision

**The actual goal (clarified 2026-08-19): get models into production and build a shared feature store.** Not, right now, to make money betting — there's no confidence yet that this has a real edge or the leverage to act on one. The bar for what ships is "not trash" (real, defensible signal), not "beats the market." That means the economic/CLV layer (§ below) is explicitly low-priority — it matters before any real staking, not before production.

An end-to-end MLB ML system for DraftKings prop research: a shared feature store feeding **multiple models** (batter hit/no-hit, pitcher K/no-K, others as they prove out).

**Current reality:** one model (`hit_predictor`), one bespoke feature pipeline (not yet on the shared store). Feature-store convergence is deliberately deferred until model #2 exists (see Mid-term).

---

## Current state (2026-08-19)

- `hit_predictor` v1–v3 were all statistically flat vs. the naive baseline **at the per-PA grain** (log_loss within ~0.001, ROC-AUC 0.51–0.52). **Resolved 2026-08-19: this was partly a framing problem, not a features ceiling** — see below.
- **Per-game aggregation check: done, positive result.** Same model/features as v2, aggregated to player-game grain (the grain a DK prop actually resolves on): resolution ~127x higher, ROC-AUC 0.516→0.636, and the model beats naive Brier for the first time in this project (it didn't at PA grain). Full writeup: `BENCHMARKS.md` §4.5. **Game grain is now the primary evaluation target**, not PA grain.
- One real caveat from that result: aggregated probabilities are currently overconfident (the `1-∏(1-p)` independence assumption doesn't account for correlation across a batter's same-game PAs) — needs recalibration before being trusted at face value. See near-term backlog.
- v4 (inning-based pitcher-role gating + game context) is built but **not yet run**.
- No economic/betting eval layer exists (devig, calibration-in-the-tails, CLV). Not urgent — see Vision above.
- Full experiment history + metrics table lives in `src/models/hit_predictor/BENCHMARKS.md` §1 — append new rows there per run, don't duplicate the table here.
- `research/feature_glossary_gap_analysis.md` has a sourced, tiered backlog of candidate features (26 candidates, external-literature-backed) — Tier 1 is build-now, no new ingestion required.

---

## Near-term backlog (hit_predictor)

Priority order, decided 2026-08-19:

1. **[ ] Recalibrate aggregated game-grain predictions** — isotonic regression (or similar) fit on `game_pred_prob` at the game grain, to fix the overconfidence the aggregation check surfaced. Do this before treating any game-grain number as trustworthy.
2. **[ ] Build Tier 1 features from `research/feature_glossary_gap_analysis.md`**, in its suggested order — extended log5 matchup, spray angle/pull%, proper Barrel% (EV/LA curve), count-state splits, then further down the list. Better-justified now than before the aggregation result: the feature set has demonstrated real signal, it just needs the right grain to show it.
3. **[ ] Run v4** (inning-based role gating + game context) — close the loop on whether the inning-based estimate beats the PA-position one. Evaluate at the game grain, not PA grain, given the above.
4. **[ ] Economic eval layer** (devig, edge-vs-vig report, calibration-focused model selection, eventually CLV) — deferred, not blocking. Revisit before this model is ever used to size a real bet, not before production/feature-store work.

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
