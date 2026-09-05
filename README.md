# MLB Prop Research

An ML system for MLB prop research (DraftKings): ingest game/odds data, build point-in-time-safe features, and test whether any prediction target here clears a naive baseline before it's treated as real signal.

---

## Data

S3 bucket `mlbdk` (us-east-2), Parquet, partitioned `{table}/{year}/{YYYY-MM-DD}.parquet` — one file per table per date, past dates never overwritten.

| Source | Coverage | Notes |
|---|---|---|
| Game data (schedule, box scores, play-by-play) — MLB Stats API | 2016–present | 2020/2021 excluded from model training (COVID-shortened season) |
| Odds (team lines + player props) — The Odds API | Live daily since ~2026-05 | A one-off historical pull (10x normal credit cost) added 2025 season-long coverage for backtesting; nothing before 2025 exists |
| DraftKings slate (contest/player pool) | Live daily | Feeds lineup/slate context, not itself a training source |

Pipeline layers (see `CLAUDE.md` for the full breakdown): raw ingestion and validation (Layers 1–2) run daily via Lambda; feature engineering (Layer 3) exists as tested, reusable functions but every model still re-pulls and rebuilds full-season data in a one-off script rather than a scheduled incremental job; a Feast-based online feature store (Layer 4) is scaffolded but never materialized; a production training/inference pipeline and MLOps layer (Layers 5–7) are not started.

**Model split, consistent across all six models below:** train on seasons `[2017, 2018, 2019, 2022, 2023, 2024, 2025]` minus val/test, val = 2024 (iterated against during development), test = 2025 (locked, never touched until a model is final). `k_predictor` additionally has a live 2026 real-market backtest (66 dates, 1,160 starts matched against real DraftKings lines) — the only check in this repo against real prices rather than a naive floor.

---

## Current working thesis

**Framed as a hypothesis, not a settled result** — see the record below before trusting it further.

A batter gets ~4 plate appearances in a game, and once contact is made, the outcome — exit velocity, launch angle, where nine fielders happen to be standing — is close to a coin flip even for a great hitter (this is why BABIP is famous in sabermetrics: it barely varies by skill). A starting pitcher's game line is built from ~22 batters faced instead. Those PAs aren't fully independent — a start really can run hot or cold as a whole (confirmed via a within-game correlation check, ICC ≈ 0.01, statistically real but small) — but even with that correlation, a pitcher's game total is resting on far more draws than any one batter's boom-or-bust line. Strikeouts and walks sharpen this further: neither has to touch a ball in play at all, so they skip the noisiest step in the chain entirely.

**Evidence for:**
- K-prop (`k_predictor`) and `low_pa` both clear naive with a verified `real_improvement` / bootstrap-CI-backed precision lift — not just a better-looking metric.
- The three-true-outcomes mechanism (K/BB/HBP resolve without contact) has real support in the sabermetrics literature independent of anything built here.

**Evidence against, or still open:**
- The mechanism-side record is 2 wins, 2 mixed — `bb_predictor` and `short_outing_predictor` both land on `overconfidence_risk` (real added discrimination, calibration slightly worse than naive), not a clean sweep.
- K-prop's win doesn't survive a sharper bar: against real 2026 DraftKings odds, v6 has a confirmed calibration bug (reliability gap vs. the market, bootstrap-significant) and shows no betting edge — disagreement win rate sits at coin-flip, short of the ~52.4% break-even line.
- Why `hit_predictor` specifically loses to a two-line statistical formula is still unresolved — a methodology confound (v1–v5 bundle many features per run with no incremental verification, unlike the shrinkage baseline's one-signal-at-a-time approach) hasn't been ruled out as the real explanation, separate from "the target is just noisy."

---

## Models: baseline vs. current

Every model is checked against a naive floor first (`utils/eval.py::summarize_verdict` — `real_improvement` / `overconfidence_risk` / `calibration_only` / `no_improvement`, computed from reliability + resolution at the grain a DK prop actually resolves on, not log_loss/ROC-AUC alone). Full experiment history: `DECISIONS.md`.

| Model | DK prop | Naive floor | First real model (baseline) | Current best | Verdict |
|---|---|---|---|---|---|
| `hit_predictor` | Hit / no-hit per PA | Game-grain resolution 0.0182, ROC-AUC 0.6448 | v1–v5 XGBoost/RF, all flat vs. naive | Statistical shrinkage cascade (v2, in-season-blended) — resolution 0.0162, ROC-AUC 0.6507 | Still short of naive on resolution — best result so far is `real_improvement` **vs. earlier baselines**, not vs. naive |
| `k_predictor` (K-prop) | Strikeout per PA → total K | PA-grain PR-AUC 0.226 | LR, PR-AUC 0.270 | v6 tuned XGBoost, PR-AUC 0.284; game-grain resolution 0.0137 vs. naive's 0.0045 | `real_improvement` vs. naive — but confirmed calibration bug + no edge vs. real 2026 market odds |
| `bb_predictor` (BB-prop) | Walk per PA | PA-grain PR-AUC 0.076 | LR, PR-AUC 0.096 | Still the v1 baseline (rolling walk rate added nothing) | `overconfidence_risk` at game grain |
| `n_pa_predictor` (`low_pa`) | Batter ≤3 PA in a game | PR-AUC 0.394 (per-slot historical rate) | — (went straight to classifier reframe) | XGBoost @ 0.85 confidence → 64.7% precision, 95% CI [55.6%, 72.8%] | Real, CI-verified lift; locked as the production operating point |
| `short_outing_predictor` | SP ≤4 IP | PR-AUC 0.309 (per-innings-bucket rate) | LR, PR-AUC 0.420 — widest naive-beating margin of any model here | v1 (+trailing-3-start IP trend, rest days), PR-AUC 0.440 | `overconfidence_risk` at start grain |
| `batters_faced_predictor`* | — supporting infra | Shrinkage cascade, MAE 2.861 | Tuned XGBoost, MAE 2.741 | v2 (+workload trend, rest days), MAE 2.647 | `real_improvement` vs. the cascade — not yet switched into production |

\* Not a standalone DK prop — its target (`realized_batters_faced`) is a candidate replacement for the shrinkage cascade that `k_predictor`'s total-strikeout prediction already depends on.

Weaker candidates considered and set aside: player rest-day/will-not-start (useful as a feature, not a standalone target), stolen-base attempts (narrow population, smaller market), home-run props (power is skill-driven, but still has real batted-ball/weather variance — not clearly better than the hit problem above).

---

**Read next**, depending on what you're after:
- `ROADMAP.md` — the living plan: current priorities and backlog
- `DECISIONS.md` — dated history of every session's findings and corrections
- `src/models/hit_predictor/BENCHMARKS.md` — full experiment results table, what "beating baseline" means here
- `src/models/hit_predictor/FEATURE_GLOSSARY.md` — every feature, implemented or not
- `CODEBASE.md` — infra reference: ingestion → processing → feature store, S3 layout
- `CLAUDE.md` — repo conventions (TDD requirement, commands, test layout)

**Non-goal right now:** proving this beats the market. The bar is "real, defensible signal" — the economic/CLV layer (comparing model output against DraftKings' actual de-vigged prices) is deliberately deferred until there's a model worth staking behind.
