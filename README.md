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

## The prop matrix

Every "simple" prop this project targets is one of three per-plate-appearance classifiers — hit, walk, strikeout (home run not yet built) — rolled up two different ways: grouped by batter (needs `n_pa_predictor`'s PA-count estimate) for a batter prop, or grouped by pitcher (needs `batters_faced_predictor`'s estimate) for the "allowed" version of the same prop. Outs recorded isn't its own model — it falls out of batters faced minus hits and walks allowed.

![The Prop Matrix](docs/prop_matrix.svg)

Two things this makes visible that the numbers alone don't: **Pitcher Hits Allowed and Pitcher Walks Allowed need zero new modeling** — both classifiers and the count model already exist, nobody has combined them — and **"built" doesn't currently mean production-ready for anything except `k_predictor`**. `hit_predictor` and `bb_predictor`'s existing game-level checks are backtests against the *realized* plate-appearance count from the box score, not something computable before a game actually starts.

## Models ranked by performance vs. naive

Ranked by verdict tier first (`real_improvement` beats `overconfidence_risk` beats "hasn't cleared naive"), then by relative lift within a tier. Metrics differ by model (PR-AUC, precision, MAE, resolution) — treat this as directional, not a literal cross-model comparison. Full experiment history: `DECISIONS.md`.

| Rank | Model | Bin | Performance vs. naive |
|---|---|---|---|
| 1 | `n_pa_predictor` (`low_pa`) | Batter | 19.8% base rate → 64.7% precision @ 0.85 confidence (+45pp, ~3.3x) — `real_improvement`, threshold already locked |
| 2 | `k_predictor` (K-prop) | Pitcher | PR-AUC 0.226 → 0.284 (+26%) — `real_improvement`, but confirmed no edge against real 2026 market odds |
| 3 | `batters_faced_predictor`* | Pitcher (infra) | MAE 2.861 → 2.647 (-7.5%) — `real_improvement`, but vs. the cascade formula, not naive |
| 4 | `short_outing_predictor` | Pitcher | PR-AUC 0.309 → 0.440 (+42%) — the largest raw lift of any model here, but `overconfidence_risk` |
| 5 | `bb_predictor` (BB-prop) | Pitcher | PR-AUC 0.076 → 0.096 (+26%) — `overconfidence_risk` |
| 6 | `hit_predictor` | Batter | Resolution 0.0182 → 0.0162 (-11%, worse) — still hasn't beaten naive |

\* Not a standalone DK prop — supporting infra for `k_predictor`'s total-strikeout rollup.

Weaker candidates considered and set aside: player rest-day/will-not-start (useful as a feature, not a standalone target), stolen-base attempts (narrow population, smaller market).

---

**Read next**, depending on what you're after:
- `ROADMAP.md` — the living plan: current priorities and backlog
- `DECISIONS.md` — dated history of every session's findings and corrections
- `src/models/hit_predictor/BENCHMARKS.md` — full experiment results table, what "beating baseline" means here
- `src/models/hit_predictor/FEATURE_GLOSSARY.md` — every feature, implemented or not
- `CODEBASE.md` — infra reference: ingestion → processing → feature store, S3 layout
- `CLAUDE.md` — repo conventions (TDD requirement, commands, test layout)

**Non-goal right now:** proving this beats the market. The bar is "real, defensible signal" — the economic/CLV layer (comparing model output against DraftKings' actual de-vigged prices) is deliberately deferred until there's a model worth staking behind.
