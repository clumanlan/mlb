# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## TDD Requirement

All new code in this project follows Test-Driven Development. Before planning or implementing any feature, run the `/impl-planning` skill located at `.claude/skills/impl-planning/`. No production code without a failing test first.

---

## Documentation Conventions

No emoji anywhere in this repo — markdown docs, code comments, commit messages, or CLI output strings. Mark status with plain text instead:

- Pipeline/architecture status (layer diagrams, integration tables): `[done]`, `[in progress]`, `[not started]`
- Have/partial/gap-style reference columns (e.g. `FEATURE_GLOSSARY.md`): bold text tags — `**have it**`, `**partial**`, `**gap**`
- Inline "this shipped" callouts: bold the status word itself (`**Built, positive.**`, `**Confirmed.**`) rather than prefixing an icon

Plain text stays legible in diffs, terminals, and screen readers, and renders consistently everywhere an icon might not.

---

## Commands

```bash
# Activate the virtual environment (required before running anything)
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_daily_process_data_handler.py -v

# Run a single test by name
pytest tests/test_daily_process_data_handler.py::TestClassName::test_method_name -v

# Lint
pylint src/ --fail-under=7.0

# Deploy a Lambda (after zipping)
make deploy-mlb-fetch
make deploy-process-data
make deploy-odds-fetch

# Build Lambda zip
make zip-mlb-fetch
make zip-odds-fetch
```

---

## System Architecture

This is an end-to-end MLB ML system built for DraftKings prop betting. The pipeline runs daily and flows through these layers:

```
Layer 1 — Raw Ingestion       [done]         fetch from MLB Stats API → S3 (raw_data/)
Layer 2 — Validation          [done]         quality gate before downstream runs
Layer 3 — Feature Engineering [in progress]  rolling window features → S3 (features/offline/)
Layer 4 — Feature Store       [not started]  Feast (offline: S3/Athena, online: DynamoDB)
Layer 5 — Model Training      [in progress]  XGBoost baseline → attention model (see "Model Layer" below)
Layer 6 — Prediction Pipeline [not started]  daily batch inference, lineup-aware
Layer 7 — MLOps               [not started]  MLflow, Evidently AI, CloudWatch
```

---

## S3 Bucket: `mlbdk` (us-east-2)

All data is Parquet, partitioned as `{table}/{year}/{YYYY-MM-DD}.parquet`. One file per table per date — never overwrite past dates.

```
raw_data/games/{schedule,game_info,batter_boxscore,pitcher_boxscore}/{year}/{date}.parquet
raw_data/playbyplay/{year}/{date}.parquet
raw_data/draftkings/{year}/{date}.parquet
raw_data/odds/{team_odds,player_props}/{year}/{date}.parquet
reference/player_info/player_info.parquet          ← not date-partitioned
processed_data/games/{table}/{year}/{date}.parquet
processed_data/prepared/{batter_boxscore,pitcher_boxscore,playbyplay}/{year}/{date}.parquet
processed_data/season_summaries/{batter_season_summary,sp_season_summary,team_season_summary}/game_season={year}/*.parquet  ← not currently read by any model, kept as-is (see Feature Engineering section)
features/offline/{batter_features,pitcher_features}/{year}/{date}.parquet
feast/features/{batter_lineup,batter_pa_volume}/{year}/backfill.parquet  ← Feast offline FileSource, real as of 2026-09-15
feast/registry.pb                                  ← Feast registry, created by `feast apply` from src/features/
lambdas/status/{function_name}/{date}.json         ← observability, always production path
```

---

## Lambda Observability — Status Writer

Every Lambda must write a status JSON to S3 at the end of execution (success or failure) using the shared utility at `src/shared/status_writer.py`.

**S3 path:** `lambdas/status/{function_name}/{date}.json` in the `mlbdk` bucket — always production path, no env prefix.

**Schema:**
```json
{
  "function_name": "daily_mlb_fetch",
  "run_date": "2026-05-03",
  "status": "success",
  "started_at": "2026-05-03T10:00:00.000000",
  "completed_at": "2026-05-03T10:01:23.456789",
  "duration_seconds": 83.4,
  "games_processed": {
    "schedule": 15,
    "game_info": 15,
    "batter_boxscore": 15,
    "pitcher_boxscore": 15,
    "playbyplay": 15
  },
  "error": null
}
```

`games_processed` is a **dict** keyed by processing step → unique game count. The shape differs per Lambda:

| Lambda | `games_processed` keys | Status writer integrated? |
|---|---|---|
| `daily_mlb_fetch` | `schedule`, `game_info`, `batter_boxscore`, `pitcher_boxscore`, `playbyplay` | Yes |
| `daily_process_data` | `schedule`, `batter_prepared`, `pitcher_prepared`, `playbyplay_prepared` | Yes |
| `daily_odds_fetch` | `team_odds`, `player_props` | Yes |
| `daily_feature_create` | — | Not yet integrated |

The goal is to verify that all games were captured at each step — a drop from 15 to 13 in `batter_boxscore` signals a data gap.

**Integration pattern:** capture `started_at` before the try block, then use `try/except`. On success call `write_status` before returning; on failure call `write_status` in the except block then re-raise. All four Lambdas (including `daily_odds_fetch`) re-raise on failure rather than returning a 500 body — the CloudWatch `Errors` alarm and SQS dead-letter queue both key off Lambda's native unhandled-exception accounting, which never fires for a normally-returned response. Import as a sibling module (`from status_writer import write_status`).

```python
started_at = datetime.utcnow()
games_processed = {}
try:
    # ... handler logic, populate games_processed ...
    completed_at = datetime.utcnow()
    write_status(function_name="...", run_date=date, status="success",
                 started_at=started_at.isoformat(), completed_at=completed_at.isoformat(),
                 duration_seconds=(completed_at - started_at).total_seconds(),
                 games_processed=games_processed, error=None)
    return {...}
except Exception as e:
    completed_at = datetime.utcnow()
    write_status(..., status="failed", games_processed=games_processed, error=str(e))
    raise
```

**Bundling:** `src/shared/status_writer.py` is copied into each Lambda zip via `cp src/shared/*.py $(BUILD_DIR)/` in the Makefile (added to `zip-mlb-fetch`, `zip-process-data`, `zip-odds-fetch`).

**IAM note:** Lambda execution roles need `s3:PutObject` on `arn:aws:s3:::mlbdk/lambdas/status/*` — not yet added to Terraform.

---

## Lambda Pipeline

Lambda handlers import sibling modules by name (not as packages) — this is how they're bundled into Lambda zip files via `make zip-*`. The sibling modules live in `src/data/modules/` (for fetch/process lambdas), `src/shared/` (shared utilities), and `src/features/transforms/` (for feature lambda).

### MLB Data Pipeline

Runs daily at 10 AM UTC (5 AM EST) via EventBridge → Step Functions.

| Lambda | Handler | What it does |
|---|---|---|
| `daily_mlb_fetch` | `src/lambdas/daily_mlb_fetch/handler.py` | Fetches from MLB Stats API, writes raw Parquet to S3 |
| `daily_process_data` | `src/lambdas/daily_process_data/handler.py` | Cleans + joins raw tables, writes processed and prepared tables |
| `daily_feature_create` | `src/lambdas/daily_feature_create/handler.py` | Computes today's snapshot for `batter_lineup`/`batter_pa_volume` and writes to S3 — no `feast` import (fixed 2026-09-16, split from materialization 2026-09-17). Triggered by an S3 event notification on `daily_lineup_fetch`'s per-game lineup writes (`raw_data/games/lineups/{year}/{date}/{game_pk}.json`) — no fixed schedule, since lineups confirm at different real times per game; skips the states-tracking file that shares the same S3 prefix. On success, invokes `daily_feature_materialize` asynchronously. `memory_size=2048`/`timeout=900` (bumped 2026-09-18 after a real OOM — see Known Issues). Terraform-managed, deployed, proven live via a real smoke test — see `DECISIONS.md`'s 2026-09-18 entry |
| `daily_feature_materialize` | `src/lambdas/daily_feature_materialize/handler.py` | Calls `FeatureStore(...).materialize_incremental()` against the real Feast registry — the only place `feast` is imported (added 2026-09-17). Container-image Lambda (`package_type=Image`, `architectures=["arm64"]`), not zip+layers — `feast[aws]`'s real dependency closure (~470MB) doesn't fit Lambda's 250MB zip+layers cap. Invoked directly by `daily_feature_create` on success, not on its own schedule. Deployed and confirmed live via a real S3-triggered smoke test: CloudWatch logs show both feature views materialized, `aws dynamodb scan` confirmed the row-count delta |

Each accepts `{date: "YYYY-MM-DD", env: "test"}` — `env: test` routes all S3 reads/writes under a `test/` prefix, leaving production data untouched.

### DraftKings Slate Pipeline

| Lambda | Handler | What it does |
|---|---|---|
| `daily_dkslate_fetch` | `src/lambdas/daily_dkslate_fetch/handler.py` | Fetches DraftKings slate |

### Odds Pipeline (independent, no Step Functions)

Runs independently. Triggered manually or via a separate EventBridge rule. Accepts `{date: "YYYY-MM-DD"}`.

| Lambda | Handler | What it does |
|---|---|---|
| `daily_odds_fetch` | `src/lambdas/daily_odds_fetch/handler.py` | Fetches team odds + player props from The Odds API (DraftKings book), writes Parquet to S3, tracks monthly quota in SSM |

**Odds modules** (in `src/data/modules/`, bundled via `make zip-odds-fetch`):
- `odds_fetch.py` — live API client: `get_games`, `get_team_odds`, `get_player_props`, `get_all_player_props`. Also `get_historical_games`, `get_historical_player_props`, `get_all_historical_player_props` — historical-endpoint equivalents (added 2026-08-31 for k_predictor's odds backtest, see below), not used by the daily Lambda.
- `odds_quota.py` — SSM quota helpers: `get_monthly_usage`, `check_quota`, `set_monthly_usage` (writes the real usage reported by The Odds API's `x-requests-used` response header — not a locally-incremented guess)

**S3 output:**
```
raw_data/odds/team_odds/{year}/{date}.parquet
raw_data/odds/player_props/{year}/{date}.parquet
```

**SSM parameters:**
- `/mlb/odds-api/api-key` — SecureString, API key for api.the-odds-api.com
- `/mlb/odds-api/requests-used/{year}/{month}` — running monthly request count (limit: 20,000, since the 2026-08-27 plan upgrade)

**Quota behavior:** warns at 90% (18,000 calls), hard-stops at 100% (20,000 calls), raises an exception with "quota" in the message.

**Smoke test:**
```bash
aws lambda invoke \
  --function-name daily_odds_fetch \
  --region us-east-2 \
  --payload '{"date": "YYYY-MM-DD"}' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json
```

**Historical odds pulls (backtests only, not the daily Lambda):** The Odds API's historical endpoint (`/v4/historical/sports/...`) bills at **10x** the normal `markets × regions` rate and is the only way to get odds for any date the daily Lambda didn't run on (e.g. all of 2025 — the live pipeline only started ~2026-05). Snapshots go back to Sept 2022.

**Snapshot-timing gotcha (real bug, cost ~480 wasted credits on 2026-08-31 before being caught):** the historical endpoint only returns markets that were *live at the requested snapshot moment* — same "only returns still-open markets" behavior as the live endpoint (see Known Issues below), but easier to hit by accident here because a historical pull naturally wants "give me everything for this whole day." Querying the games-*listing* call with an end-of-day snapshot (e.g. `date=2025-04-09T23:59:59Z`) returns only the handful of games still in progress at that instant — most of the day's games had already finished and their markets had rolled off. `get_all_historical_player_props` fixes this by listing games at `{date}T11:00:00Z` (7am ET, safely pregame for any real MLB start time) and only using each game's own `commence_time` as the snapshot for that game's individual player-props call. **Before trusting a historical pull's game counts, cross-check them against `raw_data/games/schedule/{year}/{date}.parquet`'s real game count — do this before spending credits on props, not after**, the way the 2026-08-31 bug was actually caught (a 2/15-games day was the tell).

A second, unrelated gotcha: individual events can 404 on the per-event historical props call (a genuine gap in the historical archive for that specific game, not a bug) — `get_all_historical_player_props` catches and skips these per-event rather than aborting the whole day's fetch, logging a warning.

**Backtest artifacts:** `src/models/k_predictor/backtest/` (`fetch_2025_odds.py`, `score_2025_test_dates.py`, `edge_report.py`) is the first user of the historical endpoint — a one-off analysis script, not part of the Lambda pipeline. See `ROADMAP.md`'s 2026-08-31 entry for the result.

---

## Feature Engineering

**2026-09-15: this section was rewritten after deleting a first-generation feature-store attempt that never made it to production.** `src/features/transforms/` used to hold 4 original feature-group files (`team_batter_base.py`, `player_batter_base.py`, `starting_pitcher_base.py`, `bullpen_pitcher_base.py`), a `jobs/` (nee `backfill/`) folder of one-off CLI loaders for them, and `compute_season_summaries.py` + a private `utils.py`. None of it was consumed by any model — every model has always run its own bespoke `processing/features/` pipeline instead (see Model Layer below) — no `feast apply` had ever run against the `FeatureView`s it fed, and `daily_feature_create` (the Lambda meant to keep it updated) had dangling imports to modules that never existed (`sp_features`, `bullpen_features`). Deleted along with ~317MB of the S3 data it had produced (`s3://mlbdk/feast/features/*`) — kept `processed_data/season_summaries/*`, since deleting that wasn't part of the ask. Full reasoning: `DECISIONS.md`'s 2026-09-15 entries. Standing principle going forward, stated by the user directly: **only build and maintain what feeds a model or informs a real decision — delete stale code and data aggressively rather than letting it accumulate "just in case."**

`src/features/transforms/` currently contains:
- `batter_lineup.py` — `batting_order`, backfill (box score) + incremental (announced lineup) — added 2026-09-15 for `n_pa_predictor`'s frozen feature set, the first file in this directory built for the feature store rather than as a one-off script
- `batter_pa_volume.py` — batter's own rolling `avg_n_pa_per_game`, same backfill + incremental split — added 2026-09-15, same reason
- `data_readers.py` — shared S3 readers (`read_batter_boxscore(season)`, etc.) using `awswrangler`

### Naming & organization conventions for `src/features/transforms/`

Scoped out 2026-09-15 from how Chronon (Airbnb) and Michelangelo (Uber) structure this problem at platform-team scale, recalibrated for a solo/small-team repo (their research is cited in `DECISIONS.md`'s 2026-09-15 entries). The principle that transfers regardless of scale is "one definition, thin adapters on both sides" — this repo gets that by hand (a model's `processing/features/` owns the logic, the transforms file only imports and reshapes it) rather than via a declarative DSL that auto-generates batch/streaming compute, because a human writing two small functions is cheaper than building that DSL at this repo's scale. The small-team-specific calibration (also 2026-09-15): don't build a feature store until 2+ models actually share a feature (already how this repo got here — Feast conversion was deferred until model #2 existed), and don't build a registry/catalog beyond this directory listing until the number of files actually makes "what exists" hard to answer by reading it.

- **File naming**: `{entity}_{concept}.py` (e.g. `batter_lineup.py`, `batter_pa_volume.py`). One file per feature, or per tightly-coupled feature family computed by a single query — don't split a family sharing one DuckDB `SELECT` into one-column-per-file just to satisfy a "one feature, one file" rule; that fights the engine rather than helping readability. **No `_base` suffix** — it was inherited from the deleted first-generation files (where it may have loosely meant "foundational per-entity stats") but doesn't currently distinguish anything, since nothing in this directory exists to contrast it against (dropped 2026-09-15, see `DECISIONS.md`). Reintroduce a suffix like this only if a real second category shows up later — e.g. a `_derived`/`_interaction` file combining multiple base features — where the distinction would actually mean something.
- **Function naming**: `compute_{name}_features(year)` for the backfill entry point (offline store, full season), `compute_{name}_features_for_date(date)` for the incremental entry point (online store, today's snapshot only). Both normalize to the same Feast-ready schema (`personId`/`team_id`, `event_timestamp`, ...).
- **Placement rule**: a feature lives in a model's `processing/features/` (or `experiments/v{N}_*/`, reading raw/processed S3 directly) while it's still being tried. It only gets a file here after clearing an ablation-style freeze — and even then, this layer imports the canonical logic, it never redefines it (violated once, fixed 2026-09-15 — see the Rule below).
- **Promotion checklist**, in order: (1) ablation/CI-verified freeze against the model's locked operating point, (2) a backfill vs. incremental **value**-consistency test on real overlapping data, not just column/dtype parity (`test_backfill_and_incremental_produce_identical_schema`-style tests check schema only and are not sufficient on their own — this is the actual train/serve-divergence check per Chronon/Nubank's research), (3) only then written as a file here.
- **What not to build yet**: no feature registry/manifest file, no code-generation from a declarative spec. Revisit if/when this directory's file count makes a plain listing an inadequate answer to "what features exist."

**Rule — this layer imports feature logic, it never redefines it.** A file in `src/features/transforms/` may only reshape/rename output from a feature computation that already lives in a model's `processing/features/`, or in the neutral shared layer at `src/data/modules/preprocessing.py` for logic generic enough that no one model should own it (see the Processing Layer section above — e.g. `create_batting_order`) — never reimplement the underlying logic itself. If no reusable implementation exists yet, add it to the owning model's `processing/features/` and import it here, don't write it fresh; if it turns out multiple models (or this layer) need the identical computation, that's the signal to promote it to `data/modules/` rather than one side importing another model's internals. This was violated once (`batter_lineup.py`, then still named `batter_lineup_base.py`, reimplemented `_create_batting_order`'s null-filter/dedup instead of calling it, fixed 2026-09-15 — see `DECISIONS.md`) and is exactly the "recreate the transform twice" anti-pattern that causes silent training/serving divergence industry-wide (Chronon, AWS ML Well-Architected).

**Point-in-time safety:** rolling windows close the day *before* the target game date — no same-day leakage.

---

## Model Layer — `src/models/`

The end goal is a shared feature store feeding **multiple models** (batter hit/no-hit, pitcher K/no-K, and others as they're built). `hit_predictor` (`src/models/hit_predictor/`) was the first model and is still where the batter-side hit-probability problem is worked on. `src/models/k_predictor/` (pitcher strikeout probability), `src/models/n_pa_predictor/` (batter plate-appearance count / `low_pa` classifier), `src/models/bb_predictor/` (pitcher walk probability), `src/models/short_outing_predictor/` (starting-pitcher short-outing probability — the last candidate in README's sub-problem menu), and `src/models/batters_faced_predictor/` (starting-pitcher batters-faced regression) are newer sibling models — see `README.md`'s sub-problem menu and `ROADMAP.md`'s Mid-term section for current status of each. `batters_faced_predictor` isn't itself a README sub-problem-menu candidate (it's not a standalone DK prop) — it's supporting infrastructure whose target, `realized_batters_faced`, is a better-tuned replacement candidate for `game_context.py`'s `build_expected_batters_faced` shrinkage cascade, which `k_predictor`'s total-strikeout prediction already depends on. All five compose `hit_predictor`'s existing target-agnostic feature-building machinery (season/rolling stats, role gating) via their own thin `processing/pipeline.py` + `processing/schema.py`, rather than duplicating it. `short_outing_predictor` and `batters_faced_predictor` are the two that run at a different grain than the rest — one row per starting-pitcher-*start*, not per PA or batter-game — with `short_outing_predictor` reusing `game_context.py`'s `build_expected_start_innings` (a pre-game workload-shrinkage estimate originally built for `n_pa_predictor`'s opposing-starter feature) for the pitcher's own start instead. Each still runs its own bespoke processing pipeline rather than consuming from the Layer 4 Feast store above, because the feature engineering it needs (PA-grain, point-in-time-safe, extensively feature-engineered) is still being worked out model-side before it's worth generalizing into the shared store. Expect this pipeline's proven patterns (rolling windows, point-in-time shifting, role-aware pitcher splits) to migrate toward Layer 4 now that a second model (`k_predictor`) exists and needs the same features — see `ROADMAP.md`'s "Feature-store convergence" item.

**`batters_faced_predictor` status** (see `ROADMAP.md`'s 2026-08-26/2026-08-27 entries and `src/models/batters_faced_predictor/{baseline,v1,v2,v3,v4,v5,v6,v7}_results.md` for full numbers): baseline positive (tuned XGBoost beats the cascade, MAE 2.7411 vs. 2.8609 — first model in this repo to beat a frozen shared-infrastructure formula on real evidence), v1 flat-on-new-features/positive-on-tuning (2.7347), v2 `real_improvement` (2.6471 — trailing-3-start PA trend + pitcher's own rest days + a pitches-thrown workload-density signal, targeted at established starters getting pulled early despite a clean box line), v3 flat (2.6479 — a multi-season lookback aimed at cold-start pitchers; the bucket itself looks better than the cascade but the gain was already captured by v2, so this specific fix didn't add anything net-new — genuine small-sample variance, not a missing-data gap), v4 flat (2.6432 — a same-season anomaly-count feature targeted at the same established-starter-pull failure mode v2 partially closed; ranked dead last in feature importance, essentially unused by the model — closes that failure-mode thread). v5 and v6 opened a new, independent thread — the opposing team's own scoring, not the pitcher's own workload/history — and both came back flat too: v5 (2.6369) tried scoring LEVEL (win_pct/runs_scored/run_diff, season + trailing-5-game, via the already-existing `build_team_win_loss_record`), v6 (2.6349) tried scoring VOLATILITY (mean/std/max of runs scored, via new TDD'd `build_team_scoring_volatility`) — level and volatility, at both windows, have now all been tried and all rank low in feature importance. v7 (2.6405, flat vs. v2) tried two more threads, both zero new production code: pitch-count TREND (trailing-3-start delta vs. season average, via the same `build_pbp_pitcher_rolling_feats` v1 already uses) and own-team BULLPEN STRENGTH (whip/k_rate/bb_rate/hr_rate/strike_rate pooled by team via `build_pitcher_rolling_stats_all_roles`, joined on the starter's own team — the first self-team rather than opposing-team feature tried here). Aggregate MAE stayed flat, but feature importance surfaced a real secondary finding: the trailing-3 pitch-count *level* (a new raw window, not the trend transform itself) is the #2 most important feature overall, while the trend ratio/direction built to test that hypothesis ranked near dead last — bullpen WHIP placed modestly, the other four bullpen rate stats did not. `build_expected_batters_faced` itself is still unmodified — a production switch-over is a separate, not-yet-made decision; v2's model (2.6471) remains the strongest and standing candidate after six follow-up passes (v3-v7) failed to beat it.

**Structure:**
- `processing/pipeline.py` — assembles the PA-outcome training grain from raw pbp/boxscore/schedule/game_info (`create_pa_outcome`)
- `processing/features/` — one file per feature family: `season_stats.py` (season-level batter/pitcher rates), `rolling_stats.py` (game-by-game rolling windows), `park_factors.py`, `expected_role.py` (pre-game-knowable SP/bullpen role gating), `interaction_feats.py` (trend/shrinkage features), `game_context.py` (calendar, team form/rest, probable starters, starter-innings estimate)
- `processing/schema.py` — pbp column schema, the single source of truth this pipeline reads against
- `baseline/` — naive and rules-based baselines, run before any real model (`baseline/rules/run_baseline.py`, `baseline/model/run.py`)
- `experiments/v{N}_*/train.py` — versioned experiments, each an isolated, runnable pipeline (own `mlruns/` for MLflow tracking); read the newest version's summary comment block first — it documents exactly what changed vs. the prior version
- `utils/` — `eval.py` (metrics + calibration plots), `mlflow_logging.py`, `model_prep.py` (missing-indicator/impute/encode helpers)
- `config.yaml` — seasons (train/val/test split), target column, drop columns

**Reference docs:** `FEATURE_GLOSSARY.md` is the feature-by-feature reference (what each stat measures, why, and whether it's implemented) — check it before adding a new feature or wondering if one already exists. `dashboard_spec.md` documents the portable Streamlit model-diagnostics dashboard (implemented at `src/models/hit_predictor/diagnostics/` + `shared/model_dashboard/`) used to debug any PA-grain model, not just this one. `BENCHMARKS.md` documents what "beating baseline" means here (naive-floor comparison vs. an economic/CLV-aware bar) and synthesizes external research on per-PA hit prediction ceilings and betting profitability — read before concluding a flat log_loss delta means the pipeline is broken, or before building the devig/CLV eval layer. `ROADMAP.md` (repo root) is the living project plan — current priorities, what's deferred and why — read it first when picking work back up. `DECISIONS.md` (repo root) is the dated, append-only history of every session's findings and corrections that used to live inline in `ROADMAP.md`'s own changelog.

**Point-in-time safety** here follows the same rule as Layer 3 above, enforced more granularly: season-level stats shift forward one season (`_shift_to_last_season`), rolling stats exclude the current row via `.shift(1)`, and pitcher role/TTO features are gated on pre-game-*estimable* signals (lineup position, starter history) rather than the realized in-game role, which is only knowable after the fact.

---

## Processing Layer

`src/data/modules/preprocessing.py` defines output column schemas as module-level constants (`BATTER_BOXSCORE_COLUMNS`, `PITCHER_BOXSCORE_COLUMNS`, etc.) — these are the single source of truth for downstream consumers. When adding columns, update the constant and the function.

Two-stage processing in `daily_process_data`:
1. **Process** — clean each raw table independently (`process_schedule`, `process_batter_boxscore`, etc.)
2. **Prepare** — join processed tables together, add player names and game context (`prepare_batter_boxscore`, `prepare_pitcher_boxscore`, `prepare_playbyplay`)

**This is also the promotion target for feature logic that's outgrown one model's directory.** `create_batting_order` moved here 2026-09-15 from `hit_predictor/processing/pipeline.py` (as a private `_create_batting_order`) once it was clear it had become generic, cross-model infrastructure — 4+ models plus the `batter_lineup` feature-store transform all depended on it, none of them hit_predictor-specific in what they needed from it. `hit_predictor/processing/pipeline.py` re-exports it as `_create_batting_order` so older call sites (including one-off scripts under `k_predictor/experiments/` and `k_predictor/backtest/`, left untouched since they're frozen analyses, not live code) keep working — new code should import `create_batting_order` from here directly. **The pattern going forward**: feature logic starts inside whichever model's `processing/features/` needed it first; if it turns out to be generic enough that a second model (or the feature-store transform layer) needs the identical computation, promote it here rather than having one side import the other's internals — this is the same "define once, neutral peers on both sides" principle behind the Feature Engineering section's transforms-layer rule below, one layer down.

---

## Streamlit App

`app/` is a separate Streamlit app for prop research. Not part of the Lambda pipeline.

```bash
streamlit run app/app.py
```

Pages: Slate (today's games), Player (historical distributions), Edge (model rate vs. book implied probability). Data is loaded via `app/data/loaders.py` and `app/data/transformers.py`.

---

## Infrastructure

Lambdas are provisioned via Terraform in `terraform/`. Shared dependencies (MLB-StatsAPI, requests) are packaged as a Lambda layer (`make build-layer`, `make publish-layer`). Deploy individual Lambdas with `make deploy-*`.

**Terraform resources for `daily_odds_fetch`** (all in `terraform/main.tf`, confirmed applied and live as of 2026-08-26 — `terraform plan` reports no drift):
- `aws_lambda_function.odds_fetch` — 300s timeout, 512 MB, handler `handler.handler`
- `aws_iam_role_policy.lambda_ssm_odds` — GetParameter + PutParameter on `/mlb/odds-api/*`
- `aws_sqs_queue.odds_dlq` + `aws_iam_role_policy.lambda_sqs_dlq` — DLQ for failed invocations
- `aws_sns_topic.lambda_alerts` + `aws_sns_topic_subscription.email` → `thealconomist@gmail.com` (subscription confirmed)
- `aws_cloudwatch_metric_alarm.odds_errors` — fires on Errors ≥ 1
- `aws_cloudwatch_metric_alarm.odds_zero_invocations` — fires if Invocations < 1 in last 24h, `treat_missing_data = breaching`

These alarms and the DLQ key off Lambda's native unhandled-exception accounting (the `AWS/Lambda` `Errors` metric and async-invocation DLQ routing both require an actual raised exception, not a caught-and-returned error body) — see the Known Issues entry below for a bug that silently defeated this for 9+ days.

---

## Test Coverage

| Test file | What it covers |
|---|---|
| `tests/test_status_writer.py` | `status_writer.py` unit tests — S3 key, payload schema, error field, games_processed dict, bucket, content-type |
| `tests/test_daily_mlb_fetch_handler.py` | `daily_mlb_fetch` handler unit tests — write_status called on success/failure, games_processed shape |
| `tests/test_daily_process_data_handler.py` | `daily_process_data` handler unit tests — write_status called on success/failure, unique gamepk counts per prepared table |
| `tests/test_odds_fetch.py` | `odds_fetch.py` unit tests — API routing, retry, contract shape |
| `tests/test_odds_quota.py` | `odds_quota.py` unit tests — SSM get/put, ParameterNotFound, warn/stop thresholds |
| `tests/test_daily_odds_fetch_handler.py` | `daily_odds_fetch` handler unit tests — event parsing, quota gate, fetch+store, S3 paths, 200/500 responses, write_status on success/failure |
| `tests/test_integration_odds_fetch.py` | End-to-end integration using moto (S3 + SSM mocked, HTTP mocked) — verifies both Parquet files land in S3 and quota counter increments |

Integration test approach: moto mocks AWS (S3 + SSM), `unittest.mock.patch` mocks `requests.get`. No real API or AWS calls. Smoke test against real infra is manual (see odds pipeline smoke test command above).

**Test isolation for Lambda handlers:** all three Lambda `handler.py` files share the same module name, which causes `sys.modules` collisions when running the full suite. `tests/conftest.py` only adds `daily_odds_fetch` to `sys.path` (backward compat). The new handler test files (`test_daily_mlb_fetch_handler.py`, `test_daily_process_data_handler.py`) each use an `autouse` pytest fixture that inserts their Lambda directory at `sys.path[0]` and clears `sys.modules["handler"]` before and after every test. Follow this pattern for any new Lambda handler test files.

---

## Known Issues / Deferred Work

- **[fixed 2026-09-15]** `src/features/.feastignore` was a complete no-op since it was created — the tracked file was literally named `.feastignore ` (trailing space, so Feast's CLI never found it) and its content was `.transforms/` (stray leading dot, wouldn't have matched the real `transforms/` directory even if found). Never caught because `feast apply` had never been run before 2026-09-15 — running it for the first time made Feast's repo scanner walk into `transforms/` and fail importing `batter_lineup.py`'s sibling-style imports. Fixed: renamed to `.feastignore`, content is `transforms/`.
- **`feast materialize-incremental` vs `feast materialize`** — for a *first-time* historical load into a `FeatureView` with no prior materialization, use `feast materialize <start_ts> <end_ts>` (explicit range), not `materialize-incremental`. `feast apply` sets an initial watermark near real wall-clock apply time, so `materialize-incremental <end_date>` computes an inverted/empty range (and silently succeeds with zero rows written) if `end_date` is earlier than that watermark — which it always will be for historical data. `materialize-incremental` is correct once a real prior watermark exists — which is exactly what `daily_feature_materialize`'s real, event-triggered runs do now (step 3 of the Production plan, done 2026-09-17/18). See `DECISIONS.md`'s 2026-09-15 entry, and its 2026-09-17 entry for why `event_timestamp` had to be fixed to reflect the real confirmation moment (not the calendar date) before running this multiple times a day would have worked correctly.
- **`feast`, `aiobotocore`, `s3fs` were never in `requirements.txt`** despite being used in this repo — fixed 2026-09-15 alongside a genuine `pyarrow` version conflict between `feast[aws]==0.62.0` (needs `>=21.0.0`) and `awswrangler==3.12.1` (states `<21.0.0`); resolved at `pyarrow==25.0.1` after confirming awswrangler's actual S3 reads still work at that version (its stated ceiling is more conservative than necessary) — re-verify before bumping either package further.
- **[fixed 2026-09-15]** Importing `feast` and later training an XGBoost model in the same process segfaults (native OpenMP conflict, not a Python bug) — reproduced on a plain `pytest tests/` run once `tests/features/test_feature.py` started importing `feast` for the FTI feature-store work, since it's collected alphabetically before `tests/n_pa_predictor/`'s XGBoost tests. Fixed by setting `OMP_NUM_THREADS=1` in the repo root `conftest.py` (runs before any test module import). See `feedback_pytorch_openmp_macos_deadlock.md` memory and `DECISIONS.md`'s 2026-09-15 entry — this is the same class of native-library conflict as the earlier documented torch/pyarrow deadlock, different library pair, different symptom (crash vs. hang).
- **[fixed 2026-09-05]** `daily_odds_fetch`'s `QUOTA_LIMIT` constant was left at the old free-tier value of 500 after the 2026-08-27 upgrade to the 20K plan (see the 2026-08-27 entry below), so the Lambda hard-stopped for real once usage crossed 500 on 2026-09-02 — it ran successfully 2026-08-27 through 2026-09-01, then failed every day from 2026-09-02 through 2026-09-05 with `"Monthly quota reached (1369/500)"` despite the account having ~18,600 credits of real headroom left. This is what made the dashboard's odds column and starting-pitcher predictions look stale — `raw_data/odds/team_odds` and `player_props` simply stopped writing. Fixed in code (`QUOTA_LIMIT = 20000`, TDD'd via `tests/test_daily_odds_fetch_handler.py::TestQuotaLimitConfig`), deployed via `make deploy-odds-fetch`, and confirmed live with a manual invoke for 2026-09-05 (`team_odds: 12, player_props: 12`, `status: success`). No backfill needed — the gap is only 2026-09-02 through 2026-09-04.
- **Status writer IAM** — Lambda execution roles need `s3:PutObject` on `arn:aws:s3:::mlbdk/lambdas/status/*`; not yet added to Terraform
- **`daily_feature_create` status writer** — not yet integrated; needs the same try/except pattern as the other three Lambdas; `games_processed` keys TBD once the handler bugs are fixed
- **[fixed 2026-08-26]** `daily_odds_fetch`'s handler used to catch all exceptions and return `{"statusCode": 500, ...}` instead of re-raising, which meant the DLQ and `odds_errors` CloudWatch alarm (both wired to Lambda's native `Errors` metric / unhandled-exception accounting) never fired despite the Lambda failing daily for 9+ days — both alarms sat at `OK` and the DLQ stayed empty the whole time. Handler now re-raises after `write_status`, matching the other three Lambdas' pattern.
- **[resolved 2026-08-27]** The Odds API free tier (500 credits/month) couldn't sustain this pipeline's real usage — `daily_odds_fetch` bills `markets × regions` per call, and `get_all_player_props`'s 4 markets × ~13-15 games/day (~55-60 credits/day) exhausted the monthly cap by day ~8-9, which is what produced the 2026-08-16 → 08-27 outage above. Fixed by upgrading to The Odds API's 20K plan ($30/mo, 20,000 credits — ~10x current full-usage headroom) and rotating `/mlb/odds-api/api-key` in SSM to the new key. The already-written quota-tracking fix (`set_monthly_usage` reading the real `x-requests-used` header) is now deployed to the live Lambda. **The 2026-08-16 → 08-27 data gap will not be backfilled** — see `ROADMAP.md`'s "Parked / explicitly deferred decisions" for the cost/feasibility finding (real backfill requires the separate historical-odds endpoint at 10x cost, ~7,500 credits one-time).
- **[stale note removed 2026-09-15]** The `_yesterday()` parenthesis bug and the `datetime.timezone.utc` import bug previously listed here are already fixed in the current handler (`from datetime import datetime, timedelta, timezone` + correctly-parenthesized `datetime.now(timezone.utc) - timedelta(days=1)`) — this list just hadn't been updated. `tests/test_daily_feature_create_handler.py` exists and passes (3 tests, mocks `team_batter_base`/`player_batter_base`/`sp_features`/`bullpen_features`/`feast`/`boto3`/`pandas` at the `sys.modules` level, so it never depended on those modules actually existing).
- **[fixed 2026-09-16 through 2026-09-18]** `daily_feature_create` handler used to import feature transform modules that no longer existed (`team_batter_base`, `player_batter_base`, `sp_features`, `bullpen_features` — the last two were already-broken dangling references before the 2026-09-15 deletion). Rewritten to call `compute_batter_lineup_features_for_date`/`compute_batter_pa_volume_features_for_date` directly, and `make zip-feature-create` produces a verified, self-contained zip for its nested-package dependency closure. The `feast`/`awswrangler`-runtime-layer gap flagged the next day is resolved: `feast` was removed from this Lambda entirely and moved to a new container-image Lambda, `daily_feature_materialize` (`feast[aws]`'s ~470MB dependency closure never fit Lambda's 250MB zip+layers cap in the first place), and `daily_feature_create` now attaches the `AWSSDKPandas-Python311` layer, which bundles `awswrangler` together with `pandas`. Both Lambdas are Terraform-managed, deployed, and triggered by a real S3 event notification on `daily_lineup_fetch`'s per-game writes (not a fixed schedule — see the Lambda Pipeline table above). Confirmed live end to end via a real smoke test (fake lineup write → auto-invoke → both transforms compute → auto-invoke `daily_feature_materialize` → real DynamoDB row-count delta confirmed) — see `DECISIONS.md`'s 2026-09-17 and 2026-09-18 entries.
- **[fixed 2026-09-18]** `read_playbyplay` (`src/features/transforms/data_readers.py`) read all 70 columns of the prepared playbyplay table — mostly wide Statcast per-pitch floats — for a whole season, when `batter_pa_volume.py`'s computation only ever used 4 (`gamepk`, `play_id`, `batter_id`, `play_result`). This OOM'd `daily_feature_create` twice in a row in production (512MB then 2048MB, maxed out exactly at the ceiling both times — Lambda's memory-scales-CPU behavior masked the real issue as looking like a sizing problem). Fixed with an optional `columns` parameter on `read_playbyplay` (defaults to `None`/all columns) and a `PBP_COLUMNS_NEEDED` constant in `batter_pa_volume.py` — measured ~10x memory reduction against real S3 data (one file: 7.26MB → 0.79MB; full season: ~1.26GB → 118MB). `memory_size`/`timeout` were also bumped (2048MB/900s) as a safety margin, though the column fix did most of the real work (post-fix peak: 536MB).
- `test_fetch_layer.py` requires `--date YYYY-MM-DD` flag — these are integration tests that hit real S3 data
