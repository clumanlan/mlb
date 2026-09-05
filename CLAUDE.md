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
features/offline/{batter_features,pitcher_features}/{year}/{date}.parquet
feast/features/{team_batter_base,player_batter_base,starting_pitcher_base,bullpen_pitcher_base}/{year}/{date}.parquet
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
| `daily_feature_create` | `src/lambdas/daily_feature_create/handler.py` | Runs all feature transforms, writes snapshots, materializes into Feast |

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

`src/features/transforms/` contains one file per feature group:

- `player_batter_base.py` — per-player rolling stats (7/14/30 day windows) using DuckDB window functions
- `team_batter_base.py` — same rolling windows aggregated to team level
- `starting_pitcher_base.py` — SP rolling stats
- `bullpen_pitcher_base.py` — bullpen rolling stats
- `data_readers.py` — shared S3 readers (`read_batter_boxscore(season)`, etc.) using `awswrangler`

**Pattern:** each transform function takes `year: str`, reads the full season's processed Parquet from S3 via `data_readers.py`, computes rolling windows in DuckDB (registered as in-memory tables), and returns a DataFrame. The handler then slices to a single date snapshot and writes to S3.

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

**Reference docs:** `FEATURE_GLOSSARY.md` is the feature-by-feature reference (what each stat measures, why, and whether it's implemented) — check it before adding a new feature or wondering if one already exists. `dashboard_spec.md` documents the portable Streamlit model-diagnostics dashboard (implemented at top-level `batter_pa_model/` + `shared/model_dashboard/`) used to debug any PA-grain model, not just this one. `BENCHMARKS.md` documents what "beating baseline" means here (naive-floor comparison vs. an economic/CLV-aware bar) and synthesizes external research on per-PA hit prediction ceilings and betting profitability — read before concluding a flat log_loss delta means the pipeline is broken, or before building the devig/CLV eval layer. `ROADMAP.md` (repo root) is the living project plan — current priorities, what's deferred and why — read it first when picking work back up. `DECISIONS.md` (repo root) is the dated, append-only history of every session's findings and corrections that used to live inline in `ROADMAP.md`'s own changelog.

**Point-in-time safety** here follows the same rule as Layer 3 above, enforced more granularly: season-level stats shift forward one season (`_shift_to_last_season`), rolling stats exclude the current row via `.shift(1)`, and pitcher role/TTO features are gated on pre-game-*estimable* signals (lineup position, starter history) rather than the realized in-game role, which is only knowable after the fact.

---

## Processing Layer

`src/data/modules/preprocessing.py` defines output column schemas as module-level constants (`BATTER_BOXSCORE_COLUMNS`, `PITCHER_BOXSCORE_COLUMNS`, etc.) — these are the single source of truth for downstream consumers. When adding columns, update the constant and the function.

Two-stage processing in `daily_process_data`:
1. **Process** — clean each raw table independently (`process_schedule`, `process_batter_boxscore`, etc.)
2. **Prepare** — join processed tables together, add player names and game context (`prepare_batter_boxscore`, `prepare_pitcher_boxscore`, `prepare_playbyplay`)

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

- **[fixed 2026-09-05]** `daily_odds_fetch`'s `QUOTA_LIMIT` constant was left at the old free-tier value of 500 after the 2026-08-27 upgrade to the 20K plan (see the 2026-08-27 entry below), so the Lambda hard-stopped for real once usage crossed 500 on 2026-09-02 — it ran successfully 2026-08-27 through 2026-09-01, then failed every day from 2026-09-02 through 2026-09-05 with `"Monthly quota reached (1369/500)"` despite the account having ~18,600 credits of real headroom left. This is what made the dashboard's odds column and starting-pitcher predictions look stale — `raw_data/odds/team_odds` and `player_props` simply stopped writing. Fixed in code (`QUOTA_LIMIT = 20000`, TDD'd via `tests/test_daily_odds_fetch_handler.py::TestQuotaLimitConfig`), deployed via `make deploy-odds-fetch`, and confirmed live with a manual invoke for 2026-09-05 (`team_odds: 12, player_props: 12`, `status: success`). No backfill needed — the gap is only 2026-09-02 through 2026-09-04.
- **Status writer IAM** — Lambda execution roles need `s3:PutObject` on `arn:aws:s3:::mlbdk/lambdas/status/*`; not yet added to Terraform
- **`daily_feature_create` status writer** — not yet integrated; needs the same try/except pattern as the other three Lambdas; `games_processed` keys TBD once the handler bugs are fixed
- **[fixed 2026-08-26]** `daily_odds_fetch`'s handler used to catch all exceptions and return `{"statusCode": 500, ...}` instead of re-raising, which meant the DLQ and `odds_errors` CloudWatch alarm (both wired to Lambda's native `Errors` metric / unhandled-exception accounting) never fired despite the Lambda failing daily for 9+ days — both alarms sat at `OK` and the DLQ stayed empty the whole time. Handler now re-raises after `write_status`, matching the other three Lambdas' pattern.
- **[resolved 2026-08-27]** The Odds API free tier (500 credits/month) couldn't sustain this pipeline's real usage — `daily_odds_fetch` bills `markets × regions` per call, and `get_all_player_props`'s 4 markets × ~13-15 games/day (~55-60 credits/day) exhausted the monthly cap by day ~8-9, which is what produced the 2026-08-16 → 08-27 outage above. Fixed by upgrading to The Odds API's 20K plan ($30/mo, 20,000 credits — ~10x current full-usage headroom) and rotating `/mlb/odds-api/api-key` in SSM to the new key. The already-written quota-tracking fix (`set_monthly_usage` reading the real `x-requests-used` header) is now deployed to the live Lambda. **The 2026-08-16 → 08-27 data gap will not be backfilled** — see `ROADMAP.md`'s "Parked / explicitly deferred decisions" for the cost/feasibility finding (real backfill requires the separate historical-odds endpoint at 10x cost, ~7,500 credits one-time).
- `daily_feature_create` handler has a bug in `_yesterday()` — `timedelta` parenthesis wrapping is incorrect
- `daily_feature_create` handler references `datetime.timezone.utc` as `datetime.now(datetime.timezone.utc)` — needs `from datetime import timezone`
- Feature transform modules (`team_batter_base`, `player_batter_base`, etc.) are not bundled in `daily_feature_create` zip — `make zip-feature-create` does not exist yet
- Tests for `daily_feature_create` handler were deleted (stale); backfilling is deferred
- `test_fetch_layer.py` requires `--date YYYY-MM-DD` flag — these are integration tests that hit real S3 data
