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

**Integration pattern:** capture `started_at` before the try block, then use `try/except`. On success call `write_status` before returning; on failure call `write_status` in the except block then re-raise (or return 500 for `daily_odds_fetch`). Import as a sibling module (`from status_writer import write_status`).

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
    raise  # or return {"statusCode": 500, ...}
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
- `odds_fetch.py` — API client: `get_games`, `get_team_odds`, `get_player_props`, `get_all_player_props`
- `odds_quota.py` — SSM quota helpers: `get_monthly_usage`, `check_quota`, `increment_monthly_usage`

**S3 output:**
```
raw_data/odds/team_odds/{year}/{date}.parquet
raw_data/odds/player_props/{year}/{date}.parquet
```

**SSM parameters:**
- `/mlb/odds-api/api-key` — SecureString, API key for api.the-odds-api.com
- `/mlb/odds-api/requests-used/{year}/{month}` — running monthly request count (limit: 500)

**Quota behavior:** warns at 90% (450 calls), hard-stops at 100% (500 calls), returns 500 with "quota" in body.

**Smoke test:**
```bash
aws lambda invoke \
  --function-name daily_odds_fetch \
  --region us-east-2 \
  --payload '{"date": "YYYY-MM-DD"}' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json
```

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

The end goal is a shared feature store feeding **multiple models** (batter hit/no-hit, pitcher K/no-K, and others as they're built). `hit_predictor` (`src/models/hit_predictor/`) was the first model and is still where the batter-side hit-probability problem is worked on. `src/models/k_predictor/` (pitcher strikeout probability), `src/models/n_pa_predictor/` (batter plate-appearance count / `low_pa` classifier), `src/models/bb_predictor/` (pitcher walk probability), and `src/models/short_outing_predictor/` (starting-pitcher short-outing probability — the last candidate in README's sub-problem menu) are newer sibling models — see `README.md`'s sub-problem menu and `ROADMAP.md`'s Mid-term section for current status of each. All four compose `hit_predictor`'s existing target-agnostic feature-building machinery (season/rolling stats, role gating) via their own thin `processing/pipeline.py` + `processing/schema.py`, rather than duplicating it. `short_outing_predictor` is the first to run at a different grain than the rest — one row per starting-pitcher-*start*, not per PA or batter-game — reusing `game_context.py`'s `build_expected_start_innings` (a pre-game workload-shrinkage estimate originally built for `n_pa_predictor`'s opposing-starter feature) for the pitcher's own start instead. Each still runs its own bespoke processing pipeline rather than consuming from the Layer 4 Feast store above, because the feature engineering it needs (PA-grain, point-in-time-safe, extensively feature-engineered) is still being worked out model-side before it's worth generalizing into the shared store. Expect this pipeline's proven patterns (rolling windows, point-in-time shifting, role-aware pitcher splits) to migrate toward Layer 4 now that a second model (`k_predictor`) exists and needs the same features — see `ROADMAP.md`'s "Feature-store convergence" item.

**Structure:**
- `processing/pipeline.py` — assembles the PA-outcome training grain from raw pbp/boxscore/schedule/game_info (`create_pa_outcome`)
- `processing/features/` — one file per feature family: `season_stats.py` (season-level batter/pitcher rates), `rolling_stats.py` (game-by-game rolling windows), `park_factors.py`, `expected_role.py` (pre-game-knowable SP/bullpen role gating), `interaction_feats.py` (trend/shrinkage features), `game_context.py` (calendar, team form/rest, probable starters, starter-innings estimate)
- `processing/schema.py` — pbp column schema, the single source of truth this pipeline reads against
- `baseline/` — naive and rules-based baselines, run before any real model (`baseline/rules/run_baseline.py`, `baseline/model/run.py`)
- `experiments/v{N}_*/train.py` — versioned experiments, each an isolated, runnable pipeline (own `mlruns/` for MLflow tracking); read the newest version's summary comment block first — it documents exactly what changed vs. the prior version
- `utils/` — `eval.py` (metrics + calibration plots), `mlflow_logging.py`, `model_prep.py` (missing-indicator/impute/encode helpers)
- `config.yaml` — seasons (train/val/test split), target column, drop columns

**Reference docs:** `FEATURE_GLOSSARY.md` is the feature-by-feature reference (what each stat measures, why, and whether it's implemented) — check it before adding a new feature or wondering if one already exists. `dashboard_spec.md` documents the portable Streamlit model-diagnostics dashboard (implemented at top-level `batter_pa_model/` + `shared/model_dashboard/`) used to debug any PA-grain model, not just this one. `BENCHMARKS.md` documents what "beating baseline" means here (naive-floor comparison vs. an economic/CLV-aware bar) and synthesizes external research on per-PA hit prediction ceilings and betting profitability — read before concluding a flat log_loss delta means the pipeline is broken, or before building the devig/CLV eval layer. `ROADMAP.md` (repo root) is the living project plan — current priorities, decision log, what's deferred and why — read it first when picking work back up.

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

**Terraform resources for `daily_odds_fetch`** (all in `terraform/main.tf`):
- `aws_lambda_function.odds_fetch` — 300s timeout, 512 MB, handler `handler.handler`
- `aws_iam_role_policy.lambda_ssm_odds` — GetParameter + PutParameter on `/mlb/odds-api/*`
- `aws_sqs_queue.odds_dlq` + `aws_iam_role_policy.lambda_sqs_dlq` — DLQ for failed invocations (**pending `terraform apply`** — blocked by dk-user IAM permissions)
- `aws_sns_topic.lambda_alerts` + `aws_sns_topic_subscription.email` → `thealconomist@gmail.com` (**pending `terraform apply`**)
- `aws_cloudwatch_metric_alarm.odds_errors` — fires on Errors ≥ 1 (**pending `terraform apply`**)
- `aws_cloudwatch_metric_alarm.odds_zero_invocations` — fires if Invocations < 1 in last 24h, `treat_missing_data = breaching` (**pending `terraform apply`**)

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

- **Status writer IAM** — Lambda execution roles need `s3:PutObject` on `arn:aws:s3:::mlbdk/lambdas/status/*`; not yet added to Terraform
- **`daily_feature_create` status writer** — not yet integrated; needs the same try/except pattern as the other three Lambdas; `games_processed` keys TBD once the handler bugs are fixed
- `daily_odds_fetch` DLQ + CloudWatch alarms are written in Terraform but **not yet applied** — `terraform apply` blocked by dk-user IAM permissions (needs `iam:CreatePolicy`, `iam:AttachRolePolicy`, `sqs:*`, `sns:*`, `cloudwatch:*`)
- `daily_feature_create` handler has a bug in `_yesterday()` — `timedelta` parenthesis wrapping is incorrect
- `daily_feature_create` handler references `datetime.timezone.utc` as `datetime.now(datetime.timezone.utc)` — needs `from datetime import timezone`
- Feature transform modules (`team_batter_base`, `player_batter_base`, etc.) are not bundled in `daily_feature_create` zip — `make zip-feature-create` does not exist yet
- Tests for `daily_feature_create` handler were deleted (stale); backfilling is deferred
- `test_fetch_layer.py` requires `--date YYYY-MM-DD` flag — these are integration tests that hit real S3 data
