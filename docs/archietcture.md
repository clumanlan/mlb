# MLB ML System — Architecture Document

**Last Updated:** 2026-03-21
**Author:** Solo ML Engineer
**Status:** Living document — update as each layer is built

---

## Overview

An end-to-end MLB machine learning system that ingests daily game data, engineers features, and serves predictions from a hierarchy of models — culminating in an attention-based model that predicts plate appearance outcomes and rolls them up to player game lines and game-level predictions.

**Primary prediction target:** Plate appearance (PA) level outcomes (single, double, HR, K, BB, out, etc.), rolled up to player game lines, rolled up to game outcome predictions.

**Inference pattern:** Batch-first, designed for real-time inference later. Predictions update as inputs change (lineup confirmed, injury scratch) — not intraday model retraining.

---

## System Layers

```
Layer 1 — Raw Ingestion       (fetch from MLB Stats API → S3)          ✅ complete
Layer 2 — Validation          (confirm data landed correctly)          ✅ complete
Layer 3 — Feature Engineering (compute features, point-in-time safe)   ⏳ in progress
Layer 4 — Feature Store       (offline + online store)                 ⬜ not started
Layer 5 — Model Training      (baseline models + attention model)      ⬜ not started
Layer 6 — Prediction Pipeline (daily inference, prediction storage)    ⬜ not started
Layer 7 — MLOps               (monitoring, drift detection, retraining)⬜ not started
```

---

## Layer 1 — Raw Ingestion ✅

### S3 Bucket Structure

```
s3://mlbdk/
  raw_data/
    games/
      schedule/
        {year}/
          {YYYY-MM-DD}.parquet
      game_info/
        {year}/
          {YYYY-MM-DD}.parquet
      batter_boxscore/
        {year}/
          {YYYY-MM-DD}.parquet
      pitcher_boxscore/
        {year}/
          {YYYY-MM-DD}.parquet
    playbyplay/
      {year}/
        {YYYY-MM-DD}.parquet
    reference/
      player_info/
        player_info.parquet        ← slowly changing, refresh periodically
    draftkings/
      {year}/
        {YYYY-MM-DD}.parquet
  processed_data/
    games/
      schedule/{year}/{YYYY-MM-DD}.parquet
      game_info/{year}/{YYYY-MM-DD}.parquet
      batter_boxscore/{year}/{YYYY-MM-DD}.parquet
      pitcher_boxscore/{year}/{YYYY-MM-DD}.parquet
    playbyplay/{year}/{YYYY-MM-DD}.parquet
  features/
    offline/
      batter_features/{year}/{YYYY-MM-DD}.parquet
      pitcher_features/{year}/{YYYY-MM-DD}.parquet
  models/
    predictions/
      {model_name}/
        {year}/
          {YYYY-MM-DD}.parquet
  lambda-functions/
  lambda-layers/
  model_artifacts/
```

### Conventions

- One file per table per date — never overwrite a past date's file
- Partitioned by `year/` folder at the top level for each table
- `reference/` tables (player_info) are not date-partitioned — refresh on demand
- All files stored as Parquet

### Migration Task (One-time)

`schedule` is the only table that still has a monolithic historical file. All other tables already follow the date-partitioned pattern.

**Backfill script:** `scripts/migrations/backfill_schedule.py`
Split historical schedule parquet by `game_date`, write each date to `raw_data/games/schedule/{year}/{YYYY-MM-DD}.parquet`, then delete the historical file.

### Data Sources

| Table | Source | Granularity | Key Columns |
|---|---|---|---|
| schedule | MLB Stats API | 1 row per game | `game_date`, `gamePk`, `home_id`, `away_id` |
| game_info | MLB Stats API | 1 row per game | `gamePk`, `weather_*`, `probable_pitcher_*` |
| batter_boxscore | MLB Stats API | 1 row per batter per game | `personId`, `gamePk`, `game_date`, `ab`, `h`, `hr` |
| pitcher_boxscore | MLB Stats API | 1 row per pitcher per game | `personId`, `gamePk`, `ip`, `k`, `era` |
| playbyplay | MLB Stats API | 1 row per pitch | `batter_id`, `pitcher_id`, `gamePk`, `game_date`, `pitch_type`, `play_result` |
| player_info | MLB Stats API | 1 row per player | `personId`, `fullName`, `primaryPosition` |
| draftkings | DraftKings | 1 row per player per slate | `player_name`, `salary`, `game_date` |

### Fetch Pipeline Code

**File:** `pipelines/daily_mlb_fetch/fetch.py`

Functions:
- `fetch_schedule_df(game_date)` → schedule rows, gamePks
- `fetch_game_info(gamepks, game_date)` → weather, probable pitchers
- `fetch_batter_pitcher_boxscore(gamepks, game_date)` → batter + pitcher rows
- `fetch_playbyplay_data(gamepks)` → pitch-level rows
- `send_metrics_to_cloudwatch(results, date)` → CloudWatch instrumentation

All functions return `(DataFrame, error_list)` — partial failures are tracked via `gamepk_errors` and do not crash the pipeline.

---

## Layer 2 — Validation ✅

### Purpose

A quality gate that runs immediately after ingestion. Confirms that data landed correctly before any downstream process runs. Does not transform data — only assertions are made.

If validation fails, an alert fires and the pipeline stops. Nothing downstream runs on bad data.

### Validation Checks Per Table

**All tables:**
- File exists in expected S3 path
- Row count > 0
- Row count within expected range (flag if suspiciously low)
- No null values in key ID columns (`personId`, `gamePk`, `batter_id`, `pitcher_id`)
- `game_date` column present and matches expected date

**schedule-specific:**
- At least 1 game found (unless known off-day)
- `gamePk` values present (used as input to all downstream fetches)

**playbyplay-specific:**
- `pitch_type` null rate below threshold
- `batter_id` and `pitcher_id` non-null on all rows
- `game_date` populated on all rows

### gamePk Completeness Check

After the prepare step, every `gamePk` from the schedule must be present in `batter_prepared`, `pitcher_prepared`, and `playbyplay_prepared`. Missing gamePks are logged and included in the results dict but do not hard-fail the pipeline — partial failures (postponed games, suspended games) are expected and should be visible, not blocking.

Failed gamePks written to `raw_data/errors/{YYYY-MM-DD}_failed_gamepks.json` for retry.

### Validation Tool

Pandera for schema + type validation. Custom checks for row count thresholds and null rates.

### Validation Code

**File:** `pipelines/daily_process_data/lambda_handler.py`

---

## Orchestration

### Two Independent Pipelines

**Pipeline 1 — MLB Data Pipeline**
Trigger: EventBridge, runs after games complete (1:00 AM ET)

```
EventBridge
  → Step Functions state machine (mlbdk-daily-pipeline)
      → daily_mlb_fetch
      → daily_process_data
      → daily_feature_computation    ← adding next
```

**Pipeline 2 — DraftKings Slate Pipeline**
Trigger: EventBridge, runs ~2 hours before first pitch

```
EventBridge
  → daily_dkslate_fetch
  → dk_validation
  → [FUTURE: trigger prediction pipeline]
```

### Orchestration Tool

AWS Step Functions — native to AWS, no server to manage, integrates directly with Lambda. EventBridge triggers the state machine; each Lambda passes output to the next state. If a step fails, Step Functions stops execution before downstream steps run.

### Lambda Inventory

| Lambda | Pipeline | Status |
|---|---|---|
| `daily_mlb_fetch` | MLB Data | ✅ deployed |
| `daily_process_data` | MLB Data | ✅ deployed |
| `daily_feature_computation` | MLB Data | ⏳ in progress |
| `daily_dkslate_fetch` | DK Slate | ✅ deployed |
| DK processing | DK Slate | ⏳ partially built |

---

## Layer 3 — Feature Engineering ⏳

### Status

In progress — building v1 batter and pitcher features.

### Pipeline Position

Runs as `daily_feature_computation` Lambda, third step in the MLB Data state machine:

```
daily_mlb_fetch → daily_process_data → daily_feature_computation
```

### Data Model

Every feature table has exactly two keys plus computed features. No other structure.

| Column | Role |
|---|---|
| `player_id` | entity key — who |
| `game_date` | time key — when |
| `split` | `overall` for now, handedness splits later |
| everything else | computed rolling features |

The entity does not have to be a player — same pattern applies to team-level and game-level features:

- `team_id + game_date` → team rolling stats, bullpen usage, rest days
- `gamePk + game_date` → park factors, weather, game context

### Phase 1 — Assemble

Join processed tables before computing features:

```
processed_data/games/batter_boxscore + processed_data/games/game_info
  → joined on gamePk
  → enriched with park, weather_temp, weather_wind, home/away
```

Processing layer (`daily_process_data`) handles reference data enrichment (player_info, schedule). Feature computation layer handles game context enrichment (game_info) and rolling aggregates.

### Phase 2 — Compute

DuckDB reads parquet directly from S3, SQL window functions for rolling aggregates, aws-wrangler for final write.

```sql
SELECT
  player_id,
  game_date,
  'overall' as split,
  SUM(h) OVER (
    PARTITION BY player_id
    ORDER BY game_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) as rolling_7_h,
  SUM(hr) OVER (
    PARTITION BY player_id
    ORDER BY game_date
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
  ) as rolling_30_hr
FROM enriched_batter
```

### Feature Registry

Features defined as config objects, separate from execution logic. Adding a new feature = adding one line to the registry. Job execution code never changes.

```python
BATTER_FEATURES = [
    RollingFeature(col='h',   windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='hr',  windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='rbi', windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='bb',  windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='k',   windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='avg', windows=[7, 14, 30], agg='mean'),
    RollingFeature(col='obp', windows=[7, 14, 30], agg='mean'),
    RollingFeature(col='slg', windows=[7, 14, 30], agg='mean'),
]

PITCHER_FEATURES = [
    RollingFeature(col='ip',   windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='k',    windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='bb',   windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='hr',   windows=[7, 14, 30], agg='sum'),
    RollingFeature(col='era',  windows=[7, 14, 30], agg='mean'),
    RollingFeature(col='whip', windows=[7, 14, 30], agg='mean'),
]
```

### S3 Output

```
features/offline/batter_features/{year}/{YYYY-MM-DD}.parquet
features/offline/pitcher_features/{year}/{YYYY-MM-DD}.parquet
```

One row per player per game date. Only written when player appears in a game — no backfill on rest days. At training/inference time, use a point-in-time lookup: most recent feature row before game date D.

### Point-in-Time Correctness

Features for a game on date D must only use data from games completed before date D. The rolling window closes the day before the target game. This prevents leakage — a model trained with same-day stats will look great in backtesting and fail in production.

### Tech Stack

- DuckDB — S3 parquet reads + window function computation
- aws-wrangler — final write to S3
- pandas — light orchestration only

### Splits (Future)

Add `split` column rows per player per date — same table, no schema change required:

- `overall` — already implemented
- `vs_rhp` / `vs_lhp` — filter PBP by pitcher hand, recompute
- `home` / `away` — filter by game context, recompute

### Open Questions

- Confirm DuckDB S3 integration works within Lambda memory limits on 30 days of batter_boxscore
- game_context feature table (gamePk + park + weather) — add in next iteration
- Verify `min_periods` behavior for players with fewer than 7/14/30 games in window

---

## Future State (Out of Scope Now)

### Layer 4 — Feature Store

**Offline store:** S3 + Athena (training, backtesting)
**Online store:** DynamoDB (real-time inference, flat features) + player-indexed Parquet (sequence retrieval)
**Tool:** Feast (open source, AWS-native, free)

### Layer 5 — Model Architecture

**Hierarchy:**
1. PA-level attention model → predicts distribution of PA outcomes per batter-pitcher matchup
2. Roll up PA predictions → expected player game line (hits, HR, K, BB, etc.)
3. Roll up player lines → game outcome prediction

**Baseline models** (built first, used for benchmarking + ensemble):
- XGBoost / LightGBM on flat game-level features
- Separate models per target (win/loss, player props)

**Attention model:**
- Player and pitcher embeddings (learned, updated periodically — not daily)
- Attends over recent PA sequences for batter + pitcher
- Dynamic context at inference: confirmed lineup, opposing pitcher, park, weather
- Papers: MIT Sloan Analytics Conference attention-based baseball model(s)

**Retraining cadence:** Periodic (weekly, mid-season, end of season) — not daily. Embeddings are stable between retrains. Only inference features refresh daily.

### Layer 6 — Prediction Pipeline

Daily inference triggered after validation passes + lineup confirmation:

```
lineup confirmed (~2hrs before first pitch)
  → pull flat features from online store
  → pull sequence features for today's batters + pitchers
  → run PA-level attention model
  → Monte Carlo simulation over lineup order → player game line distributions
  → roll up to game outcome predictions
  → write to models/predictions/{model_name}/{date}/
```

### Layer 7 — MLOps

- **Experiment tracking:** MLflow
- **Model registry:** MLflow Model Registry — Staging → Production promotion
- **Monitoring:** Evidently AI — data drift, feature drift, prediction distribution shift
- **Dashboards:** CloudWatch (pipeline health) + Grafana (model performance)
- **Alerts:** CloudWatch Alarms → SNS → Slack
- **Prediction tracking:** Compare predictions vs actuals daily, track calibration over time

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Orchestration | Step Functions | AWS-native, no server, handles dependency chain cleanly |
| Validation tool | Pandera | Declarative schemas, lightweight, easy to update |
| Storage format | Parquet | Columnar, Athena-compatible, efficient for analytics |
| Partition strategy | `year/YYYY-MM-DD.parquet` | Supports both date-range queries and single-day lookups |
| Feature computation | DuckDB | Reads parquet from S3 directly, native window functions, fast |
| Feature registry | Config objects separate from execution | Adding features = one line, job code never changes |
| Feature table keys | entity_id + game_date + split | Matches industry standard (Feast, Databricks, SageMaker) |
| Splits | `overall` only for now, same table later | No schema change needed when splits are added |
| Feature store | Feast (future) | Open source, S3+DynamoDB backend, enforces point-in-time correctness |
| Sequence storage | Player-indexed Parquet (future) | Flat feature stores not designed for sequence retrieval |
| ML framework | XGBoost baseline → Attention model | Build interpretable baseline first, then graduate to attention |
| Retraining | Periodic, not daily | Embeddings stable; only inference context refreshes daily |

---

## Notes + Open Questions

- `get_schedule()` in `GetData` class still hardcoded to `historical/` path — update after backfill
- `GetData` reads full seasons into pandas memory — fine for exploration, will need Athena for feature computation at scale
- DK slate processing script scope TBD — formatting for model input or something else?
- Confirm `game_date` is always populated in PBP rows (parsed from `about.startTime`)
- Two papers on attention-based baseball models (one featured at MIT Sloan) — retrieve and review before designing attention model architecture
- `player_info` in `daily_process_data` reads from hardcoded production path even in test mode — intentional but worth a comment