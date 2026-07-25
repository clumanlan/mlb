# CODEBASE.md

Reference document for the MLB DraftKings ML pipeline. Covers every layer from raw ingestion to feature store, with data contracts, S3 paths, and known quality issues.

---

## Architecture

```
Event (EventBridge)
  └── Step Functions
        ├── daily_mlb_fetch      → raw Parquet to S3
        ├── daily_process_data   → cleaned + joined tables to S3
        └── daily_feature_create → rolling features + Feast materialization

Independent:
  daily_odds_fetch   → odds + player props to S3
  daily_dkslate_fetch → DK contest/player data to S3
```

All data lives in S3 bucket `mlbdk` (us-east-2). Every table is Parquet, partitioned `{table}/{year}/{YYYY-MM-DD}.parquet`. One file per table per date — never overwrite past dates.

Lambda event contract: `{"date": "YYYY-MM-DD", "env": "test"}` — `env: test` routes all reads/writes under `test/` prefix.

---

## Layer 1 — Raw Ingestion

### `src/data/modules/fetchers.py`

Calls MLB StatsAPI and writes raw Parquet to S3.

| Function | Source | S3 Output Path |
|---|---|---|
| `fetch_schedule_df(date)` | `statsapi.schedule()` | `raw_data/games/schedule/{year}/{date}.parquet` |
| `fetch_game_info(gamepks)` | `statsapi.get('game', ...)` | `raw_data/games/game_info/{year}/{date}.parquet` |
| `fetch_batter_pitcher_boxscore(gamepks)` | `statsapi.boxscore_data()` | `raw_data/games/batter_boxscore/{year}/{date}.parquet` and `raw_data/games/pitcher_boxscore/{year}/{date}.parquet` |
| `fetch_playbyplay_data(gamepks)` | `GET /api/v1/game/{gamepk}/playByPlay` | `raw_data/playbyplay/{year}/{date}.parquet` |

**Raw batter boxscore columns:** `personId`, `gamePk`, `team_id`, `ab`, `h`, `r`, `doubles`, `triples`, `hr`, `rbi`, `bb`, `k`, `lob`, `battingOrder`, `name`, `team_name`

**Raw pitcher boxscore columns:** `personId`, `gamePk`, `team_id`, `ip`, `h`, `r`, `er`, `bb`, `k`, `hr`, `p`, `s`, `note`, `name`, `team_name`

**Raw play-by-play columns (~60):** `play_id`, `inning`, `half_inning`, `batter_id`, `pitcher_id`, `event_type`, `is_pitch`, `pitch_type`, `start_speed`, `end_speed`, `plate_x`, `plate_z`, `spin_rate`, `launch_speed`, `launch_angle`, `total_distance`, `gamepk`, `game_date` + all Statcast coordinates and break metrics

**Quality notes:**
- `pd.to_numeric(errors='coerce').fillna(0)` on `venue_id` (line 75) — non-numeric IDs silently become 0
- ID coercion chain `str(int(float(y)))` for `gamePk` (line 481 in preprocessing) — fails silently on null/non-numeric values
- `except Exception` on lines 143 and 706 is intentional error-collection (log + continue per gamepk); leave as-is

---

### `src/data/modules/odds_fetch.py`

Calls The Odds API (DraftKings book).

| Function | Output |
|---|---|
| `get_games(date, api_key)` | `[{id, away_team, home_team, commence_time}]` |
| `get_team_odds(date, api_key)` | Raw API response — list of game dicts with nested `bookmakers` |
| `get_player_props(event_id, api_key)` | Raw props for markets: `pitcher_strikeouts`, `batter_hits`, `batter_home_runs`, `pitcher_hits_allowed` |
| `get_all_player_props(date, api_key)` | `[{game: {...}, props: {...}}]` — one entry per game |

S3 output (written by `daily_odds_fetch` Lambda):
- `raw_data/odds/team_odds/{year}/{date}.parquet`
- `raw_data/odds/player_props/{year}/{date}.parquet`

---

### `src/data/modules/daily_dkslate_fetch.py`

Fetches DraftKings contest slate.

| Output Table | S3 Path | Key Columns |
|---|---|---|
| contests | `raw_data/draftkings/contests/{year}/{timestamp}.parquet` | `contest_id`, `draft_group_id`, `name`, `entry_fee`, `max_entries`, `payout`, `contest_type` |
| games | `raw_data/draftkings/games/{year}/{timestamp}.parquet` | `draft_group_id`, `competition_id`, `away_team_id`, `home_team_id`, `starts_at`, `are_starting_lineups_available` |
| contest_players | `raw_data/draftkings/contest_players/{year}/{timestamp}.parquet` | `player_id`, `salary`, `position_name`, `display_name`, `team_abbreviation`, `team_id`, `competition_id` |

**Quality note:** `contest_type == 28` hardcoded as "classic" game type (line 197) — brittle if DraftKings changes their type IDs.

---

### `src/data/modules/reference_fetch.py`

Fetches player bio data from MLB StatsAPI. Not date-partitioned.

**Output:** `reference/player_info/player_info.parquet`

| Column | Type | Notes |
|---|---|---|
| `personId` | str | Join key |
| `player_name` | str | Full name |
| `weight` | float32 | lbs |
| `height_in_inches` | int32 | Parsed from `"6' 3\""` format |
| `birthDate` | datetime64[ns] | |
| `strikeZoneTop` | float32 | |
| `strikeZoneBottom` | float32 | |
| `batSide` | str | L/R |
| `pitchHand` | str | L/R |

---

## Layer 2 — Processing / Validation

### `src/data/modules/preprocessing.py`

Single source of truth for all output column schemas. All downstream code imports constants from here.

**Two-stage processing:**
1. **Process** — clean each raw table independently, enforce schema
2. **Prepare** — join tables, add player names and game context

**Entity ID contract:** All join keys (`personId`, `gamepk`, `team_id`, `away_id`, `home_id`) are `str`. This is intentional — enforced in schema constants.

**Game type filter:** `RELEVANT_GAME_TYPES = ['R', 'F', 'D', 'L', 'W']` excludes spring training (`S`), exhibition (`E`), all-star (`A`).

#### Schema Constants

```python
SCHEDULE_COLUMNS = [
    'gamepk', 'game_datetime', 'game_date', 'game_type', 'status',
    'away_name', 'home_name', 'away_id', 'home_id', 'game_num',
    'home_probable_pitcher', 'away_probable_pitcher',
    'away_score', 'home_score', 'venue_id', 'venue_name',
    'winning_team', 'losing_team', 'winning_pitcher', 'losing_pitcher',
]

PITCHER_BOXSCORE_COLUMNS = [
    'personId', 'gamepk', 'team_id', 'ip', 'h', 'r', 'er',
    'bb', 'k', 'hr', 'p', 's', 'outcome', 'fantasy_points',
]

BATTER_BOXSCORE_COLUMNS = [
    'gamepk', 'batting_order', 'personId', 'team_id', 'ab', 'h', 'r',
    'doubles', 'triples', 'hr', 'rbi', 'sb', 'bb', 'k', 'lob',
    'plate_appearances', 'singles', 'total_bases_from_h', 'total_bases', 'fantasy_points',
]

PLAYER_INFO_COLUMNS = [
    'personId', 'player_name', 'weight', 'height_in_inches',
    'birthDate', 'strikeZoneTop', 'strikeZoneBottom', 'batSide', 'pitchHand',
]

GAME_INFO_COLUMNS = [
    'gamepk', 'game_season', 'weather_condition', 'weather_temp', 'weather_wind',
    'probable_pitcher_away_id', 'probable_pitcher_away_fullName',
    'probable_pitcher_home_id', 'probable_pitcher_home_fullName', 'game_duration_minutes',
]

# prepare_* adds player_name, game_type, game_date, game_datetime, game_season
PREPARE_BATTER_BOXSCORE_COLUMNS  = BATTER_BOXSCORE_COLUMNS  + ['player_name', 'game_type', 'game_date', 'game_datetime', 'game_season']
PREPARE_PITCHER_BOXSCORE_COLUMNS = PITCHER_BOXSCORE_COLUMNS + ['player_name', 'game_type', 'game_date', 'game_datetime', 'game_season']
```

#### Column Type Schemas

```python
SCHEDULE_SCHEMA = {
    'gamepk': 'str', 'game_datetime': 'datetime64[ns]', 'game_date': 'datetime64[ns]',
    'game_type': 'str', 'status': 'str', 'away_name': 'str', 'home_name': 'str',
    'away_id': 'str', 'home_id': 'str', 'game_num': 'int32',
    'home_probable_pitcher': 'str', 'away_probable_pitcher': 'str',
    'away_score': 'float32', 'home_score': 'float32',   # float32 to handle null (postponed games)
    'venue_id': 'str', 'venue_name': 'str',
    'winning_team': 'str', 'losing_team': 'str', 'winning_pitcher': 'str', 'losing_pitcher': 'str',
}

BATTER_BOXSCORE_SCHEMA = {
    'gamepk': 'str', 'batting_order': 'str', 'personId': 'str', 'team_id': 'str',
    'ab': 'int32', 'h': 'int32', 'r': 'int32', 'doubles': 'int32', 'triples': 'int32',
    'hr': 'int32', 'rbi': 'int32', 'sb': 'int32', 'bb': 'int32', 'k': 'int32',
    'lob': 'int32', 'plate_appearances': 'int32', 'singles': 'int32',
    'total_bases_from_h': 'int32', 'total_bases': 'int32', 'fantasy_points': 'float32',
}

PITCHER_BOXSCORE_SCHEMA = {
    'personId': 'str', 'gamepk': 'str', 'team_id': 'str',
    'ip': 'float32', 'h': 'float32', 'r': 'float32', 'er': 'float32',
    'bb': 'float32', 'k': 'float32', 'hr': 'float32', 'p': 'float32', 's': 'float32',
    'outcome': 'str', 'fantasy_points': 'float32',
}

PLAYER_INFO_SCHEMA = {
    'personId': 'str', 'player_name': 'str', 'weight': 'float32',
    'height_in_inches': 'int32', 'birthDate': 'datetime64[ns]',
    'strikeZoneTop': 'float32', 'strikeZoneBottom': 'float32',
    'batSide': 'str', 'pitchHand': 'str',
}

GAME_INFO_SCHEMA = {
    'gamepk': 'str', 'game_season': 'Int32',   # nullable Int32
    'weather_condition': 'str', 'weather_temp': 'str', 'weather_wind': 'str',
    'probable_pitcher_away_id': 'str', 'probable_pitcher_away_fullName': 'str',
    'probable_pitcher_home_id': 'str', 'probable_pitcher_home_fullName': 'str',
    'game_duration_minutes': 'float32',
}

PREPARE_BATTER_BOXSCORE_SCHEMA = {
    # all of BATTER_BOXSCORE_SCHEMA plus:
    'player_name': 'str', 'game_type': 'str',
    'game_date': 'datetime64[ns]', 'game_datetime': 'datetime64[ns]',
    'game_season': 'Int32',  # nullable — some games missing from processed schedule
}

PREPARE_PITCHER_BOXSCORE_SCHEMA = {
    # all of PITCHER_BOXSCORE_SCHEMA plus:
    'player_name': 'str', 'game_type': 'str',
    'game_date': 'datetime64[ns]', 'game_datetime': 'datetime64[ns]',
    'game_season': 'Int32',
}
```

**Play-by-play output** (`PLAYBYPLAY_COLUMNS`, ~64 columns):

Core: `gamepk`, `play_id`, `inning`, `half_inning`, `batter_id`, `batter_name`, `pitcher_id`, `pitcher_name`, `batter_team_id`, `batter_team_name`, `pitcher_team_id`, `pitcher_team_name`, `play_result`, `play_description`, `event_type`, `is_pitch`, `count_balls`, `count_strikes`, `count_outs`

Statcast (float64, nullable for pre-Statcast games): `start_speed`, `end_speed`, `plate_x`, `plate_z`, `release_pos_x/y/z`, `pfx_x/z`, `spin_rate`, `spin_direction`, `launch_speed`, `launch_angle`, `total_distance`, `break_angle/length/vertical/horizontal`, `hit_coord_x/y`

**Quality notes:**
- `gamepk != '716404'` hardcoded filter in `process_batter_boxscore` (line 509) — undocumented exclusion, silent data removal
- `str(int(float(y)))` coercion for gamePk (line 481) — silently drops null IDs; no guard before the chain
- `batting_order` uses `battingOrder % 100` logic to extract batting position — can fail silently if value is null/NaN

---

### `src/lambdas/daily_process_data/handler.py`

Orchestrates processing for a single date. Accepts `{date, env}`.

**Reads from S3:**
```
{prefix}raw_data/games/schedule/{year}/{date}.parquet
{prefix}raw_data/games/game_info/{year}/{date}.parquet
{prefix}raw_data/games/batter_boxscore/{year}/{date}.parquet
{prefix}raw_data/games/pitcher_boxscore/{year}/{date}.parquet
{prefix}raw_data/playbyplay/{year}/{date}.parquet
reference/player_info/player_info.parquet          ← NO prefix — always reads production
```

**Writes to S3:**
```
{prefix}processed_data/games/schedule/{year}/{date}.parquet
{prefix}processed_data/games/game_info/{year}/{date}.parquet
{prefix}processed_data/games/batter_boxscore/{year}/{date}.parquet
{prefix}processed_data/games/pitcher_boxscore/{year}/{date}.parquet
{prefix}processed_data/prepared/batter_boxscore/{year}/{date}.parquet
{prefix}processed_data/prepared/pitcher_boxscore/{year}/{date}.parquet
{prefix}processed_data/prepared/playbyplay/{year}/{date}.parquet
```

Includes a gamePk completeness check: logs a warning if any gamepk from the schedule is missing from batter, pitcher, or playbyplay prepared tables.

**Quality note:** `player_info` does not respect `env: test` — line 57 hardcodes the path without `prefix`. Test runs read production player_info.

---

## Layer 3 — Feature Engineering

### `src/features/transforms/data_readers.py`

Thin S3 wrappers. All return full-season DataFrames (not single-date).

```python
read_batter_boxscore(season: int)  → s3://mlbdk/processed_data/prepared/batter_boxscore/{season}/
read_pitcher_boxscore(season: int) → s3://mlbdk/processed_data/prepared/pitcher_boxscore/{season}/
read_playbyplay(season: int)       → s3://mlbdk/processed_data/prepared/playbyplay/{season}/
read_schedule(season: int)         → s3://mlbdk/processed_data/games/schedule/{season}/
read_game_info(season: int)        → s3://mlbdk/processed_data/games/game_info/{season}/
```

---

### `src/features/transforms/utils.py`

| Function | Input | Output | Notes |
|---|---|---|---|
| `mlb_ip_to_true(ip: Series)` | Series of MLB IP notation (6.1 = 6⅓) | Series of decimal innings | `(fraction / 0.3)` converts .1→⅓, .2→⅔; handles null with `np.nan` |
| `identify_starters_from_pbp(pbp: DataFrame)` | prepared play-by-play | `DataFrame[gamepk, team_id, starter_personId, starter_name]` | First pitcher per (gamepk, pitcher_team_id), sorted by inning + event_index |

---

### `src/features/transforms/team_batter_base.py`

**Input:** `read_batter_boxscore(year)` — full prepared batter boxscore for the season

**Output columns:**

| Column | Type | Description |
|---|---|---|
| `team_id` | str | Join key |
| `gamepk` | str | Join key |
| `game_date` | date | Game date |
| `game_season` | int | Season year |
| `total_plate_appearances` | int | Per-game team PA |
| `total_ab`, `total_h`, `total_r` | int | Counting stats |
| `total_singles`, `total_doubles`, `total_triples`, `total_hr` | int | Hit breakdown |
| `total_bases`, `total_bases_from_h` | int | Total base counts |
| `total_rbi`, `total_lob`, `total_sb`, `total_bb_bat`, `total_k_bat` | int | |
| `total_batter_fantasy_points` | float | DK points |
| Rolling windows (3-game, 7-game, season) for PA, AB, H, R | float | `rolling_3game_pa_avg`, etc. |

**S3 output:** `feast/features/team_batter_base/{year}/{date}.parquet`

**Point-in-time safety:** Rolling windows close the day *before* game_date. No same-day leakage.

---

### `src/features/transforms/player_batter_base.py`

**Input:** `read_batter_boxscore(year)` — full prepared batter boxscore

**Output columns:**

| Column | Type | Description |
|---|---|---|
| `personId` | str | Join key |
| `team_id` | str | Join key |
| `gamepk` | str | Join key |
| `game_date` | date | |
| `game_season` | int | |
| `player_name` | str | |
| `batting_order` | str | "1"–"9" or null |
| Raw game stats: `ab`, `h`, `r`, `doubles`, `triples`, `hr`, `rbi`, `sb`, `bb`, `k`, `lob`, `plate_appearances`, `singles`, `total_bases`, `fantasy_points` | int/float | Single-game values |
| Rolling windows (3-game, 7-game, season) for H, K, TB, HR, RBI, BB, PA, fantasy_points | float | `rolling_7game_h_avg`, etc. |

**S3 output:** `feast/features/player_batter_base/{year}/{date}.parquet`

---

### `src/features/transforms/starting_pitcher_base.py`

**Inputs:** `read_pitcher_boxscore(year)` + `read_playbyplay(year)` (to identify starters via `identify_starters_from_pbp`)

**Per-game rate stats (capped to prevent outlier explosion on short outings):**

| Column | Cap | Notes |
|---|---|---|
| `era_gm` | 27 | ER/IP×9 |
| `whip_gm` | 9 | (H+BB)/IP |
| `k9_gm` | 27 | K/IP×9 |
| `bb9_gm` | 27 | BB/IP×9 |
| `hr9_gm` | 9 | HR/IP×9 |
| `k_per_ip` | 3 | K/IP |
| `bb_per_ip` | 3 | BB/IP |
| `hr_per_ip` | 1 | HR/IP |
| `h_per_ip` | 4 | H/IP |
| `er_per_ip` | 3 | ER/IP |

**Rolling windows:** 1-start, 3-start, 5-start, 10-start, season for all rate stats above.

**S3 output:** `feast/features/starting_pitcher_base/{year}/{date}.parquet`

**Quality note:** All `.clip(upper=N)` bounds are domain-knowledge caps for micro-sample outings (e.g., 1-batter appearances). They are intentional but undocumented in code.

---

### `src/features/transforms/bullpen_pitcher_base.py`

**Inputs:** `read_pitcher_boxscore(year)` + `read_playbyplay(year)`

Aggregates all non-starter pitchers per team per game.

**Per-game bullpen stats:**

| Column | Cap | Notes |
|---|---|---|
| `bp_ip_sum` | — | Total bullpen IP |
| `bp_h_sum`, `bp_er_sum`, `bp_bb_sum`, `bp_k_sum`, `bp_hr_sum` | — | Counting stats |
| `bp_pitchers` | — | Number of pitchers used |
| `bp_era` | 27 | Same cap logic as SP |
| `bp_whip` | 9 | |
| `bp_k9`, `bp_bb9`, `bp_hr9` | 27 | |
| `bp_k_per_ip`, `bp_bb_per_ip`, `bp_er_per_ip` | 3 | |
| `bp_hr_per_ip` | 1 | |
| `bp_h_per_ip` | 4 | |

**Rolling windows:** 1-game, 7-game, 14-game, season for all stats.

**S3 output:** `feast/features/bullpen_pitcher_base/{year}/{date}.parquet`

---

### `src/features/transforms/compute_season_summaries.py`

CLI backfill script (not Lambda). Run directly: `python compute_season_summaries.py [year]`.

Computes full-season aggregates for use in the Streamlit app and Bayesian priors.

| Function | Input | Output |
|---|---|---|
| `compute_batter_season_summary(year)` | `read_batter_boxscore` | Per-player per-season: PA, H, HR, BB, K, TB, OBP, SLG, ISO, wOBA, beta distribution parameters |
| `compute_sp_season_summary(year)` | `read_pitcher_boxscore` + `read_playbyplay` | Per-SP per-season: starts, IP, ERA, WHIP, K9, BB9, HR9, k_per_ip, avg IP/start |
| `compute_team_season_summary(year)` | both boxscores | Per-team per-season: offensive + pitching totals and rates |

Output S3 path: `processed_data/season_summaries/{batter_season_summary,sp_season_summary,team_season_summary}/game_season={year}/`

**Note:** Excludes 2020 (COVID season) in all DuckDB queries.

---

## Layer 4 — Feature Store

### `src/features/feature.py`

Feast registry definitions. Entities and FeatureViews registered here are materialized by `daily_feature_create`.

| FeatureView | Entity | TTL | Source |
|---|---|---|---|
| `team_batter_base_fv` | `team` (join: `team_id`) | 7 days | `feast/features/team_batter_base/` |
| `player_batter_base_fv` | `player` (join: `personId`) | 7 days | `feast/features/player_batter_base/` |
| `starting_pitcher_base_fv` | `player` (join: `personId`) | 7 days | `feast/features/starting_pitcher_base/` |
| `bullpen_pitcher_base_fv` | `team` (join: `team_id`) | 7 days | `feast/features/bullpen_pitcher_base/` |

---

### `src/lambdas/daily_feature_create/handler.py`

Runs all four feature transforms and materializes into Feast. Accepts `{date, env}`.

For each transform: compute full-season features → slice to single-date snapshot → write Parquet → materialize.

**Reads from S3** (via data_readers inside each transform): full season prepared data

**Writes to S3:**
```
feast/features/team_batter_base/{year}/{date}.parquet
feast/features/player_batter_base/{year}/{date}.parquet
feast/features/starting_pitcher_base/{year}/{date}.parquet
feast/features/bullpen_pitcher_base/{year}/{date}.parquet
```

Returns `{"statusCode": 200, "succeeded": [...], "failed": [...], "results": {...}}` — 207 if any transform failed.

**Known bugs (not yet fixed):**
- Line 49: Missing `f` prefix — `key="feast/features/team_batter_base/{year}/{date}.parquet"` is a plain string. `{year}` and `{date}` are literal text. All team_batter_base snapshots overwrite the same path. The other three transforms have the f-string correctly.
- Line 121: `results['materialzie']` typo — error state writes to wrong key, silently lost from response.

---

## S3 Layout

```
mlbdk/
├── raw_data/
│   ├── games/
│   │   ├── schedule/{year}/{date}.parquet
│   │   ├── game_info/{year}/{date}.parquet
│   │   ├── batter_boxscore/{year}/{date}.parquet
│   │   └── pitcher_boxscore/{year}/{date}.parquet
│   ├── playbyplay/{year}/{date}.parquet
│   ├── odds/
│   │   ├── team_odds/{year}/{date}.parquet
│   │   └── player_props/{year}/{date}.parquet
│   └── draftkings/{contests,games,contest_players}/{year}/{timestamp}.parquet
├── reference/
│   └── player_info/player_info.parquet          ← not date-partitioned
├── processed_data/
│   ├── games/
│   │   ├── schedule/{year}/{date}.parquet
│   │   └── game_info/{year}/{date}.parquet
│   ├── games/
│   │   ├── batter_boxscore/{year}/{date}.parquet
│   │   └── pitcher_boxscore/{year}/{date}.parquet
│   ├── prepared/
│   │   ├── batter_boxscore/{year}/{date}.parquet
│   │   ├── pitcher_boxscore/{year}/{date}.parquet
│   │   └── playbyplay/{year}/{date}.parquet
│   └── season_summaries/{table}/game_season={year}/
└── feast/
    └── features/
        ├── team_batter_base/{year}/{date}.parquet
        ├── player_batter_base/{year}/{date}.parquet
        ├── starting_pitcher_base/{year}/{date}.parquet
        └── bullpen_pitcher_base/{year}/{date}.parquet
```

`env: test` routes all Lambda reads/writes under `test/` prefix — except `reference/player_info/player_info.parquet`, which always reads production.

---

## Data Contract Rules

1. **All join keys are `str`**: `personId`, `gamepk`, `team_id`, `away_id`, `home_id`. Never cast to int before joining.

2. **Numeric stats**: counts (`ab`, `h`, etc.) are `int32`; rates and money stats are `float32`; Statcast measurements are `float64`.

3. **Nullable integers**: `game_season` uses pandas `Int32` (nullable) — some games appear in raw data without a corresponding Final entry in schedule.

4. **Date columns**: `game_date` is `datetime64[ns]`, `game_datetime` is `datetime64[ns]` UTC-naive. MLB API returns Eastern — normalized at source.

5. **Rolling windows close before game date**: Features for game on date D use data through D-1. No same-day leakage.

6. **Season filter**: 2020 excluded from all rolling features and season summaries (COVID shortened season distorts per-game rates).

7. **Game type filter**: `RELEVANT_GAME_TYPES = ['R', 'F', 'D', 'L', 'W']` — applied in prepare_* and feature transforms. Spring training (`S`), exhibition (`E`), all-star (`A`) are excluded.

---

## Quality Issues for Model Building

### Critical (data corruption)

| File | Location | Issue |
|---|---|---|
| `daily_feature_create/handler.py` | Line 49 | Missing `f` prefix — team_batter_base snapshot writes to literal path `"feast/features/team_batter_base/{year}/{date}.parquet"`, overwriting every run. Other three transforms are fine. |

### High (silent data issues)

| File | Location | Issue |
|---|---|---|
| `preprocessing.py` | Line 509 | `gamepk != '716404'` hardcoded filter silently drops one specific game. Reason undocumented. |
| `daily_process_data/handler.py` | Line 57 | `player_info` path ignores `env: test` prefix — test runs always read production player data. |
| `fetchers.py` | Line 75 | `pd.to_numeric(errors='coerce').fillna(0)` on `venue_id` — failed coercions silently become 0, a valid-looking ID. |
| `starting_pitcher_base.py` / `bullpen_pitcher_base.py` | Lines 31-41, 59-69 | Rate stats capped with hardcoded `.clip(upper=N)` bounds (ERA≤27, WHIP≤9, etc.). Intentional for micro-sample protection but undocumented — will silently distort features if data patterns change. |

### Medium (brittleness / model risk)

| File | Location | Issue |
|---|---|---|
| `preprocessing.py` | Line 481 | `str(int(float(y)))` for gamePk — no null guard; silently fails on non-numeric IDs |
| `preprocessing.py` | Line 482 | `battingOrder % 100` — NaN batting order silently propagates through modulo |
| `preprocessing.py` | Line 364 | `df[df['ip'] != 'IP']` header removal — brittle string match |
| `daily_feature_create/handler.py` | Line 121 | `results['materialzie']` typo — materialization errors silently write to wrong key in response body |
| `daily_dkslate_fetch.py` | Line 197 | DraftKings `contest_type == 28` hardcoded as "classic" — undocumented magic number |
| `compute_season_summaries.py` | — | `S3_BASE` and heavy imports (`s3fs`, `pyarrow`) defined inside `if __name__ == "__main__"` — `write_season_summary()` will fail if called as imported function |
