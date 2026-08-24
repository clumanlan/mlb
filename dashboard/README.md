# MLB Pre-Game Dashboard

Personal dashboard for MLB DraftKings prop research. Reads from the `mlbdk` S3 bucket — no separate database.

## Quick start

```bash
cd mlb                          # all make targets run from the repo root
make dashboard-dev              # starts backend + frontend, opens browser
make dashboard-stop             # kills both servers
```

The first run requires setup (see below). After that, `make dashboard-dev` is all you need.

---

## First-time setup

### Backend

```bash
cd dashboard/backend
cp .env.example .env      # fill in AWS credentials
source /Users/clumanlan/projects/mlb/.venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd dashboard/frontend
npm install
```

---

## Make targets

All targets live in the root `mlb/Makefile` and are run from `mlb/`.

| Command | What it does |
|---|---|
| `make dashboard-dev` | Start backend + frontend, open browser |
| `make dashboard-backend` | Start only the FastAPI backend on :8000 |
| `make dashboard-frontend` | Start only the Vite frontend on :5173 |
| `make dashboard-stop` | Kill both dev servers |
| `make dashboard-open` | Open the dashboard (servers must be running) |
| `make dashboard-test` | Run unit tests (no AWS needed) |
| `make dashboard-test-int` | Run integration tests against real S3 (yesterday's date) |
| `make dashboard-test-int DATE=2026-05-09` | Integration tests for a specific date |

---

## What's built

### Stage 1 — Data Pipeline Status (done)

Three cards showing whether each Lambda ran successfully and on time:

| Card | Lambda | Staleness rule |
|---|---|---|
| Raw — MLB fetch | `daily_mlb_fetch` | stale if `run_date != yesterday` |
| Processed data | `daily_process_data` | stale if `run_date != yesterday` |
| Odds fetch | `daily_odds_fetch` | stale if `run_date != today` |

Each card shows: run date · duration · games processed per step · any error. A summary bar below shows `2/3 ok · 1 stale`.

### Stage 2 — Today's Slate (done)

Game table sorted by CT game time. Columns:

| Column | Source | Notes |
|---|---|---|
| Time (CT) | Schedule | UTC converted to Central Time |
| Matchup | Schedule | Away @ Home |
| Lineup | `raw_data/lineups/states/` | CONFIRMED / PENDING / SKIPPED |
| Odds | `raw_data/odds/team_odds/` | shown if present, dash if missing |
| Prediction | — | Placeholder for Stage 3 |

### Stage 3 — Model Predictions (not yet built)
### Stage 4 — Bet Recommendations (not yet built)

---

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Liveness check → `{"status": "ok"}` |
| `GET /api/pipeline-status` | Three Lambda cards + summary |
| `GET /api/today-slate` | Today's games with lineup + odds |
| `GET /api/today-slate?date=YYYY-MM-DD` | Games for a specific date (useful for debugging) |

Interactive docs: http://localhost:8000/docs

---

## S3 data sources

All data lives in the `mlbdk` bucket (us-east-2).

```
lambdas/status/{function_name}/{date}.json   → pipeline status cards
  daily_mlb_fetch / daily_process_data / daily_odds_fetch

raw_data/schedule/{year}/{date}.json         → slate schedule (daily_schedule_fetch)
raw_data/games/schedule/{year}/schedule_{date}.parquet  → fallback (daily_mlb_fetch)

raw_data/lineups/states/{year}/{date}.json   → lineup confirmation states
raw_data/odds/team_odds/{year}/{date}.parquet → team odds (DraftKings book)
```

**Schedule fallback:** the route tries the new JSON path first (`daily_schedule_fetch`), then falls back to the legacy Parquet path (`daily_mlb_fetch`). The dashboard shows real game data even before `daily_schedule_fetch` is deployed to production.

---

## Project structure

```
dashboard/
├── Makefile                  # dev, test, stop targets
├── README.md
├── backend/
│   ├── main.py               # FastAPI app — all routes and business logic
│   ├── s3_client.py          # boto3 helpers: get_latest_s3_json, get_s3_json, get_s3_parquet
│   ├── requirements.txt
│   ├── pytest.ini            # sets pythonpath and testpaths for the backend
│   ├── .env.example          # copy to .env and fill in AWS credentials
│   └── tests/
│       ├── conftest.py       # sys.path setup, integration marker, run_date fixture
│       ├── test_app.py       # health endpoint
│       ├── test_s3_client.py # unit tests for all three S3 helpers (28 total)
│       ├── test_pipeline_status.py  # staleness logic + route tests
│       ├── test_today_slate.py      # UTC→CT, schedule join, odds join
│       └── test_integration.py      # real S3 smoke tests (--date flag)
└── frontend/
    ├── index.html
    ├── vite.config.js        # proxies /api/* to localhost:8000
    ├── package.json
    └── src/
        ├── main.jsx          # React entry point
        ├── App.jsx           # root layout — header + section list
        ├── index.css         # CSS variables, all shared styles
        └── components/
            ├── PipelineStatus.jsx   # Section 1 — three pipeline cards
            └── TodaysSlate.jsx      # Section 2 — game table
```

---

## Running tests

```bash
# From mlb/ root — no cd needed
make dashboard-test                        # unit tests, no AWS
make dashboard-test-int                    # integration, uses yesterday
make dashboard-test-int DATE=2026-05-09   # integration, specific date
```

Unit test count: **28 passing**  
Integration test count: **8 passing** (against real S3 data)
