# Implementation Plan: MLB Pre-Game Dashboard

## Audit
- ✅ 134 tests passing in the `mlb/` Lambda project — baseline is clean
- This is a **greenfield project** at `/Users/clumanlan/projects/mlb-dashboard/`

## S3 Path Corrections (vs. original spec)
The spec listed `raw_data/odds/{date}.parquet` — the actual path written by `daily_odds_fetch` is:
- `raw_data/odds/team_odds/{year}/{date}.parquet`

All other paths confirmed correct:
- Schedule: `raw_data/schedule/{year}/{date}.json`
- Lineup states: `raw_data/lineups/states/{year}/{date}.json`

## Project Layers

| Layer | Files | TDD Pattern |
|---|---|---|
| S3 Client | `backend/s3_client.py` | Mock boto3, test return value contracts |
| API Routes | `backend/main.py` | FastAPI `TestClient`, patch `s3_client` functions |
| Frontend Shell | `App.jsx`, `index.css` | Visual verification in browser |
| Frontend Components | `PipelineStatus.jsx`, `TodaysSlate.jsx` | Visual verification in browser |

**Why no frontend unit tests?** The user is learning React. Wiring up Jest/Vitest adds tooling friction before concepts land. We verify visually — loading state, error state, happy path. TDD for the backend covers the contracts that the frontend depends on.

---

## Integration Test Setup

At the end of each backend EPIC, a real-data smoke test verifies that the implementation works against actual S3 data from **2026-05-08**.

**Why a separate marker?** Integration tests hit real AWS. Normal `pytest tests/` runs are fast and offline (all mocks). The marker keeps them opt-in.

**conftest.py snippet (add to `backend/tests/conftest.py`):**
```python
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: hits real AWS S3 — requires credentials in .env"
    )
```

**Run integration tests only:**
```bash
cd mlb-dashboard/backend
pytest tests/ -m integration -v
```

**Run unit tests only (default):**
```bash
pytest tests/ -v           # integration tests skipped automatically
```

**Credential requirement:** Integration tests read from `mlbdk` in us-east-2. AWS credentials must be present in `backend/.env` or in the shell environment before running.

---

## EPIC 1: Project Scaffolding
**Why:** Establishes the directory structure and running skeleton before any feature code. Both backend and frontend must boot without errors before we write a single business-logic test.

**Depends on:** none

---

### STORY 1.1: Backend foundation

---

#### TASK 1.1.1 — Scaffold FastAPI project and smoke test

```
RED — Write the failing test first
  File: mlb-dashboard/backend/tests/test_app.py
  Test name: test_health_endpoint_returns_ok
  What to assert:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
  Run: cd mlb-dashboard/backend && pytest tests/ -v
  Expected failure: ModuleNotFoundError: No module named 'main'

GREEN — Minimal code to pass
  File: mlb-dashboard/backend/main.py
  Create FastAPI app with:
    - CORSMiddleware allowing localhost:5173
    - GET /api/health returning {"status": "ok"}
  Create backend/requirements.txt:
    fastapi
    uvicorn
    boto3
    pandas
    pyarrow
    python-dotenv
  Install: pip install -r requirements.txt
  Run: pytest tests/ -v → should pass

REFACTOR — none needed at this stage
```

---

### STORY 1.2: Frontend foundation

---

#### TASK 1.2.1 — Scaffold Vite + React project

No unit tests here — this is tooling setup only.

```
Setup steps:
  cd mlb-dashboard
  npm create vite@latest frontend -- --template react
  cd frontend && npm install

Verify:
  npm run dev → loads at localhost:5173 without errors

Then replace the generated files with project structure:
  - src/App.jsx → section layout shell (header + two placeholder sections)
  - src/index.css → CSS variables + global styles
  - src/components/PipelineStatus.jsx → placeholder
  - src/components/TodaysSlate.jsx → placeholder

Visual check:
  - Warm off-white background
  - Dashboard header with today's date visible
  - Two section placeholders with ALL-CAPS labels
```

---

## EPIC 2: S3 Client Layer
**Why:** Isolates all S3 I/O into tested helper functions. Route tests mock at `s3_client` — they never touch real AWS. This boundary keeps tests fast and deterministic.

**Depends on:** EPIC 1

---

### STORY 2.1: S3 JSON and Parquet helpers

---

#### TASK 2.1.1 — `get_latest_s3_json(bucket, prefix)` helper

The function: lists all objects under `prefix`, picks the one with the most recent `LastModified`, fetches it, parses JSON, returns a dict.

Why this function? The status files are written once per day and we always want the latest. Instead of hardcoding dates in routes, we let S3 tell us what's most recent.

```
RED — Write the failing test first
  File: mlb-dashboard/backend/tests/test_s3_client.py

  Test 1: test_get_latest_s3_json_returns_parsed_json
    Setup: mock boto3 list_objects_v2 to return one object key
           mock get_object to return json bytes b'{"status": "success"}'
    Assert: result == {"status": "success"}

  Test 2: test_get_latest_s3_json_returns_most_recent_when_multiple_objects
    Setup: mock list_objects_v2 to return three objects with different LastModified timestamps
    Assert: the object with the latest timestamp is fetched (verify via mock call args)

  Test 3: test_get_latest_s3_json_raises_file_not_found_when_prefix_empty
    Setup: mock list_objects_v2 to return empty Contents
    Assert: raises FileNotFoundError

  Run: pytest tests/test_s3_client.py -v
  Expected failure: ModuleNotFoundError: No module named 's3_client'

GREEN — Minimal code to pass
  File: mlb-dashboard/backend/s3_client.py
  def get_latest_s3_json(bucket: str, prefix: str) -> dict:
    - boto3.client("s3")
    - list_objects_v2 with Prefix
    - if no Contents: raise FileNotFoundError
    - sort by LastModified, pick latest Key
    - get_object, read Body, json.loads, return

REFACTOR
  Extract boto3 client creation so it can be injected in tests (avoids patching path fragility)
  Run: pytest tests/test_s3_client.py -v → still green
```

---

#### TASK 2.1.2 — `get_s3_parquet(bucket, key)` helper

The function: fetches a Parquet file from S3 by exact key, returns a pandas DataFrame.

Why pandas + pyarrow? The odds file has nested `bookmakers` column — Parquet handles it natively. We only need to know if a team pair exists, so we read the file and check for row presence.

```
RED — Write the failing test first
  File: mlb-dashboard/backend/tests/test_s3_client.py

  Test 1: test_get_s3_parquet_returns_dataframe
    Setup: build a tiny real DataFrame, write to io.BytesIO as Parquet
           mock s3 get_object Body to return those bytes
    Assert: result is a pd.DataFrame, has expected columns

  Test 2: test_get_s3_parquet_raises_file_not_found_on_missing_key
    Setup: mock get_object to raise ClientError with code "NoSuchKey"
    Assert: raises FileNotFoundError

  Run: pytest tests/test_s3_client.py -v → new tests fail

GREEN — Minimal code to pass
  def get_s3_parquet(bucket: str, key: str) -> pd.DataFrame:
    - try: get_object, read Body into BytesIO
    - pd.read_parquet(buffer)
    - except ClientError where code is NoSuchKey: raise FileNotFoundError

REFACTOR — none needed
```

---

### Integration Smoke Test — EPIC 2

After both S3 helpers are green, run this against real S3 data from 2026-05-08 to confirm the actual file shapes match your mocked expectations.

```
File: mlb-dashboard/backend/tests/test_integration.py

Setup:
  import pytest
  import s3_client

@pytest.mark.integration
class TestS3ClientIntegration:

  def test_get_latest_s3_json_reads_real_schedule(self):
    """Reads the real schedule file for 2026-05-08 from S3."""
    result = s3_client.get_latest_s3_json("mlbdk", "raw_data/schedule/2026/")
    assert isinstance(result, list), "schedule should be a list of game dicts"
    assert len(result) > 0, "expected at least one game on 2026-05-08"
    game = result[0]
    for field in ("game_pk", "home_team_name", "away_team_name", "game_time_utc", "venue_name"):
        assert field in game, f"missing field: {field}"

  def test_get_latest_s3_json_reads_real_lineup_states(self):
    """Reads real lineup states from S3 and checks structure."""
    result = s3_client.get_latest_s3_json("mlbdk", "raw_data/lineups/states/2026/")
    assert isinstance(result, dict), "lineup states should be a dict keyed by game_pk string"
    for game_pk_str, state in result.items():
        assert isinstance(game_pk_str, str), "game_pk keys must be strings"
        assert "status" in state
        assert state["status"] in ("PENDING", "CONFIRMED", "SKIPPED")

  def test_get_s3_parquet_reads_real_team_odds(self):
    """Reads real team odds Parquet for 2026-05-08 and checks columns."""
    df = s3_client.get_s3_parquet("mlbdk", "raw_data/odds/team_odds/2026/2026-05-08.parquet")
    assert not df.empty, "expected at least one odds row for 2026-05-08"
    for col in ("home_team", "away_team"):
        assert col in df.columns, f"missing column: {col}"

Run: pytest tests/test_integration.py -m integration -v
Expected: all three pass — if any fail, the shape of the real data differs from your
          fixture assumptions and your unit test fixtures need to be updated.
```

---

## EPIC 3: Pipeline Status Endpoint
**Why:** Powers Section 1 of the dashboard. This endpoint reads the three Lambda status files and computes staleness.

**Depends on:** EPIC 2

---

### STORY 3.1: GET /api/pipeline-status

---

#### TASK 3.1.1 — Staleness functions

Write pure functions that decide whether a card is stale. Pure means no `datetime.now()` inside — the caller passes `today` in. This makes testing trivial: no need for `freezegun`, just pass dates as strings.

```
RED — Write the failing test first
  File: mlb-dashboard/backend/tests/test_pipeline_status.py

  Test 1: test_is_stale_mlb_fetch_when_run_date_not_yesterday
    Call: is_stale_mlb_fetch(run_date="2026-05-06", today=date(2026, 5, 9))
    Assert: True  (expected yesterday=2026-05-08, got 2026-05-06)

  Test 2: test_is_not_stale_mlb_fetch_when_run_date_is_yesterday
    Call: is_stale_mlb_fetch(run_date="2026-05-08", today=date(2026, 5, 9))
    Assert: False

  Test 3: test_is_stale_odds_fetch_when_run_date_not_today
    Call: is_stale_odds_fetch(run_date="2026-05-08", today=date(2026, 5, 9))
    Assert: True

  Test 4: test_is_not_stale_odds_fetch_when_run_date_is_today
    Call: is_stale_odds_fetch(run_date="2026-05-09", today=date(2026, 5, 9))
    Assert: False

  Run: pytest tests/test_pipeline_status.py -v → fail (no module yet)

GREEN — Minimal code to pass
  In main.py (or a small helpers.py):
    def is_stale_mlb_fetch(run_date: str, today: date) -> bool:
        return date.fromisoformat(run_date) != (today - timedelta(days=1))

    def is_stale_odds_fetch(run_date: str, today: date) -> bool:
        return date.fromisoformat(run_date) != today

REFACTOR — none; these are already minimal
```

---

#### TASK 3.1.2 — Route assembly and response shape

```
RED — Write the failing test first
  File: mlb-dashboard/backend/tests/test_pipeline_status.py

  Fixtures: define three status dicts matching the real schema
    MLB_FETCH_STATUS = {"function_name": "daily_mlb_fetch", "run_date": "2026-05-08",
                        "status": "success", "completed_at": "2026-05-08T10:01:38.701783",
                        "duration_seconds": 55.96,
                        "games_processed": {"schedule": 15, "game_info": 15, ...}, "error": null}
    ... (similar for daily_process_data and daily_odds_fetch)

  Test 1: test_pipeline_status_returns_three_cards
    Setup: patch s3_client.get_latest_s3_json to return fixtures by call order
    Call: client.get("/api/pipeline-status")
    Assert: response.status_code == 200
            len(response.json()["cards"]) == 3

  Test 2: test_pipeline_status_card_has_required_fields
    (same mock setup)
    Assert: each card has keys:
      title, run_date, completed_at, duration_seconds, is_stale, games_processed, error

  Test 3: test_pipeline_status_summary_counts_one_stale
    Setup: MLB fetch run_date = "2026-05-06" (stale), others current
           patch get_latest_s3_json accordingly
           patch datetime.date.today() to return 2026-05-09
    Assert: summary["ok_count"] == 2, summary["stale_count"] == 1

  Test 4: test_pipeline_status_handles_missing_s3_file
    Setup: first get_latest_s3_json call raises FileNotFoundError
    Assert: response.status_code == 200  (no crash)
            cards[0]["error"] is not None  (error surfaced in the card)

  Run: pytest tests/test_pipeline_status.py -v → fail

GREEN — Minimal code to pass
  In main.py:
    CARD_CONFIGS = [
      {"key": "daily_mlb_fetch",    "title": "Raw — MLB fetch",    "staleness": "yesterday"},
      {"key": "daily_process_data", "title": "Processed data",      "staleness": "yesterday"},
      {"key": "daily_odds_fetch",   "title": "Odds fetch",          "staleness": "today"},
    ]

    @app.get("/api/pipeline-status")
    def pipeline_status():
        today = date.today()
        cards = []
        for config in CARD_CONFIGS:
            try:
                data = s3_client.get_latest_s3_json("mlbdk", f"lambdas/status/{config['key']}/")
                is_stale = ... (call appropriate staleness function)
                cards.append({
                    "title": config["title"],
                    "run_date": data.get("run_date"),
                    "completed_at": data.get("completed_at"),
                    "duration_seconds": data.get("duration_seconds"),
                    "is_stale": is_stale,
                    "games_processed": data.get("games_processed", {}),
                    "error": data.get("error"),
                })
            except FileNotFoundError:
                cards.append({"title": config["title"], "error": "status file not found", ...})

        ok_count = sum(1 for c in cards if not c["is_stale"] and not c["error"])
        stale_count = sum(1 for c in cards if c.get("is_stale"))
        return {"cards": cards, "summary": {"ok_count": ok_count, "stale_count": stale_count}}

REFACTOR
  Extract card-building logic out of the route function — route stays thin
  Run: pytest tests/test_pipeline_status.py -v → still green
```

---

### Integration Smoke Test — EPIC 3

After the pipeline status route is green, hit the real endpoint with no mocks and verify it returns well-formed cards with the correct `games_processed` keys for each Lambda.

```
File: mlb-dashboard/backend/tests/test_integration.py
(add to the existing file — no separate class needed)

@pytest.mark.integration
class TestPipelineStatusIntegration:

  def test_returns_three_cards_with_correct_shape(self):
    """Calls the real endpoint (real S3, no mocks) and checks structure."""
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    response = client.get("/api/pipeline-status")
    assert response.status_code == 200
    body = response.json()

    assert len(body["cards"]) == 3
    assert "summary" in body
    assert "ok_count" in body["summary"]
    assert "stale_count" in body["summary"]

    for card in body["cards"]:
        for field in ("title", "run_date", "duration_seconds", "is_stale", "games_processed", "error"):
            assert field in card, f"card missing field: {field}"

  def test_mlb_fetch_card_has_correct_games_processed_keys(self):
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    body = client.get("/api/pipeline-status").json()
    cards = {c["title"]: c for c in body["cards"]}

    mlb = cards["Raw — MLB fetch"]
    assert set(mlb["games_processed"].keys()) == {
        "schedule", "game_info", "batter_boxscore", "pitcher_boxscore", "playbyplay"
    }

    process = cards["Processed data"]
    assert set(process["games_processed"].keys()) == {
        "schedule", "batter_prepared", "pitcher_prepared", "playbyplay_prepared"
    }

    odds = cards["Odds fetch"]
    assert set(odds["games_processed"].keys()) == {"team_odds", "player_props"}

Run: pytest tests/test_integration.py::TestPipelineStatusIntegration -m integration -v
If the games_processed key sets don't match → update the Lambda CARD_CONFIGS titles or
the expected sets above to match what's actually in S3.
```

---

## EPIC 4: Today's Slate Endpoint
**Why:** Powers Section 2. More joins involved — schedule + lineups + odds all come together here.

**Depends on:** EPIC 2

**Design note — optional `?date` param:** Add `date: str = None` as a query param to the route (defaulting to `str(date.today())`). This is needed for the integration smoke test (pass `?date=2026-05-08` to read known data) and is useful for manual debugging in general. Do not add it to the spec — just make it work in the route.

---

### STORY 4.1: GET /api/today-slate

---

#### TASK 4.1.1 — Schedule + lineup join, CT conversion

```
RED — Write the failing test first
  File: mlb-dashboard/backend/tests/test_today_slate.py

  Fixtures:
    SCHEDULE = [
      {"game_pk": 745123, "home_team_name": "Toronto Blue Jays",
       "away_team_name": "Los Angeles Angels", "game_time_utc": "2026-05-09T19:08:00Z",
       "venue_name": "Rogers Centre", "status": "Scheduled"},
      {"game_pk": 745124, "home_team_name": "Baltimore Orioles",
       "away_team_name": "Oakland Athletics", "game_time_utc": "2026-05-09T20:05:00Z",
       "venue_name": "Oriole Park", "status": "Scheduled"},
    ]
    LINEUP_STATES = {
      "745123": {"status": "PENDING"},
      "745124": {"status": "CONFIRMED"},
    }

  Test 1: test_today_slate_returns_games_list
    Mock: get_latest_s3_json returns SCHEDULE for schedule path
          get_latest_s3_json returns LINEUP_STATES for lineups path
          get_s3_parquet raises FileNotFoundError (no odds file)
    Call: client.get("/api/today-slate")
    Assert: response.status_code == 200
            len(response.json()["games"]) == 2

  Test 2: test_today_slate_game_has_required_fields
    (same mocks)
    Assert: each game has: game_pk, away_team, home_team, game_time_utc,
            game_time_ct, venue, lineup_status, has_odds, prediction

  Test 3: test_today_slate_games_sorted_ascending_by_time
    Setup: SCHEDULE with game_pk 745123 starting AFTER 745124
    Assert: response games list has 745124 first

  Test 4: test_today_slate_converts_utc_to_central_time
    game_time_utc = "2026-05-09T19:08:00Z"  (19:08 UTC = 14:08 CT)
    Assert: game_time_ct == "2:08 PM"

  Test 5: test_today_slate_lineup_status_defaults_to_pending_for_unknown_game
    Setup: LINEUP_STATES has no entry for game_pk 745123
    Assert: game 745123 has lineup_status == "PENDING"

  Run: pytest tests/test_today_slate.py -v → fail

GREEN — Minimal code to pass
  @app.get("/api/today-slate")
  def today_slate():
      today = date.today()
      year = str(today.year)

      schedule = s3_client.get_latest_s3_json("mlbdk", f"raw_data/schedule/{year}/")
      try:
          lineup_states = s3_client.get_latest_s3_json("mlbdk", f"raw_data/lineups/states/{year}/")
      except FileNotFoundError:
          lineup_states = {}

      games = []
      for game in sorted(schedule, key=lambda g: g["game_time_utc"]):
          game_pk_str = str(game["game_pk"])
          lineup_entry = lineup_states.get(game_pk_str, {})
          lineup_status = lineup_entry.get("status", "PENDING")
          game_time_ct = utc_to_ct(game["game_time_utc"])  # see below
          games.append({
              "game_pk": game["game_pk"],
              "away_team": game["away_team_name"],
              "home_team": game["home_team_name"],
              "game_time_utc": game["game_time_utc"],
              "game_time_ct": game_time_ct,
              "venue": game["venue_name"],
              "lineup_status": lineup_status,
              "has_odds": False,  # filled in next task
              "prediction": None,
          })
      return {"date": str(today), "games": games, "games_sorted_by_time": True}

  UTC to CT helper:
    from datetime import timezone
    import zoneinfo
    CT = zoneinfo.ZoneInfo("America/Chicago")
    def utc_to_ct(utc_str: str) -> str:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        ct = dt.astimezone(CT)
        hour = ct.hour % 12 or 12
        return f"{hour}:{ct.strftime('%M')} {'AM' if ct.hour < 12 else 'PM'}"

REFACTOR
  Extract utc_to_ct into a standalone function so it can be unit-tested independently
  Run: pytest tests/test_today_slate.py -v → still green
```

---

#### TASK 4.1.2 — Odds join (team name match)

```
RED — Write the failing test first
  File: mlb-dashboard/backend/tests/test_today_slate.py

  Fixtures:
    ODDS_DF = pd.DataFrame([
      {"home_team": "Toronto Blue Jays", "away_team": "Los Angeles Angels",
       "commence_time": "2026-05-09T19:08:00Z", "bookmakers": []},
    ])

  Test 1: test_today_slate_has_odds_true_when_teams_match
    Mock get_s3_parquet to return ODDS_DF
    Assert: game with game_pk 745123 (TOR vs LAA) has has_odds == True

  Test 2: test_today_slate_has_odds_false_when_no_team_match
    Assert: game 745124 (BAL vs OAK) has has_odds == False (not in ODDS_DF)

  Test 3: test_today_slate_handles_missing_odds_file_gracefully
    Mock get_s3_parquet raises FileNotFoundError
    Assert: response.status_code == 200
            all games have has_odds == False

  Run: pytest tests/test_today_slate.py -v → tests 1-2 fail (has_odds always False now)

GREEN — Minimal code to pass
  In the today_slate route, after building the games list:
    try:
        key = f"raw_data/odds/team_odds/{year}/{today}.parquet"
        odds_df = s3_client.get_s3_parquet("mlbdk", key)
        odds_pairs = set(zip(odds_df["home_team"], odds_df["away_team"]))
    except FileNotFoundError:
        odds_pairs = set()

    for game in games:
        game["has_odds"] = (game["home_team"], game["away_team"]) in odds_pairs

REFACTOR — none needed
```

---

### Integration Smoke Test — EPIC 4

After the slate route is green, hit it with `?date=2026-05-08` against real S3. This catches join bugs (wrong team name format, missing lineup default) that mocks can't catch.

```
File: mlb-dashboard/backend/tests/test_integration.py

@pytest.mark.integration
class TestTodaysSlateIntegration:

  def test_returns_games_for_2026_05_08(self):
    """Calls real endpoint with a specific historical date and checks structure."""
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    response = client.get("/api/today-slate?date=2026-05-08")
    assert response.status_code == 200
    body = response.json()

    assert body["date"] == "2026-05-08"
    assert isinstance(body["games"], list)
    assert len(body["games"]) > 0, "expected games on 2026-05-08"

  def test_all_games_have_required_fields(self):
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    body = client.get("/api/today-slate?date=2026-05-08").json()
    for game in body["games"]:
        for field in ("game_pk", "away_team", "home_team", "game_time_utc",
                      "game_time_ct", "venue", "lineup_status", "has_odds", "prediction"):
            assert field in game, f"game missing field: {field}"
        assert game["lineup_status"] in ("PENDING", "CONFIRMED", "SKIPPED")
        assert isinstance(game["has_odds"], bool)
        assert game["prediction"] is None

  def test_games_sorted_ascending_by_time(self):
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    body = client.get("/api/today-slate?date=2026-05-08").json()
    times = [g["game_time_utc"] for g in body["games"]]
    assert times == sorted(times), "games must be sorted ascending by UTC time"

  def test_ct_time_format_is_valid(self):
    from main import app
    from fastapi.testclient import TestClient
    import re
    client = TestClient(app)

    body = client.get("/api/today-slate?date=2026-05-08").json()
    ct_pattern = re.compile(r"^\d{1,2}:\d{2} (AM|PM)$")
    for game in body["games"]:
        assert ct_pattern.match(game["game_time_ct"]), \
            f"bad CT format: {game['game_time_ct']!r} — expected e.g. '2:08 PM'"

Run: pytest tests/test_integration.py::TestTodaysSlateIntegration -m integration -v
If game count is 0 → the schedule file path or date format is wrong.
If lineup_status contains unexpected values → the lineup states file has a different shape.
If has_odds is always False → the odds team name strings don't match schedule team name strings.
```

---

## EPIC 5: Frontend App Shell
**Why:** Establishes the layout container, CSS variables, and the extensible section structure before any real components go in.

**Depends on:** EPIC 1.2

---

### STORY 5.1: App.jsx + index.css

No unit tests. Build and verify visually.

#### TASK 5.1.1 — Global CSS variables and base styles

```
File: frontend/src/index.css

CSS variables to define:
  --color-bg: #f5f3ef;
  --color-surface: #ffffff;
  --color-border: #e2ddd5;
  --color-text-primary: #1a1a18;
  --color-text-muted: #8a8680;
  --color-ok: #4a7c59;       /* muted green */
  --color-ok-bg: #edf4f0;
  --color-stale: #a07040;    /* amber */
  --color-stale-bg: #faf3e8;
  --color-pending: #7a7770;
  --color-confirmed: #4a7c59;
  --font-body: "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", "Fira Code", monospace;

Global resets:
  box-sizing: border-box
  body background, font, color from variables

Verify in browser:
  - Background is warm off-white (#f5f3ef)
  - Body text uses Inter or system font
```

#### TASK 5.1.2 — App.jsx section layout

```
File: frontend/src/App.jsx

/* === WHAT THIS FILE DOES ===
 * App is the root component — it's what React renders into the page.
 * It owns the overall layout: header at top, then sections stacked vertically.
 *
 * WHY STRUCTURE IT THIS WAY:
 * Each section is a self-contained component (PipelineStatus, TodaysSlate).
 * Adding Stage 2 means dropping <ModelPredictions /> between TodaysSlate and
 * the closing div. No other changes needed. This is called "component composition."
 */

Structure:
  <div className="app">
    <header className="app-header">
      <h1>MLB pre-game dashboard</h1>
      <span className="app-date">{today's date formatted}</span>
    </header>
    <main className="app-main">
      <PipelineStatus />
      <TodaysSlate />
      {/* Stage 2: <ModelPredictions /> goes here */}
    </main>
  </div>

Verify in browser:
  - Header shows "MLB pre-game dashboard" and today's date
  - Both sections render (currently placeholders)
  - Warm background, editorial feel
```

---

## EPIC 6: PipelineStatus Component
**Why:** Section 1. Reads from `/api/pipeline-status` and renders three cards.

**Depends on:** EPIC 3 (backend must be running), EPIC 5

---

### STORY 6.1: PipelineStatus.jsx

No unit tests. Build section by section, verify in browser after each sub-step.

#### TASK 6.1.1 — Fetch + loading + error states

```
File: frontend/src/components/PipelineStatus.jsx

/* === HOOKS USED ===
 * useState: holds the fetched data and the loading/error state.
 *   Think of state as "reactive variables" — when they change, React re-renders.
 *
 * useEffect: runs side effects (like fetch calls) after the component mounts.
 *   The empty dependency array [] means "run once when the component appears."
 *   Without it, fetch would run on every render — an infinite loop.
 */

const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  fetch("/api/pipeline-status")
    .then(res => res.json())
    .then(json => { setData(json); setLoading(false); })
    .catch(err => { setError(err.message); setLoading(false); });
}, []);

Render:
  if loading → <div className="section-loading">Loading pipeline status…</div>
  if error   → <div className="section-error">Failed to load: {error}</div>
  else       → render cards (Task 6.1.2)

Verify in browser:
  - Loading state appears while fetch is in-flight (add an artificial delay if needed)
  - Error state appears when backend is killed
  - Cards appear when backend responds
```

#### TASK 6.1.2 — Pipeline card component

```
Render the cards list from data.cards:

/* === PROPS ===
 * PipelineCard receives a `card` prop — a plain JS object with the fields
 * returned by the API: title, run_date, completed_at, duration_seconds,
 * is_stale, games_processed, error.
 *
 * Props are how parent components pass data to child components.
 * Think of them like function arguments.
 */

function PipelineCard({ card }) {
  return (
    <div className={`pipeline-card ${card.is_stale ? "card-stale" : "card-ok"}`}>
      <div className="card-header">
        <span className="card-title">{card.title}</span>
        <span className={`badge ${card.is_stale ? "badge-stale" : "badge-ok"}`}>
          {card.is_stale ? "stale" : "✓ ok"}
        </span>
      </div>
      <div className="card-meta">
        <span className="mono">{card.run_date}</span>
        {" · "}
        <span className="mono">{formatDuration(card.duration_seconds)}s</span>
      </div>
      <ul className="games-processed">
        {Object.entries(card.games_processed).map(([key, count]) => (
          <li key={key}><span className="label">{key}</span> · <span className="mono">{count}</span></li>
        ))}
      </ul>
      {card.error && <div className="card-error">{card.error}</div>}
    </div>
  );
}

CSS: cards side by side with CSS grid (auto-fill, min 240px)

Verify in browser:
  - Three cards render with correct titles
  - Stale card shows amber badge, ok cards show green badge
  - games_processed list renders with label · count format
  - Monospace font on all counts and times
```

#### TASK 6.1.3 — Summary bar

```
Below the cards:
  <div className="pipeline-summary">
    <span>{data.summary.ok_count}/{data.cards.length} ok</span>
    {data.summary.stale_count > 0 && (
      <span className="summary-stale"> · {data.summary.stale_count} stale</span>
    )}
  </div>

Verify in browser:
  - Shows "3/3 ok" when all fresh
  - Shows "2/3 ok · 1 stale" when one is stale
```

---

## EPIC 7: TodaysSlate Component
**Why:** Section 2. Reads from `/api/today-slate` and renders the game table.

**Depends on:** EPIC 4 (backend must be running), EPIC 5

---

### STORY 7.1: TodaysSlate.jsx

No unit tests. Verify visually.

#### TASK 7.1.1 — Fetch + loading + error states

```
File: frontend/src/components/TodaysSlate.jsx

/* Same useState/useEffect pattern as PipelineStatus.
 * Each section fetches independently — if odds data is slow,
 * the pipeline status section is not blocked.
 */

const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  fetch("/api/today-slate")
    .then(res => res.json())
    .then(json => { setData(json); setLoading(false); })
    .catch(err => { setError(err.message); setLoading(false); });
}, []);

Verify: loading/error states render correctly
```

#### TASK 7.1.2 — Game table

```
/* === WHY A TABLE ===
 * Game data is tabular — fixed columns, one row per game.
 * A <table> element is semantic HTML for this use case.
 * It also handles column alignment automatically.
 */

<table className="slate-table">
  <thead>
    <tr>
      <th>Time (CT)</th>
      <th>Matchup</th>
      <th>Lineup</th>
      <th>Odds</th>
      <th>Prediction</th>
    </tr>
  </thead>
  <tbody>
    {data.games.map(game => (
      <tr key={game.game_pk}>
        <td className="mono">{game.game_time_ct}</td>
        <td>{game.away_team} @ {game.home_team}</td>
        <td><LineupBadge status={game.lineup_status} /></td>
        <td>{game.has_odds ? "✓" : "—"}</td>
        <td className="muted">—</td>
      </tr>
    ))}
  </tbody>
</table>

LineupBadge sub-component:
  /* Props: status = "PENDING" | "CONFIRMED" | "SKIPPED"
   * Returns a styled span. The className changes based on the status value.
   * This is the pattern for conditional styling in React.
   */
  function LineupBadge({ status }) {
    const classMap = {
      CONFIRMED: "badge-confirmed",
      PENDING:   "badge-pending",
      SKIPPED:   "badge-skipped",
    };
    return (
      <span className={`badge ${classMap[status] ?? "badge-pending"}`}>
        {status === "SKIPPED" ? <s>{status}</s> : status}
      </span>
    );
  }

Verify in browser:
  - Table renders one row per game
  - Games sorted by CT time
  - CONFIRMED = green badge, PENDING = gray, SKIPPED = muted strikethrough
  - Odds column: ✓ if has_odds, — if not
  - Prediction column: always —
```

---

## Execution Order

| Step | Epic | What it unlocks |
|---|---|---|
| 1 | EPIC 1.1 — Backend scaffold | Backend tests can run |
| 2 | EPIC 1.2 — Frontend scaffold | Browser preview available |
| 3 | EPIC 2 — S3 client | Foundation for all routes |
| 4 | EPIC 3 — Pipeline status endpoint | Section 1 backend done |
| 5 | EPIC 5 — Frontend shell | Layout + CSS in place |
| 6 | EPIC 6 — PipelineStatus component | Section 1 visible end-to-end |
| 7 | EPIC 4 — Today's slate endpoint | Section 2 backend done |
| 8 | EPIC 7 — TodaysSlate component | Section 2 visible end-to-end |

---

## Before marking any task complete:
- [ ] Watched the backend test FAIL before writing any code
- [ ] Test failed for the right reason (no module / no route, not a typo)
- [ ] Wrote minimal code — nothing extra, no extra routes, no extra fields
- [ ] All tests pass after GREEN step (`pytest tests/ -v`)
- [ ] No new warnings or errors in pytest output
- [ ] **For EPICs 2, 3, 4:** integration smoke tests pass (`pytest tests/ -m integration -v`)
- [ ] For frontend tasks: loading state, error state, and happy path all verified in browser
- [ ] React component comments explain WHY each hook is used, not just WHAT it does

---

## Reference: Python TDD Tools for This Project

| Tool | Purpose |
|---|---|
| pytest | Test runner |
| httpx / TestClient | FastAPI test client (comes with fastapi[testing]) |
| unittest.mock.patch | Mock boto3 calls in tests |
| io.BytesIO | Build real Parquet bytes for s3_parquet tests |
| zoneinfo | UTC → Central Time conversion (stdlib, Python 3.9+) |
