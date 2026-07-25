# Implementation Plan: daily_lineup_fetch Lambda

## Baseline
84 tests passing, 0 failures — baseline is clean.

## Layer
Data pulling / ingestion only. Touches: `src/lambdas/daily_lineup_fetch/handler.py`, `tests/test_daily_lineup_fetch_handler.py`, `Makefile`.

---

## Key Design Decisions

- **No model inference** — lineup confirmation writes the file to S3 and stops. Model invocation is a separate Lambda built separately.
- **Pitchers fetched on confirmation only** — when both batting orders are non-empty, we immediately fetch probable pitchers for that game, then build+write the lineup payload. No upfront pitcher fetch for all PENDING games.
- **One pass per invocation** — no sleep/loop inside the Lambda. EventBridge fires every 15 min.

---

## S3 Paths

```
raw_data/schedule/{year}/{date}.json                      ← read by read_schedule
raw_data/lineups/states/{year}/{date}.json                ← read+write each invocation
raw_data/lineups/{year}/{date}/{game_pk}.json             ← written on confirmation
lambdas/status/daily_lineup_fetch/{date}.json             ← written by write_status
```

---

## EPIC 1 — S3 I/O Helpers

**Why:** Every other function depends on S3 reads/writes being correct. Build and test these first.
**Depends on:** none

---

### STORY 1.1: read_schedule
**Acceptance:** `read_schedule(date)` calls s3.get_object with the correct key and returns a parsed list of game dicts.
**Layer:** data pulling

#### TASK 1.1.1

**RED**
- File: `tests/test_daily_lineup_fetch_handler.py`
- Test class: `TestReadSchedule`
- Tests:
  - `test_read_schedule_returns_expected_list` — mock `s3.get_object` to return JSON bytes of 2 fake game dicts; assert result is a list of length 2
  - `test_read_schedule_reads_from_correct_s3_key` — assert `get_object` called with `Bucket="mlbdk"`, `Key="raw_data/schedule/2026/2026-05-09.json"`
- Run: `pytest tests/test_daily_lineup_fetch_handler.py::TestReadSchedule -v` — confirm FAILS (handler doesn't exist)

**GREEN**
- File: `src/lambdas/daily_lineup_fetch/handler.py`
- Implement:
  ```python
  def read_schedule(date):
      s3 = boto3.client("s3")
      year = date[:4]
      obj = s3.get_object(Bucket="mlbdk", Key=f"raw_data/schedule/{year}/{date}.json")
      return json.loads(obj["Body"].read())
  ```
- Run: confirm PASSES

**REFACTOR:** none

---

### STORY 1.2: read_game_states
**Acceptance:** `read_game_states(date)` returns the parsed states dict; raises `ClientError` (from botocore) when the key is missing so the handler can catch it at the top level.
**Layer:** data pulling

#### TASK 1.2.1

**RED**
- Test class: `TestReadGameStates`
- Tests:
  - `test_read_game_states_returns_expected_structure` — mock `get_object` returning valid states JSON; assert keys `date`, `last_checked`, `games` present
  - `test_read_game_states_raises_client_error_on_missing_file` — mock `get_object` raises `ClientError` with code `NoSuchKey`; assert `ClientError` propagates
- Run: confirm FAILS

**GREEN**
- Implement:
  ```python
  def read_game_states(date):
      s3 = boto3.client("s3")
      year = date[:4]
      obj = s3.get_object(Bucket="mlbdk", Key=f"raw_data/lineups/states/{year}/{date}.json")
      return json.loads(obj["Body"].read())
  ```
- Run: confirm PASSES

---

### STORY 1.3: write_game_states
**Acceptance:** `write_game_states(date, states)` puts the correct JSON to the correct key with `ContentType="application/json"`.
**Layer:** data pulling

#### TASK 1.3.1

**RED**
- Test class: `TestWriteGameStates`
- Tests:
  - `test_write_game_states_puts_to_correct_s3_path` — call with fake states dict; assert `put_object` called with `Bucket="mlbdk"`, `Key="raw_data/lineups/states/2026/2026-05-09.json"`, `ContentType="application/json"`, `Body` is valid JSON matching input
- Run: confirm FAILS

**GREEN**
- Implement:
  ```python
  def write_game_states(date, states):
      s3 = boto3.client("s3")
      year = date[:4]
      s3.put_object(
          Bucket="mlbdk",
          Key=f"raw_data/lineups/states/{year}/{date}.json",
          Body=json.dumps(states),
          ContentType="application/json",
      )
  ```
- Run: confirm PASSES

---

### STORY 1.4: write_lineup
**Acceptance:** `write_lineup(date, game_pk, payload)` puts the payload JSON to the correct per-game key; payload has all 9 required fields.
**Layer:** data pulling

#### TASK 1.4.1

**RED**
- Test class: `TestWriteLineup`
- Tests:
  - `test_write_lineup_puts_to_correct_s3_path` — assert `Key="raw_data/lineups/2026/2026-05-09/745453.json"`
  - `test_write_lineup_payload_schema` — build a fake payload; assert it contains keys: `game_pk`, `date`, `confirmed_at`, `home_team_id`, `away_team_id`, `home_lineup`, `away_lineup`, `home_pitcher_id`, `away_pitcher_id`
  - `test_write_lineup_home_lineup_is_list_of_batting_order_dicts` — assert `home_lineup` is a list of `{"batting_order": int, "player_id": int}` dicts
- Run: confirm FAILS

**GREEN**
- Implement:
  ```python
  def write_lineup(date, game_pk, payload):
      s3 = boto3.client("s3")
      year = date[:4]
      s3.put_object(
          Bucket="mlbdk",
          Key=f"raw_data/lineups/{year}/{date}/{game_pk}.json",
          Body=json.dumps(payload),
          ContentType="application/json",
      )
  ```
- Run: confirm PASSES

---

## EPIC 2 — MLB API Calls

**Why:** Two separate API calls with distinct failure modes. `fetch_boxscore` is polled every invocation for PENDING games. `fetch_probable_pitchers` is called only when a game confirms. Tests mock `statsapi.get` — never hit the real API.
**Depends on:** EPIC 1 (for integration into handler later)

---

### STORY 2.1: fetch_boxscore + is_lineup_confirmed
**Acceptance:** `fetch_boxscore` returns the raw boxscore dict; `is_lineup_confirmed` returns True only when both batting order lists are non-empty.

#### TASK 2.1.1 — fetch_boxscore

**RED**
- Test class: `TestFetchBoxscore`
- Tests:
  - `test_fetch_boxscore_returns_teams_dict` — mock `statsapi.get("game_boxscore", {"gamePk": 745453})` returning a dict; assert function returns that dict
- Run: confirm FAILS

**GREEN**
- Implement:
  ```python
  import statsapi

  def fetch_boxscore(game_pk):
      return statsapi.get("game_boxscore", {"gamePk": game_pk})
  ```
- Run: confirm PASSES

#### TASK 2.1.2 — is_lineup_confirmed

**RED**
- Test class: `TestIsLineupConfirmed`
- Tests (pure function — no mocks needed):
  - `test_confirmed_when_both_batting_orders_nonempty` — pass boxscore with `teams.home.battingOrder=[660271]` and `teams.away.battingOrder=[545361]`; assert True
  - `test_not_confirmed_when_home_batting_order_empty` — `home.battingOrder=[]`; assert False
  - `test_not_confirmed_when_away_batting_order_empty` — `away.battingOrder=[]`; assert False
  - `test_not_confirmed_when_both_empty` — both `[]`; assert False
- Run: confirm FAILS

**GREEN**
- Implement:
  ```python
  def is_lineup_confirmed(boxscore):
      home = boxscore["teams"]["home"]["battingOrder"]
      away = boxscore["teams"]["away"]["battingOrder"]
      return bool(home) and bool(away)
  ```
- Run: confirm PASSES

---

### STORY 2.2: fetch_probable_pitchers
**Acceptance:** Returns `{"home": int_or_None, "away": int_or_None}`. Called only after batting orders confirm. Handles missing probable pitchers gracefully.

#### TASK 2.2.1

**RED**
- Test class: `TestFetchProbablePitchers`
- Tests:
  - `test_returns_both_pitcher_ids_when_present` — mock `statsapi.get("game", {"gamePk": 745453})` returning `{"gameData": {"probablePitchers": {"home": {"id": 453286}, "away": {"id": 592789}}}}`; assert returns `{"home": 453286, "away": 592789}`
  - `test_returns_none_for_home_when_home_missing` — `probablePitchers` dict has only `"away"` key; assert `{"home": None, "away": 592789}`
  - `test_returns_none_for_away_when_away_missing` — vice versa
  - `test_returns_none_both_when_probable_pitchers_key_missing` — `gameData` has no `probablePitchers` key; assert `{"home": None, "away": None}`
- Run: confirm FAILS

**GREEN**
- Implement:
  ```python
  def fetch_probable_pitchers(game_pk):
      game = statsapi.get("game", {"gamePk": game_pk})
      pitchers = game.get("gameData", {}).get("probablePitchers", {})
      return {
          "home": pitchers.get("home", {}).get("id"),
          "away": pitchers.get("away", {}).get("id"),
      }
  ```
- Run: confirm PASSES

---

### STORY 2.3: build_lineup_payload
**Acceptance:** Combines confirmed boxscore + fetched pitchers into the canonical lineup JSON structure.

#### TASK 2.3.1

**RED**
- Test class: `TestBuildLineupPayload`
- Tests:
  - `test_build_lineup_payload_returns_correct_schema` — call with fake `(game_pk, date, schedule_game, boxscore, pitchers)` args; assert all 9 keys present
  - `test_build_lineup_payload_batting_order_is_one_based` — assert first entry in `home_lineup` has `batting_order=1`
  - `test_build_lineup_payload_uses_player_ids_from_batting_order` — batting order `[660271, 518692]` → `home_lineup[0]["player_id"] == 660271`
  - `test_build_lineup_payload_pitcher_ids_come_from_pitchers_arg` — assert `home_pitcher_id` and `away_pitcher_id` match the pitchers dict
  - `test_build_lineup_payload_confirmed_at_ends_with_Z` — assert `confirmed_at` string ends with `"Z"`
- Run: confirm FAILS

**GREEN**
- Implement:
  ```python
  def build_lineup_payload(game_pk, date, schedule_game, boxscore, pitchers):
      home_order = boxscore["teams"]["home"]["battingOrder"]
      away_order = boxscore["teams"]["away"]["battingOrder"]
      return {
          "game_pk": game_pk,
          "date": date,
          "confirmed_at": datetime.utcnow().isoformat() + "Z",
          "home_team_id": schedule_game["home_team_id"],
          "away_team_id": schedule_game["away_team_id"],
          "home_lineup": [{"batting_order": i + 1, "player_id": pid} for i, pid in enumerate(home_order)],
          "away_lineup": [{"batting_order": i + 1, "player_id": pid} for i, pid in enumerate(away_order)],
          "home_pitcher_id": pitchers["home"],
          "away_pitcher_id": pitchers["away"],
      }
  ```
- Run: confirm PASSES

---

## EPIC 3 — finish()

**Why:** When all games are resolved, the Lambda must disable its own EventBridge rule and write the final status file. Disable-rule failure must not prevent the status file from writing.
**Depends on:** EPIC 1

---

### STORY 3.1: finish()
**Acceptance:** Disables EventBridge rule by name; writes status file with `scheduled`, `confirmed`, `skipped` counts; still writes status if disable_rule raises.

#### TASK 3.1.1

**RED**
- Test class: `TestFinish`
- Tests:
  - `test_finish_disables_eventbridge_rule_by_name` — mock events client; assert `disable_rule(Name="mlbdk-daily-lineup-fetch")` called
  - `test_finish_calls_write_status_with_correct_counts` — states with 3 CONFIRMED + 2 SKIPPED; assert `write_status` called with `games_processed={"scheduled": 5, "confirmed": 3, "skipped": 2}`
  - `test_finish_still_writes_status_if_disable_rule_raises` — `disable_rule` raises `Exception`; assert `write_status` still called (not re-raised)
  - `test_finish_write_status_function_name_is_correct` — assert `function_name="daily_lineup_fetch"`
- Run: confirm FAILS

**GREEN**
- Implement:
  ```python
  def finish(date, states):
      try:
          events_client = boto3.client("events")
          events_client.disable_rule(Name=os.environ["LINEUP_EVENTBRIDGE_RULE_NAME"])
          logger.info("lineup polling rule disabled — all games resolved")
      except Exception as e:
          logger.error(f"failed to disable EventBridge rule: {e}")

      confirmed = sum(1 for s in states["games"].values() if s == "CONFIRMED")
      skipped = sum(1 for s in states["games"].values() if s == "SKIPPED")
      write_status(
          function_name="daily_lineup_fetch",
          run_date=date,
          status="success",
          started_at=states.get("_started_at", datetime.utcnow().isoformat()),
          completed_at=datetime.utcnow().isoformat(),
          duration_seconds=0,
          games_processed={
              "scheduled": len(states["games"]),
              "confirmed": confirmed,
              "skipped": skipped,
          },
          error=None,
      )
  ```
- Run: confirm PASSES

---

## EPIC 4 — Main Handler Logic

**Why:** Orchestrates all the above. Each test scenario exercises one behaviour of the handler — the tests together describe the complete algorithm.
**Depends on:** EPICs 1, 2, 3

---

### STORY 4.1: Event parsing
**Acceptance:** Uses `date` from event; defaults to today's UTC date when absent.

#### TASK 4.1.1

**RED**
- Test class: `TestHandlerEventParsing`
- Tests:
  - `test_handler_uses_date_from_event`
  - `test_handler_defaults_to_today_when_no_date` — freeze `datetime.utcnow` via `unittest.mock.patch`
- Run: confirm FAILS

**GREEN**
- Implement handler skeleton:
  ```python
  def handler(event, context):
      date = event.get("date") or datetime.utcnow().strftime("%Y-%m-%d")
      started_at = datetime.utcnow()
      ...
  ```
- Run: confirm PASSES

---

### STORY 4.2: Missing game_states file
**Acceptance:** When `read_game_states` raises `ClientError`, handler logs the error, writes a failure status file, and returns 500 — does not proceed to poll boxscores.

#### TASK 4.2.1

**RED**
- Test class: `TestHandlerMissingStates`
- Tests:
  - `test_handler_returns_500_when_game_states_missing` — mock `read_game_states` raises `ClientError`; assert 500
  - `test_handler_writes_failure_status_when_game_states_missing` — assert `write_status` called with `status="failed"`
  - `test_handler_does_not_call_fetch_boxscore_when_states_missing` — assert `fetch_boxscore` never called
- Run: confirm FAILS

**GREEN**
- Wrap `read_game_states` in `try/except ClientError` at the top of the handler body.
- Run: confirm PASSES

---

### STORY 4.3: Hard cutoff gate
**Acceptance:** When `utcnow >= cutoff`, all PENDING games become SKIPPED, `finish()` is called, and no boxscore calls are made.

#### TASK 4.3.1

**RED**
- Test class: `TestHandlerHardCutoff`
- Tests:
  - `test_handler_marks_pending_as_skipped_when_cutoff_reached` — freeze time past cutoff; assert `write_game_states` called with all games SKIPPED
  - `test_handler_calls_finish_when_cutoff_reached` — assert `finish` called after cutoff
  - `test_handler_does_not_poll_boxscores_when_cutoff_reached` — assert `fetch_boxscore` never called
- Run: confirm FAILS

**GREEN**
- Implement cutoff check before the per-game loop:
  ```python
  first_pitch_utc = min(
      datetime.strptime(g["game_time_utc"], "%Y-%m-%dT%H:%M:%SZ")
      for g in schedule
  )
  cutoff = first_pitch_utc - timedelta(minutes=HARD_CUTOFF_MINUTES)
  if datetime.utcnow() >= cutoff:
      for game_pk, state in states["games"].items():
          if state == "PENDING":
              states["games"][game_pk] = "SKIPPED"
              logger.info(f"cutoff reached, skipping: {game_pk}")
      write_game_states(date, states)
      finish(date, states)
      return {"statusCode": 200, "body": "cutoff reached — all pending games skipped"}
  ```
- Run: confirm PASSES

---

### STORY 4.4: Per-game polling loop
**Acceptance:** Only PENDING games are polled. On confirmation: pitchers fetched, payload built, lineup written, state set to CONFIRMED. Boxscore failure leaves game PENDING for next invocation.

#### TASK 4.4.1

**RED**
- Test class: `TestHandlerPollingLoop`
- Tests:
  - `test_handler_skips_already_confirmed_games` — states has one CONFIRMED game; assert `fetch_boxscore` not called for it
  - `test_handler_skips_already_skipped_games` — states has one SKIPPED; same assertion
  - `test_handler_confirms_game_when_both_batting_orders_nonempty` — `fetch_boxscore` returns non-empty orders; assert `states["games"]["745453"] == "CONFIRMED"` written to S3
  - `test_handler_fetches_pitchers_only_after_batting_order_confirms` — assert `fetch_probable_pitchers` called only when `is_lineup_confirmed` returns True
  - `test_handler_writes_lineup_on_confirmation` — assert `write_lineup` called with correct `game_pk`
  - `test_handler_leaves_game_pending_when_boxscore_raises` — `fetch_boxscore` raises; assert game stays PENDING in written states
  - `test_handler_leaves_game_pending_when_batting_orders_empty` — boxscore returns empty lists; assert game stays PENDING
- Run: confirm FAILS

**GREEN**
- Implement the per-game loop:
  ```python
  schedule_by_pk = {str(g["game_pk"]): g for g in schedule}
  for game_pk, state in states["games"].items():
      if state != "PENDING":
          continue
      try:
          boxscore = fetch_boxscore(game_pk)
          if is_lineup_confirmed(boxscore):
              pitchers = fetch_probable_pitchers(game_pk)
              payload = build_lineup_payload(
                  int(game_pk), date, schedule_by_pk[game_pk], boxscore, pitchers
              )
              write_lineup(date, game_pk, payload)
              states["games"][game_pk] = "CONFIRMED"
              logger.info(f"lineup confirmed: {game_pk}")
      except Exception as e:
          logger.warning(f"boxscore fetch failed for {game_pk}: {e}")
  ```
- Run: confirm PASSES

---

### STORY 4.5: State persistence + finish trigger
**Acceptance:** `last_checked` is updated every invocation. `finish()` called when no PENDING games remain.

#### TASK 4.5.1

**RED**
- Test class: `TestHandlerStatePersistence`
- Tests:
  - `test_handler_updates_last_checked_timestamp` — assert `write_game_states` called with non-None `last_checked`
  - `test_handler_calls_finish_when_all_games_confirmed` — all CONFIRMED; assert `finish` called
  - `test_handler_calls_finish_when_all_games_skipped_or_confirmed` — mix of CONFIRMED + SKIPPED; assert `finish` called
  - `test_handler_does_not_call_finish_when_games_still_pending` — one PENDING remains; assert `finish` not called
- Run: confirm FAILS

**GREEN**
- After the loop:
  ```python
  states["last_checked"] = datetime.utcnow().isoformat()
  write_game_states(date, states)
  if all(s != "PENDING" for s in states["games"].values()):
      finish(date, states)
  ```
- Run: confirm PASSES

---

### STORY 4.6: Handler write_status + HTTP responses
**Acceptance:** Returns 200 on normal exit; returns 500 and writes failure status on unrecoverable exception.

#### TASK 4.6.1

**RED**
- Test class: `TestHandlerWriteStatus`
- Tests:
  - `test_handler_returns_200_on_success`
  - `test_handler_returns_500_on_unrecoverable_exception` — e.g. `write_game_states` raises unexpectedly
  - `test_handler_writes_failure_status_on_unrecoverable_exception` — assert `write_status` called with `status="failed"` and error text
  - `test_handler_write_status_games_processed_shape` — on success, `games_processed` has keys `scheduled`, `confirmed`, `skipped`
- Run: confirm FAILS

**GREEN**
- Wrap the full handler body in `try/except Exception`:
  ```python
  except Exception as e:
      completed_at = datetime.utcnow()
      write_status(
          function_name="daily_lineup_fetch",
          run_date=date,
          status="failed",
          started_at=started_at.isoformat(),
          completed_at=completed_at.isoformat(),
          duration_seconds=(completed_at - started_at).total_seconds(),
          games_processed={},
          error=str(e),
      )
      return {"statusCode": 500, "body": str(e)}
  ```
- Run: confirm PASSES

---

## EPIC 5 — Makefile Build Target

**Why:** Lambda cannot be deployed without a zip rule.
**Depends on:** EPIC 4

### TASK 5.1.1

**RED (manual check)**
- Run: `make zip-lineup-fetch` — confirm FAILS with "No rule to make target"

**GREEN**
- Add to `Makefile`:
  ```makefile
  zip-lineup-fetch:
  	mkdir -p $(BUILD_DIR)
  	cp src/lambdas/daily_lineup_fetch/handler.py $(BUILD_DIR)/
  	cp src/shared/*.py $(BUILD_DIR)/
  	cd $(BUILD_DIR) && zip daily_lineup_fetch.zip *.py
  	rm $(BUILD_DIR)/*.py

  deploy-lineup-fetch: zip-lineup-fetch
  	aws lambda update-function-code \
  	  --region $(REGION) \
  	  --function-name daily_lineup_fetch \
  	  --zip-file fileb://$(BUILD_DIR)/daily_lineup_fetch.zip
  ```
- Run: `make zip-lineup-fetch` — confirm zip created

---

## Test File Structure

```python
# tests/test_daily_lineup_fetch_handler.py
import sys, os, json, logging, pytest
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError

HANDLER_DIR = os.path.join(os.path.dirname(__file__), "../src/lambdas/daily_lineup_fetch")

@pytest.fixture(autouse=True)
def handler_path():
    sys.path.insert(0, HANDLER_DIR)
    sys.modules.pop("handler", None)
    yield
    sys.modules.pop("handler", None)
    if HANDLER_DIR in sys.path:
        sys.path.remove(HANDLER_DIR)
```

**Fake data constants** (define at top of test file for reuse):
```python
FAKE_SCHEDULE = [
    {
        "game_pk": 745453,
        "home_team_id": 111,
        "away_team_id": 147,
        "game_time_utc": "2026-05-09T17:10:00Z",
    }
]

FAKE_STATES = {
    "date": "2026-05-09",
    "last_checked": None,
    "games": {"745453": "PENDING"},
}

FAKE_BOXSCORE_CONFIRMED = {
    "teams": {
        "home": {"battingOrder": [660271, 518692, 543257]},
        "away": {"battingOrder": [545361, 596019, 608369]},
    }
}

FAKE_BOXSCORE_EMPTY = {
    "teams": {
        "home": {"battingOrder": []},
        "away": {"battingOrder": []},
    }
}

FAKE_PITCHERS = {"home": 453286, "away": 592789}
```

---

## Handler File Layout

```
src/lambdas/daily_lineup_fetch/
└── handler.py   ← single file; status_writer.py bundled via make zip-lineup-fetch
```

Top of `handler.py`:
```python
import json
import logging
import os
import boto3
import statsapi
from datetime import datetime, timedelta
from status_writer import write_status

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

S3_BUCKET = "mlbdk"
HARD_CUTOFF_MINUTES = int(os.environ.get("HARD_CUTOFF_MINUTES", 30))
```

---

## IAM Permissions Required (document in handler comment)

```
s3:GetObject       — read schedule + game_states
s3:PutObject       — write lineups + game_states + status file
events:DisableRule — disable own EventBridge rule on completion
```

---

## Before marking any task complete:
- [ ] Watched the test FAIL before writing any code
- [ ] Test failed for the right reason (feature missing, not a typo)
- [ ] Wrote minimal code — nothing extra
- [ ] All 84 existing tests still pass after GREEN
- [ ] No new warnings or errors in pytest output
- [ ] Mocks used only for I/O boundaries (statsapi.get, boto3.client)
