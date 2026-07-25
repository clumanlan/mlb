from datetime import date
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app, is_stale_mlb_fetch, is_stale_odds_fetch

client = TestClient(app)

# ---------------------------------------------------------------------------
# Staleness logic
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_mlb_fetch_stale_when_run_date_not_yesterday(self):
        assert is_stale_mlb_fetch("2026-05-06", date(2026, 5, 9)) is True

    def test_mlb_fetch_not_stale_when_run_date_is_yesterday(self):
        assert is_stale_mlb_fetch("2026-05-08", date(2026, 5, 9)) is False

    def test_odds_fetch_stale_when_run_date_not_today(self):
        assert is_stale_odds_fetch("2026-05-08", date(2026, 5, 9)) is True

    def test_odds_fetch_not_stale_when_run_date_is_today(self):
        assert is_stale_odds_fetch("2026-05-09", date(2026, 5, 9)) is False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MLB_FETCH_STATUS = {
    "function_name": "daily_mlb_fetch",
    "run_date": "2026-05-08",
    "status": "success",
    "started_at": "2026-05-08T10:00:00.000000",
    "completed_at": "2026-05-08T10:01:38.701783",
    "duration_seconds": 55.96,
    "games_processed": {
        "schedule": 15, "game_info": 15, "batter_boxscore": 15,
        "pitcher_boxscore": 15, "playbyplay": 15,
    },
    "error": None,
}

PROCESS_DATA_STATUS = {
    "function_name": "daily_process_data",
    "run_date": "2026-05-08",
    "status": "success",
    "started_at": "2026-05-08T10:02:00.000000",
    "completed_at": "2026-05-08T10:02:01.446648",
    "duration_seconds": 1.45,
    "games_processed": {
        "schedule": 15, "batter_prepared": 15,
        "pitcher_prepared": 15, "playbyplay_prepared": 15,
    },
    "error": None,
}

ODDS_FETCH_STATUS = {
    "function_name": "daily_odds_fetch",
    "run_date": "2026-05-09",
    "status": "success",
    "started_at": "2026-05-09T10:00:00.000000",
    "completed_at": "2026-05-09T10:00:09.179323",
    "duration_seconds": 9.18,
    "games_processed": {"team_odds": 13, "player_props": 13},
    "error": None,
}


def _mock_s3_side_effect(bucket, prefix):
    if "daily_mlb_fetch" in prefix:
        return MLB_FETCH_STATUS
    if "daily_process_data" in prefix:
        return PROCESS_DATA_STATUS
    if "daily_odds_fetch" in prefix:
        return ODDS_FETCH_STATUS
    raise FileNotFoundError(f"unexpected prefix: {prefix}")


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestPipelineStatusRoute:
    def test_returns_three_cards(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_mock_s3_side_effect), \
             patch("main.date") as mock_date:
            mock_date.today.return_value = date(2026, 5, 9)
            mock_date.fromisoformat.side_effect = date.fromisoformat
            response = client.get("/api/pipeline-status")

        assert response.status_code == 200
        assert len(response.json()["cards"]) == 3

    def test_card_has_required_fields(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_mock_s3_side_effect), \
             patch("main.date") as mock_date:
            mock_date.today.return_value = date(2026, 5, 9)
            mock_date.fromisoformat.side_effect = date.fromisoformat
            body = client.get("/api/pipeline-status").json()

        for card in body["cards"]:
            for field in ("title", "run_date", "completed_at", "duration_seconds",
                          "is_stale", "games_processed", "error"):
                assert field in card, f"card '{card.get('title')}' missing field: {field}"

    def test_summary_counts_one_stale(self):
        # mlb_fetch and process_data ran 2026-05-06 (stale relative to today 2026-05-09, yesterday=2026-05-08)
        stale_mlb = {**MLB_FETCH_STATUS, "run_date": "2026-05-06"}
        stale_process = {**PROCESS_DATA_STATUS, "run_date": "2026-05-06"}

        def stale_side_effect(bucket, prefix):
            if "daily_mlb_fetch" in prefix:
                return stale_mlb
            if "daily_process_data" in prefix:
                return stale_process
            return ODDS_FETCH_STATUS

        with patch("main.s3_client.get_latest_s3_json", side_effect=stale_side_effect), \
             patch("main.date") as mock_date:
            mock_date.today.return_value = date(2026, 5, 9)
            mock_date.fromisoformat.side_effect = date.fromisoformat
            body = client.get("/api/pipeline-status").json()

        assert body["summary"]["stale_count"] == 2
        assert body["summary"]["ok_count"] == 1

    def test_missing_s3_file_surfaces_as_card_error(self):
        def missing_side_effect(bucket, prefix):
            if "daily_mlb_fetch" in prefix:
                raise FileNotFoundError("no file")
            if "daily_process_data" in prefix:
                return PROCESS_DATA_STATUS
            return ODDS_FETCH_STATUS

        with patch("main.s3_client.get_latest_s3_json", side_effect=missing_side_effect), \
             patch("main.date") as mock_date:
            mock_date.today.return_value = date(2026, 5, 9)
            mock_date.fromisoformat.side_effect = date.fromisoformat
            response = client.get("/api/pipeline-status")

        assert response.status_code == 200
        cards = {c["title"]: c for c in response.json()["cards"]}
        assert cards["Raw — MLB fetch"]["error"] is not None
