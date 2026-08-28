import re
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app, utc_to_ct

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEDULE = [
    {
        "game_pk": 745123,
        "home_team_name": "Toronto Blue Jays",
        "away_team_name": "Los Angeles Angels",
        "game_time_utc": "2026-05-09T19:08:00Z",
        "venue_name": "Rogers Centre",
        "status": "Scheduled",
    },
    {
        "game_pk": 745124,
        "home_team_name": "Baltimore Orioles",
        "away_team_name": "Oakland Athletics",
        "game_time_utc": "2026-05-09T20:05:00Z",
        "venue_name": "Oriole Park at Camden Yards",
        "status": "Scheduled",
    },
]

LINEUP_STATES = {
    "date": "2026-05-09",
    "last_checked": "2026-05-09T12:00:00",
    "games": {
        "745123": "PENDING",
        "745124": "CONFIRMED",
    },
}

ODDS_DF = pd.DataFrame([
    {
        "home_team": "Toronto Blue Jays",
        "away_team": "Los Angeles Angels",
        "commence_time": "2026-05-09T19:08:00Z",
        "bookmakers": [],
    }
])

ODDS_DF_WITH_MARKETS = pd.DataFrame([
    {
        "home_team": "Toronto Blue Jays",
        "away_team": "Los Angeles Angels",
        "commence_time": "2026-05-09T19:08:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Los Angeles Angels", "point": 1.5, "price": 120},
                            {"name": "Toronto Blue Jays", "point": -1.5, "price": -142},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 8.5, "price": -110},
                            {"name": "Under", "point": 8.5, "price": -110},
                        ],
                    },
                ],
            }
        ],
    }
])


def _schedule_side_effect(bucket, prefix):
    if "schedule" in prefix:
        return SCHEDULE
    if "lineups" in prefix:
        return LINEUP_STATES
    raise FileNotFoundError(f"unexpected prefix: {prefix}")


# ---------------------------------------------------------------------------
# utc_to_ct unit tests
# ---------------------------------------------------------------------------

class TestUtcToCt:
    def test_converts_19_08_utc_to_2_08_pm_ct(self):
        # 19:08 UTC = 14:08 CDT (UTC-5)
        assert utc_to_ct("2026-05-09T19:08:00Z") == "2:08 PM"

    def test_converts_midnight_utc_to_previous_evening_ct(self):
        # 00:00 UTC = 19:00 CDT previous day
        result = utc_to_ct("2026-05-09T00:00:00Z")
        assert result.endswith("PM")

    def test_output_matches_h_mm_ampm_format(self):
        result = utc_to_ct("2026-05-09T19:08:00Z")
        assert re.match(r"^\d{1,2}:\d{2} (AM|PM)$", result)


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestTodaysSlateRoute:
    def test_returns_games_list(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_schedule_side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no odds")):
            response = client.get("/api/today-slate")

        assert response.status_code == 200
        assert len(response.json()["games"]) == 2

    def test_returns_games_for_specific_date(self):
        def side_effect(bucket, key):
            if "schedule" in key:
                return SCHEDULE
            if "lineups" in key:
                return LINEUP_STATES
            raise FileNotFoundError

        with patch("main.s3_client.get_s3_json", side_effect=side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no odds")):
            response = client.get("/api/today-slate?date=2026-05-08")

        assert response.status_code == 200
        assert response.json()["date"] == "2026-05-08"

    def test_game_has_required_fields(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_schedule_side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no odds")):
            body = client.get("/api/today-slate").json()

        for game in body["games"]:
            for field in ("game_pk", "away_team", "home_team", "game_time_utc",
                          "game_time_ct", "venue", "lineup_status", "has_odds", "odds", "prediction"):
                assert field in game, f"missing field: {field}"

    def test_games_sorted_ascending_by_time(self):
        reversed_schedule = list(reversed(SCHEDULE))

        def side_effect(bucket, prefix):
            if "schedule" in prefix:
                return reversed_schedule
            return LINEUP_STATES

        with patch("main.s3_client.get_latest_s3_json", side_effect=side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no odds")):
            body = client.get("/api/today-slate").json()

        times = [g["game_time_utc"] for g in body["games"]]
        assert times == sorted(times)

    def test_lineup_status_defaults_to_pending_for_unknown_game(self):
        empty_lineups = {}

        def side_effect(bucket, prefix):
            if "schedule" in prefix:
                return SCHEDULE
            return empty_lineups

        with patch("main.s3_client.get_latest_s3_json", side_effect=side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no odds")):
            body = client.get("/api/today-slate").json()

        for game in body["games"]:
            assert game["lineup_status"] == "PENDING"

    def test_has_odds_true_when_teams_match(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_schedule_side_effect), \
             patch("main.s3_client.get_s3_parquet", return_value=ODDS_DF):
            body = client.get("/api/today-slate").json()

        games = {g["game_pk"]: g for g in body["games"]}
        assert games[745123]["has_odds"] is True
        assert games[745124]["has_odds"] is False

    def test_has_odds_false_when_odds_file_missing(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_schedule_side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no odds")):
            body = client.get("/api/today-slate").json()

        assert all(not g["has_odds"] for g in body["games"])

    def test_odds_field_populated_with_run_line_and_total_when_dk_markets_present(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_schedule_side_effect), \
             patch("main.s3_client.get_s3_parquet", return_value=ODDS_DF_WITH_MARKETS):
            body = client.get("/api/today-slate").json()

        games = {g["game_pk"]: g for g in body["games"]}
        assert games[745123]["odds"] == {
            "run_line": {"team": "Toronto Blue Jays", "point": -1.5, "price": -142},
            "total": {"point": 8.5, "over_price": -110, "under_price": -110},
        }
        assert games[745124]["odds"] is None

    def test_odds_field_none_when_odds_row_has_no_markets(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_schedule_side_effect), \
             patch("main.s3_client.get_s3_parquet", return_value=ODDS_DF):
            body = client.get("/api/today-slate").json()

        games = {g["game_pk"]: g for g in body["games"]}
        assert games[745123]["odds"] is None

    def test_odds_field_none_when_odds_file_missing(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_schedule_side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no odds")):
            body = client.get("/api/today-slate").json()

        assert all(g["odds"] is None for g in body["games"])

    def test_prediction_is_always_none(self):
        with patch("main.s3_client.get_latest_s3_json", side_effect=_schedule_side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no odds")):
            body = client.get("/api/today-slate").json()

        for game in body["games"]:
            assert game["prediction"] is None
