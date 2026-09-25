from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SCHEDULE = [
    {
        "game_pk": 745123,
        "home_team_name": "Toronto Blue Jays",
        "away_team_name": "Los Angeles Angels",
        "game_time_utc": "2026-05-09T19:08:00Z",
        "venue_name": "Rogers Centre",
        "status": "Scheduled",
    },
]


class TestStartingPitcherPredictionsRoute:
    def test_returns_200_with_two_pitchers_for_the_days_one_game(self):
        with patch("main._load_schedule", return_value=SCHEDULE), \
             patch("main.s3_client.get_s3_json", side_effect=FileNotFoundError("no lineup")), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no player info")):
            response = client.get("/api/starting-pitcher-predictions")

        assert response.status_code == 200
        body = response.json()
        assert len(body["pitchers"]) == 2
        assert {p["game_pk"] for p in body["pitchers"]} == {745123}

    def test_flags_response_as_having_no_real_predictions(self):
        with patch("main._load_schedule", return_value=SCHEDULE), \
             patch("main.s3_client.get_s3_json", side_effect=FileNotFoundError("no lineup")), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no player info")):
            body = client.get("/api/starting-pitcher-predictions").json()

        assert body["has_predictions"] is False

    def test_no_games_today_returns_empty_pitchers_list(self):
        with patch("main._load_schedule", return_value=[]), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no player info")):
            body = client.get("/api/starting-pitcher-predictions").json()

        assert body["pitchers"] == []

    def test_pitcher_name_resolved_for_confirmed_game(self):
        lineup = {"home_pitcher_id": 501, "away_pitcher_id": 502}
        player_info_df = pd.DataFrame([
            {"person_id": 501, "player_name": "Home Starter"},
            {"person_id": 502, "player_name": "Away Starter"},
        ])

        def json_side_effect(bucket, key):
            if "lineups" in key:
                return lineup
            raise FileNotFoundError(f"not mocked: {key}")

        def parquet_side_effect(bucket, key):
            if "player_info" in key:
                return player_info_df
            raise FileNotFoundError(f"not mocked: {key}")

        with patch("main._load_schedule", return_value=SCHEDULE), \
             patch("main.s3_client.get_s3_json", side_effect=json_side_effect), \
             patch("main.s3_client.get_s3_parquet", side_effect=parquet_side_effect):
            body = client.get("/api/starting-pitcher-predictions").json()

        names = {p["team"]: p["pitcher_name"] for p in body["pitchers"]}
        assert names["Toronto Blue Jays"] == "Home Starter"
        assert names["Los Angeles Angels"] == "Away Starter"

    def test_pitcher_name_is_tbd_when_lineup_file_missing(self):
        with patch("main._load_schedule", return_value=SCHEDULE), \
             patch("main.s3_client.get_s3_json", side_effect=FileNotFoundError("no lineup")), \
             patch("main.s3_client.get_s3_parquet", side_effect=FileNotFoundError("no player info")):
            body = client.get("/api/starting-pitcher-predictions").json()

        assert all(p["pitcher_name"] == "TBD" for p in body["pitchers"])
