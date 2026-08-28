from unittest.mock import patch

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
        with patch("main._load_schedule", return_value=SCHEDULE):
            response = client.get("/api/starting-pitcher-predictions")

        assert response.status_code == 200
        body = response.json()
        assert len(body["pitchers"]) == 2
        assert {p["game_pk"] for p in body["pitchers"]} == {745123}

    def test_flags_response_as_sample_data(self):
        with patch("main._load_schedule", return_value=SCHEDULE):
            body = client.get("/api/starting-pitcher-predictions").json()

        assert body["is_sample_data"] is True

    def test_no_games_today_returns_empty_pitchers_list(self):
        with patch("main._load_schedule", return_value=[]):
            body = client.get("/api/starting-pitcher-predictions").json()

        assert body["pitchers"] == []
