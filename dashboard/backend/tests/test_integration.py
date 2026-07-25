"""
Integration smoke tests — hit real S3 for a given date.

Run with:
  pytest tests/test_integration.py -m integration -v                     # uses yesterday
  pytest tests/test_integration.py -m integration -v --date 2026-05-08  # specific date

Requires AWS credentials in backend/.env or shell environment.
"""
import re

import pytest
from fastapi.testclient import TestClient

import s3_client
from main import app

client = TestClient(app)


@pytest.mark.integration
class TestS3ClientIntegration:
    def test_reads_schedule_parquet_for_date(self, run_date):
        """Reads the legacy schedule Parquet (daily_mlb_fetch path) for the given date."""
        year = run_date[:4]
        df = s3_client.get_s3_parquet("mlbdk", f"raw_data/games/schedule/{year}/schedule_{run_date}.parquet")
        assert not df.empty, f"expected rows in schedule Parquet for {run_date}"
        for col in ("game_id", "home_name", "away_name", "game_datetime", "venue_name"):
            assert col in df.columns, f"missing column: {col}"

    def test_reads_team_odds_parquet_for_date(self, run_date):
        """Reads team odds Parquet for the given date."""
        year = run_date[:4]
        df = s3_client.get_s3_parquet("mlbdk", f"raw_data/odds/team_odds/{year}/{run_date}.parquet")
        assert not df.empty, f"expected at least one odds row for {run_date}"
        for col in ("home_team", "away_team"):
            assert col in df.columns, f"missing column: {col}"


@pytest.mark.integration
class TestPipelineStatusIntegration:
    def test_returns_three_cards_with_correct_shape(self):
        response = client.get("/api/pipeline-status")
        assert response.status_code == 200
        body = response.json()
        assert len(body["cards"]) == 3
        assert "summary" in body
        for card in body["cards"]:
            for field in ("title", "run_date", "duration_seconds", "is_stale", "games_processed", "error"):
                assert field in card, f"card missing field: {field}"

    def test_mlb_fetch_card_has_correct_games_processed_keys(self):
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


@pytest.mark.integration
class TestTodaysSlateIntegration:
    def test_returns_games_for_date(self, run_date):
        response = client.get(f"/api/today-slate?date={run_date}")
        assert response.status_code == 200
        body = response.json()
        assert body["date"] == run_date
        assert isinstance(body["games"], list)
        assert len(body["games"]) > 0, f"expected games on {run_date}"

    def test_all_games_have_required_fields(self, run_date):
        body = client.get(f"/api/today-slate?date={run_date}").json()
        for game in body["games"]:
            for field in ("game_pk", "away_team", "home_team", "game_time_utc",
                          "game_time_ct", "venue", "lineup_status", "has_odds", "prediction"):
                assert field in game, f"game missing field: {field}"
            assert game["lineup_status"] in ("PENDING", "CONFIRMED", "SKIPPED")
            assert isinstance(game["has_odds"], bool)
            assert game["prediction"] is None

    def test_games_sorted_ascending_by_time(self, run_date):
        body = client.get(f"/api/today-slate?date={run_date}").json()
        times = [g["game_time_utc"] for g in body["games"]]
        assert times == sorted(times), "games must be sorted ascending by UTC time"

    def test_ct_time_format_is_valid(self, run_date):
        body = client.get(f"/api/today-slate?date={run_date}").json()
        ct_pattern = re.compile(r"^\d{1,2}:\d{2} (AM|PM)$")
        for game in body["games"]:
            assert ct_pattern.match(game["game_time_ct"]), \
                f"bad CT format: {game['game_time_ct']!r}"
