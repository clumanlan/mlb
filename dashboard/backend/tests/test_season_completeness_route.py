from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

FAKE_AUDIT_RESULT = {
    "year": "2026",
    "total_scheduled_games": 3,
    "tables": {
        "batter_boxscore": {"missing_count": 0, "missing_games": []},
        "pitcher_boxscore": {"missing_count": 0, "missing_games": []},
        "playbyplay": {"missing_count": 0, "missing_games": []},
    },
    "is_complete": True,
    "checked_at": "2026-08-26T00:00:00+00:00",
}


class TestSeasonCompletenessRoute:
    def test_returns_audit_result(self):
        with patch("main.completeness_audit.run_season_completeness_audit", return_value=FAKE_AUDIT_RESULT) as mock_run:
            response = client.get("/api/season-completeness")

        assert response.status_code == 200
        assert response.json() == FAKE_AUDIT_RESULT
        mock_run.assert_called_once_with("mlbdk", str(date.today().year))

    def test_year_query_param_is_passed_through(self):
        with patch("main.completeness_audit.run_season_completeness_audit", return_value=FAKE_AUDIT_RESULT) as mock_run:
            response = client.get("/api/season-completeness?year=2024")

        assert response.status_code == 200
        mock_run.assert_called_once_with("mlbdk", "2024")
