from unittest.mock import patch, MagicMock
import pytest


def make_response(status_code):
    mock = MagicMock()
    mock.status_code = status_code
    mock.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return mock


def make_game_response(games):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = games
    mock.headers = {"x-requests-remaining": "490"}
    return mock


class TestGetAllPlayerProps:
    def test_get_all_player_props_calls_get_player_props_per_game(self):
        fake_games = [
            {"id": "game1", "away_team": "Yankees", "home_team": "Red Sox", "commence_time": "2026-04-29T17:05:00Z"},
            {"id": "game2", "away_team": "Cubs", "home_team": "Cardinals", "commence_time": "2026-04-29T19:05:00Z"},
        ]
        fake_props = {"id": "game1", "bookmakers": []}

        with patch("odds_fetch.get_games", return_value=fake_games) as mock_games, \
             patch("odds_fetch.get_player_props", return_value=fake_props) as mock_props:
            from odds_fetch import get_all_player_props
            result = get_all_player_props("2026-04-29", api_key="test-key")
            assert mock_games.call_count == 1
            assert mock_props.call_count == 2
            assert len(result) == 2
            assert result[0]["game"] == fake_games[0]
            assert result[0]["props"] == fake_props

    def test_get_all_player_props_returns_empty_when_no_games(self):
        with patch("odds_fetch.get_games", return_value=[]):
            from odds_fetch import get_all_player_props
            result = get_all_player_props("2026-04-29", api_key="test-key")
            assert result == []


class TestGetTeamOdds:
    def test_get_team_odds_returns_raw_api_response(self):
        fake_team_odds = [
            {
                "id": "game1",
                "home_team": "Boston Red Sox",
                "away_team": "New York Yankees",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {"key": "spreads", "outcomes": []},
                            {"key": "totals", "outcomes": []},
                        ],
                    }
                ],
            }
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_team_odds
        mock_resp.headers = {"x-requests-remaining": "489"}

        with patch("odds_fetch.requests.get", return_value=mock_resp):
            from odds_fetch import get_team_odds
            result = get_team_odds("2026-04-29", api_key="test-key")
            assert isinstance(result, list)
            assert result[0]["id"] == "game1"
            assert "bookmakers" in result[0]

    def test_get_team_odds_returns_empty_list_when_no_games(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.headers = {"x-requests-remaining": "489"}

        with patch("odds_fetch.requests.get", return_value=mock_resp):
            from odds_fetch import get_team_odds
            result = get_team_odds("2026-04-29", api_key="test-key")
            assert result == []


class TestGetGames:
    def test_get_games_returns_game_schema(self):
        fake_games = [
            {
                "id": "abc123",
                "away_team": "New York Yankees",
                "home_team": "Boston Red Sox",
                "commence_time": "2026-04-29T17:05:00Z",
                "extra_field": "should_be_dropped",
            }
        ]
        with patch("odds_fetch.requests.get", return_value=make_game_response(fake_games)):
            from odds_fetch import get_games
            result = get_games("2026-04-29", api_key="test-key")
            assert len(result) == 1
            game = result[0]
            assert game["id"] == "abc123"
            assert game["away_team"] == "New York Yankees"
            assert game["home_team"] == "Boston Red Sox"
            assert game["commence_time"] == "2026-04-29T17:05:00Z"
            assert "extra_field" not in game

    def test_get_games_returns_empty_list_when_no_games(self):
        with patch("odds_fetch.requests.get", return_value=make_game_response([])):
            from odds_fetch import get_games
            result = get_games("2026-04-29", api_key="test-key")
            assert result == []


class TestApiGet:
    def test_api_get_raises_after_max_retries(self):
        with patch("odds_fetch.requests.get", return_value=make_response(500)):
            from odds_fetch import api_get
            with pytest.raises(Exception):
                api_get("http://example.com", {}, api_key="test-key", retries=3, delay=0)

    def test_api_get_logs_requests_remaining(self, caplog):
        import logging
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.headers = {"x-requests-remaining": "490"}

        with patch("odds_fetch.requests.get", return_value=mock_resp):
            from odds_fetch import api_get
            with caplog.at_level(logging.INFO):
                api_get("http://example.com", {}, api_key="test-key")

        assert any("490" in r.message for r in caplog.records)

    def test_api_get_logs_rate_limit_warning(self, caplog):
        import logging
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {}
        mock_200.headers = {"x-requests-remaining": "489"}

        with patch("odds_fetch.requests.get", side_effect=[mock_429, mock_200]):
            from odds_fetch import api_get
            with caplog.at_level(logging.WARNING):
                api_get("http://example.com", {}, api_key="test-key", delay=0)

        assert any("rate limit" in r.message.lower() for r in caplog.records)

    def test_api_get_returns_json_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": "value"}
        mock_resp.headers = {"x-requests-remaining": "490"}

        with patch("odds_fetch.requests.get", return_value=mock_resp):
            from odds_fetch import api_get
            result = api_get("http://example.com", {}, api_key="test-key")
            assert result == {"data": "value"}
