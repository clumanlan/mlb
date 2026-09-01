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


class TestGetHistoricalGames:
    def test_get_historical_games_calls_historical_endpoint_with_snapshot_date(self):
        fake_envelope = {
            "timestamp": "2025-04-09T17:55:39Z",
            "previous_timestamp": "2025-04-09T17:50:39Z",
            "next_timestamp": "2025-04-09T18:00:39Z",
            "data": [
                {
                    "id": "abc123",
                    "away_team": "St. Louis Cardinals",
                    "home_team": "Pittsburgh Pirates",
                    "commence_time": "2025-04-09T16:35:00Z",
                    "extra_field": "should_be_dropped",
                }
            ],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_envelope
        mock_resp.headers = {"x-requests-remaining": "19990", "x-requests-used": "10"}

        with patch("odds_fetch.requests.get", return_value=mock_resp) as mock_get:
            from odds_fetch import get_historical_games
            result = get_historical_games("2025-04-09", "2025-04-09T18:00:00Z", api_key="test-key")

            call_args = mock_get.call_args
            assert "/v4/historical/sports/baseball_mlb/odds/" in call_args[0][0]
            assert call_args[1]["params"]["date"] == "2025-04-09T18:00:00Z"

            assert len(result) == 1
            game = result[0]
            assert game["id"] == "abc123"
            assert game["away_team"] == "St. Louis Cardinals"
            assert game["home_team"] == "Pittsburgh Pirates"
            assert game["commence_time"] == "2025-04-09T16:35:00Z"
            assert "extra_field" not in game

    def test_get_historical_games_returns_empty_list_when_no_games(self):
        fake_envelope = {"timestamp": "t", "previous_timestamp": None, "next_timestamp": None, "data": []}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_envelope
        mock_resp.headers = {"x-requests-remaining": "19990"}

        with patch("odds_fetch.requests.get", return_value=mock_resp):
            from odds_fetch import get_historical_games
            result = get_historical_games("2025-04-09", "2025-04-09T18:00:00Z", api_key="test-key")
            assert result == []


class TestGetHistoricalPlayerProps:
    def test_get_historical_player_props_requests_pitcher_strikeouts_only(self):
        fake_envelope = {
            "timestamp": "2025-04-09T16:25:39Z",
            "previous_timestamp": "2025-04-09T16:20:39Z",
            "next_timestamp": "2025-04-09T16:30:38Z",
            "data": {
                "id": "abc123",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "pitcher_strikeouts",
                                "outcomes": [
                                    {"name": "Over", "description": "Mitch Keller", "price": -140, "point": 4.5},
                                    {"name": "Under", "description": "Mitch Keller", "price": 110, "point": 4.5},
                                ],
                            }
                        ],
                    }
                ],
            },
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_envelope
        mock_resp.headers = {"x-requests-remaining": "19980", "x-requests-used": "20"}

        with patch("odds_fetch.requests.get", return_value=mock_resp) as mock_get:
            from odds_fetch import get_historical_player_props
            result = get_historical_player_props("abc123", "2025-04-09T16:30:00Z", api_key="test-key")

            call_args = mock_get.call_args
            assert "/v4/historical/sports/baseball_mlb/events/abc123/odds/" in call_args[0][0]
            assert call_args[1]["params"]["markets"] == "pitcher_strikeouts"
            assert call_args[1]["params"]["date"] == "2025-04-09T16:30:00Z"

            assert result["id"] == "abc123"
            assert result["bookmakers"][0]["markets"][0]["key"] == "pitcher_strikeouts"


class TestGetAllHistoricalPlayerProps:
    def test_get_all_historical_player_props_calls_props_per_game(self):
        fake_games = [
            {"id": "game1", "away_team": "Yankees", "home_team": "Red Sox", "commence_time": "2025-04-09T17:05:00Z"},
            {"id": "game2", "away_team": "Cubs", "home_team": "Cardinals", "commence_time": "2025-04-09T19:05:00Z"},
        ]
        fake_props = {"id": "game1", "bookmakers": []}

        with patch("odds_fetch.get_historical_games", return_value=fake_games) as mock_games, \
             patch("odds_fetch.get_historical_player_props", return_value=fake_props) as mock_props:
            from odds_fetch import get_all_historical_player_props
            result = get_all_historical_player_props("2025-04-09", api_key="test-key")

            assert mock_games.call_count == 1
            assert mock_props.call_count == 2
            assert len(result) == 2
            assert result[0]["game"] == fake_games[0]
            assert result[0]["props"] == fake_props
            # each game's own commence_time is used as its props snapshot date --
            # a single fixed daily snapshot would catch early games post-start.
            assert mock_props.call_args_list[0][0][1] == "2025-04-09T17:05:00Z"
            assert mock_props.call_args_list[1][0][1] == "2025-04-09T19:05:00Z"

    def test_get_all_historical_player_props_returns_empty_when_no_games(self):
        with patch("odds_fetch.get_historical_games", return_value=[]):
            from odds_fetch import get_all_historical_player_props
            result = get_all_historical_player_props("2025-04-09", api_key="test-key")
            assert result == []

    def test_get_all_historical_player_props_skips_a_game_that_404s(self, caplog):
        import logging
        fake_games = [
            {"id": "game1", "away_team": "Yankees", "home_team": "Red Sox", "commence_time": "2025-04-09T17:05:00Z"},
            {"id": "game2", "away_team": "Cubs", "home_team": "Cardinals", "commence_time": "2025-04-09T19:05:00Z"},
        ]
        fake_props = {"id": "game2", "bookmakers": []}

        def props_side_effect(event_id, snapshot_date, api_key):
            if event_id == "game1":
                raise Exception("404 Client Error: Not Found")
            return fake_props

        with patch("odds_fetch.get_historical_games", return_value=fake_games), \
             patch("odds_fetch.get_historical_player_props", side_effect=props_side_effect):
            from odds_fetch import get_all_historical_player_props
            with caplog.at_level(logging.WARNING):
                result = get_all_historical_player_props("2025-04-09", api_key="test-key")

            assert len(result) == 1
            assert result[0]["game"] == fake_games[1]
            assert any("game1" in r.message for r in caplog.records)


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

    def test_api_get_captures_requests_used_header(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": "value"}
        mock_resp.headers = {"x-requests-remaining": "13", "x-requests-used": "487"}

        with patch("odds_fetch.requests.get", return_value=mock_resp):
            import odds_fetch
            odds_fetch.api_get("http://example.com", {}, api_key="test-key")
            assert odds_fetch.get_last_usage() == 487

    def test_get_last_usage_returns_none_when_header_missing(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": "value"}
        mock_resp.headers = {}

        with patch("odds_fetch.requests.get", return_value=mock_resp):
            import odds_fetch
            odds_fetch._last_requests_used = None
            odds_fetch.api_get("http://example.com", {}, api_key="test-key")
            assert odds_fetch.get_last_usage() is None
