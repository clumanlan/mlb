import sys
import os
import json
import logging
import pytest
from unittest.mock import patch, MagicMock

HANDLER_DIR = os.path.join(os.path.dirname(__file__), "../src/lambdas/daily_schedule_fetch")

@pytest.fixture(autouse=True)
def handler_path():
    sys.path.insert(0, HANDLER_DIR)
    sys.modules.pop("handler", None)
    yield
    sys.modules.pop("handler", None)
    if HANDLER_DIR in sys.path:
        sys.path.remove(HANDLER_DIR)


FAKE_API_RESPONSE = {
    "dates": [{
        "date": "2026-05-09",
        "games": [{
            "gamePk": 745453,
            "gameDate": "2026-05-09T17:10:00Z",
            "status": {"detailedState": "Preview"},
            "teams": {
                "home": {"team": {"id": 111, "name": "Boston Red Sox"}},
                "away": {"team": {"id": 147, "name": "New York Yankees"}},
            },
            "venue": {"name": "Fenway Park"},
        }]
    }]
}

FAKE_GAMES = [
    {
        "game_pk": 745453,
        "home_team_id": 111,
        "home_team_name": "Boston Red Sox",
        "away_team_id": 147,
        "away_team_name": "New York Yankees",
        "game_time_utc": "2026-05-09T17:10:00Z",
        "venue_name": "Fenway Park",
        "status": "Preview",
    }
]


class TestFetchSchedule:
    def test_fetch_schedule_calls_correct_url_and_params(self):
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_API_RESPONSE

        with patch("handler.requests.get", return_value=mock_response) as mock_get:
            from handler import fetch_schedule
            fetch_schedule("2026-05-09")

        mock_get.assert_called_once_with(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": "2026-05-09", "hydrate": "venue"},
        )

    def test_fetch_schedule_returns_expected_schema(self):
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_API_RESPONSE

        with patch("handler.requests.get", return_value=mock_response):
            from handler import fetch_schedule
            result = fetch_schedule("2026-05-09")

        assert len(result) == 1
        game = result[0]
        assert game["game_pk"] == 745453
        assert game["home_team_id"] == 111
        assert game["home_team_name"] == "Boston Red Sox"
        assert game["away_team_id"] == 147
        assert game["away_team_name"] == "New York Yankees"
        assert game["game_time_utc"] == "2026-05-09T17:10:00Z"
        assert game["venue_name"] == "Fenway Park"
        assert game["status"] == "Preview"

    def test_fetch_schedule_raises_on_http_error(self):
        import requests as req
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError("500 Server Error")

        with patch("handler.requests.get", return_value=mock_response):
            from handler import fetch_schedule
            with pytest.raises(req.HTTPError):
                fetch_schedule("2026-05-09")

    def test_fetch_schedule_returns_empty_list_when_no_dates(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"dates": []}

        with patch("handler.requests.get", return_value=mock_response):
            from handler import fetch_schedule
            result = fetch_schedule("2026-05-09")

        assert result == []


ENV = {"LINEUP_EVENTBRIDGE_RULE_NAME": "mlbdk-daily-lineup-fetch"}


def _mock_clients():
    mock_s3 = MagicMock()
    mock_events = MagicMock()

    def client_factory(service):
        return mock_s3 if service == "s3" else mock_events

    return MagicMock(side_effect=client_factory), mock_s3, mock_events


class TestRun:
    def test_run_writes_schedule_to_correct_s3_path(self):
        mock_boto3, mock_s3, _ = _mock_clients()

        with patch("handler.fetch_schedule", return_value=FAKE_GAMES * 3), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            run("2026-05-09")

        # first put_object call is the schedule file
        call_kwargs = mock_s3.put_object.call_args_list[0].kwargs
        assert call_kwargs["Bucket"] == "mlbdk"
        assert call_kwargs["Key"] == "raw_data/schedule/2026/2026-05-09.json"
        assert call_kwargs["ContentType"] == "application/json"
        body = json.loads(call_kwargs["Body"])
        assert isinstance(body, list)
        assert len(body) == 3

    def test_run_returns_games_processed_dict(self):
        mock_boto3, _, _ = _mock_clients()

        with patch("handler.fetch_schedule", return_value=FAKE_GAMES * 3), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            result = run("2026-05-09")

        assert result == {"scheduled_games": 3}


class TestRunGameStates:
    def test_run_writes_game_states_to_correct_s3_path(self):
        mock_boto3, mock_s3, _ = _mock_clients()

        with patch("handler.fetch_schedule", return_value=FAKE_GAMES), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            run("2026-05-09")

        # second put_object call is the states file
        call_kwargs = mock_s3.put_object.call_args_list[1].kwargs
        assert call_kwargs["Bucket"] == "mlbdk"
        assert call_kwargs["Key"] == "raw_data/games/lineups/states/2026/2026-05-09.json"
        assert call_kwargs["ContentType"] == "application/json"

    def test_run_game_states_structure(self):
        mock_boto3, mock_s3, _ = _mock_clients()

        with patch("handler.fetch_schedule", return_value=FAKE_GAMES), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            run("2026-05-09")

        body = json.loads(mock_s3.put_object.call_args_list[1].kwargs["Body"])
        assert body["date"] == "2026-05-09"
        assert body["last_checked"] is None
        assert body["games"] == {"745453": "PENDING"}

    def test_run_game_states_empty_dict_on_off_day(self):
        mock_boto3, mock_s3, _ = _mock_clients()

        with patch("handler.fetch_schedule", return_value=[]), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            run("2026-05-09")

        body = json.loads(mock_s3.put_object.call_args_list[1].kwargs["Body"])
        assert body["games"] == {}

    def test_run_does_not_write_states_if_schedule_write_fails(self):
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = Exception("S3 unavailable")
        mock_boto3 = MagicMock(side_effect=lambda svc: mock_s3 if svc == "s3" else MagicMock())

        with patch("handler.fetch_schedule", return_value=FAKE_GAMES), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            with pytest.raises(Exception, match="S3 unavailable"):
                run("2026-05-09")

        assert mock_s3.put_object.call_count == 1  # schedule write attempted, states write never reached


class TestRunEnableRule:
    def test_run_enables_lineup_rule_after_s3_write(self):
        mock_boto3, _, mock_events = _mock_clients()

        with patch("handler.fetch_schedule", return_value=FAKE_GAMES), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            run("2026-05-09")

        mock_events.enable_rule.assert_called_once_with(Name="mlbdk-daily-lineup-fetch")

    def test_run_does_not_enable_rule_if_s3_write_fails(self):
        mock_s3 = MagicMock()
        mock_events = MagicMock()
        mock_s3.put_object.side_effect = Exception("S3 unavailable")

        def client_factory(service):
            return mock_s3 if service == "s3" else mock_events

        with patch("handler.fetch_schedule", return_value=FAKE_GAMES), \
             patch("handler.boto3.client", MagicMock(side_effect=client_factory)), \
             patch.dict("os.environ", ENV):
            from handler import run
            with pytest.raises(Exception, match="S3 unavailable"):
                run("2026-05-09")

        mock_events.enable_rule.assert_not_called()

    def test_run_does_not_enable_rule_if_states_write_fails(self):
        mock_s3 = MagicMock()
        mock_events = MagicMock()
        # first put_object (schedule) succeeds, second (states) fails
        mock_s3.put_object.side_effect = [None, Exception("states write failed")]

        def client_factory(service):
            return mock_s3 if service == "s3" else mock_events

        with patch("handler.fetch_schedule", return_value=FAKE_GAMES), \
             patch("handler.boto3.client", MagicMock(side_effect=client_factory)), \
             patch.dict("os.environ", ENV):
            from handler import run
            with pytest.raises(Exception, match="states write failed"):
                run("2026-05-09")

        mock_events.enable_rule.assert_not_called()

    def test_run_does_not_enable_rule_if_fetch_fails(self):
        mock_boto3, _, mock_events = _mock_clients()

        with patch("handler.fetch_schedule", side_effect=Exception("API down")), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            with pytest.raises(Exception, match="API down"):
                run("2026-05-09")

        mock_events.enable_rule.assert_not_called()


class TestRunEdgeCases:
    def test_run_returns_zero_games_on_off_day(self):
        mock_boto3, _, _ = _mock_clients()

        with patch("handler.fetch_schedule", return_value=[]), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            result = run("2026-05-09")

        assert result == {"scheduled_games": 0}

    def test_run_logs_warning_on_off_day(self, caplog):
        mock_boto3, _, _ = _mock_clients()

        with patch("handler.fetch_schedule", return_value=[]), \
             patch("handler.boto3.client", mock_boto3), \
             patch.dict("os.environ", ENV):
            from handler import run
            with caplog.at_level(logging.WARNING):
                run("2026-05-09")

        assert any("No games" in r.message or "off day" in r.message for r in caplog.records)


class TestHandlerEventParsing:
    def test_handler_uses_date_from_event(self):
        with patch("handler.run", return_value={"scheduled_games": 5}) as mock_run, \
             patch("handler.write_status"):
            from handler import handler
            handler({"date": "2026-05-09"}, {})

        mock_run.assert_called_once_with("2026-05-09")

    def test_handler_defaults_to_today_when_no_date(self):
        with patch("handler.run", return_value={"scheduled_games": 0}) as mock_run, \
             patch("handler.datetime") as mock_dt, \
             patch("handler.write_status"):
            mock_dt.utcnow.return_value.strftime.return_value = "2026-05-09"
            mock_dt.utcnow.return_value.isoformat.return_value = "2026-05-09T00:00:00"
            from handler import handler
            handler({}, {})

        mock_run.assert_called_once_with("2026-05-09")

    def test_handler_returns_200_on_success(self):
        with patch("handler.run", return_value={"scheduled_games": 5}), \
             patch("handler.write_status"):
            from handler import handler
            result = handler({"date": "2026-05-09"}, {})

        assert result["statusCode"] == 200

    def test_handler_returns_500_on_exception(self):
        with patch("handler.run", side_effect=Exception("API down")), \
             patch("handler.write_status"):
            from handler import handler
            result = handler({"date": "2026-05-09"}, {})

        assert result["statusCode"] == 500
        assert "API down" in result["body"]


class TestHandlerWriteStatus:
    def test_handler_calls_write_status_on_success(self):
        with patch("handler.run", return_value={"scheduled_games": 5}), \
             patch("handler.write_status") as mock_ws:
            from handler import handler
            handler({"date": "2026-05-09"}, {})

        mock_ws.assert_called_once()
        kw = mock_ws.call_args.kwargs
        assert kw["status"] == "success"
        assert kw["error"] is None
        assert kw["function_name"] == "daily_schedule_fetch"
        assert kw["run_date"] == "2026-05-09"

    def test_handler_calls_write_status_on_failure(self):
        with patch("handler.run", side_effect=RuntimeError("API down")), \
             patch("handler.write_status") as mock_ws:
            from handler import handler
            handler({"date": "2026-05-09"}, {})

        mock_ws.assert_called_once()
        kw = mock_ws.call_args.kwargs
        assert kw["status"] == "failed"
        assert "API down" in kw["error"]

    def test_handler_write_status_games_processed_shape(self):
        with patch("handler.run", return_value={"scheduled_games": 13}), \
             patch("handler.write_status") as mock_ws:
            from handler import handler
            handler({"date": "2026-05-09"}, {})

        gp = mock_ws.call_args.kwargs["games_processed"]
        assert "scheduled_games" in gp
        assert gp["scheduled_games"] == 13
