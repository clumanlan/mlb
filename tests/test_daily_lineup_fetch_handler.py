import sys
import os
import json
import pytest
from datetime import datetime
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


# --- Shared fake data ---

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

# First pitch in the past — cutoff always reached in any real test run
PAST_SCHEDULE = [
    {
        "game_pk": 745453,
        "home_team_id": 111,
        "away_team_id": 147,
        "game_time_utc": "2000-01-01T00:00:00Z",
    }
]

# First pitch far in the future — cutoff never reached
FUTURE_SCHEDULE = [
    {
        "game_pk": 745453,
        "home_team_id": 111,
        "away_team_id": 147,
        "game_time_utc": "2099-12-31T23:59:00Z",
    }
]

ENV = {
    "LINEUP_EVENTBRIDGE_RULE_NAME": "mlbdk-daily-lineup-fetch",
    "HARD_CUTOFF_MINUTES": "30",
}


def _client_error(code="NoSuchKey"):
    return ClientError({"Error": {"Code": code, "Message": "Not found"}}, "GetObject")


def _make_s3_mock(body_data=None):
    mock_s3 = MagicMock()
    if body_data is not None:
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(body_data).encode()
        mock_s3.get_object.return_value = {"Body": mock_body}
    return mock_s3


# =============================================================================
# EPIC 1: S3 I/O Helpers
# =============================================================================

class TestReadSchedule:
    def test_read_schedule_returns_expected_list(self):
        mock_s3 = _make_s3_mock(FAKE_SCHEDULE)
        with patch("handler.boto3.client", return_value=mock_s3):
            from handler import read_schedule
            result = read_schedule("2026-05-09")
        assert len(result) == 1
        assert result[0]["game_pk"] == 745453

    def test_read_schedule_reads_from_correct_s3_key(self):
        mock_s3 = _make_s3_mock(FAKE_SCHEDULE)
        with patch("handler.boto3.client", return_value=mock_s3):
            from handler import read_schedule
            read_schedule("2026-05-09")
        mock_s3.get_object.assert_called_once_with(
            Bucket="mlbdk", Key="raw_data/schedule/2026/2026-05-09.json"
        )


class TestReadGameStates:
    def test_read_game_states_returns_expected_structure(self):
        mock_s3 = _make_s3_mock(FAKE_STATES)
        with patch("handler.boto3.client", return_value=mock_s3):
            from handler import read_game_states
            result = read_game_states("2026-05-09")
        assert result["date"] == "2026-05-09"
        assert result["last_checked"] is None
        assert "745453" in result["games"]

    def test_read_game_states_raises_client_error_on_missing_file(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _client_error("NoSuchKey")
        with patch("handler.boto3.client", return_value=mock_s3):
            from handler import read_game_states
            with pytest.raises(ClientError):
                read_game_states("2026-05-09")


class TestWriteGameStates:
    def test_write_game_states_puts_to_correct_s3_path(self):
        mock_s3 = MagicMock()
        with patch("handler.boto3.client", return_value=mock_s3):
            from handler import write_game_states
            write_game_states("2026-05-09", FAKE_STATES)
        kw = mock_s3.put_object.call_args.kwargs
        assert kw["Bucket"] == "mlbdk"
        assert kw["Key"] == "raw_data/games/lineups/states/2026/2026-05-09.json"
        assert kw["ContentType"] == "application/json"
        body = json.loads(kw["Body"])
        assert body["date"] == "2026-05-09"


class TestWriteLineup:
    FAKE_PAYLOAD = {
        "game_pk": 745453,
        "date": "2026-05-09",
        "confirmed_at": "2026-05-09T17:00:00Z",
        "home_team_id": 111,
        "away_team_id": 147,
        "home_lineup": [{"batting_order": 1, "player_id": 660271}],
        "away_lineup": [{"batting_order": 1, "player_id": 545361}],
        "home_pitcher_id": 453286,
        "away_pitcher_id": 592789,
    }

    def test_write_lineup_puts_to_correct_s3_path(self):
        mock_s3 = MagicMock()
        with patch("handler.boto3.client", return_value=mock_s3):
            from handler import write_lineup
            write_lineup("2026-05-09", "745453", self.FAKE_PAYLOAD)
        kw = mock_s3.put_object.call_args.kwargs
        assert kw["Bucket"] == "mlbdk"
        assert kw["Key"] == "raw_data/games/lineups/2026/2026-05-09/745453.json"
        assert kw["ContentType"] == "application/json"

    def test_write_lineup_payload_schema(self):
        mock_s3 = MagicMock()
        with patch("handler.boto3.client", return_value=mock_s3):
            from handler import write_lineup
            write_lineup("2026-05-09", "745453", self.FAKE_PAYLOAD)
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"])
        for key in [
            "game_pk", "date", "confirmed_at", "home_team_id", "away_team_id",
            "home_lineup", "away_lineup", "home_pitcher_id", "away_pitcher_id",
        ]:
            assert key in body

    def test_write_lineup_home_lineup_is_list_of_batting_order_dicts(self):
        mock_s3 = MagicMock()
        with patch("handler.boto3.client", return_value=mock_s3):
            from handler import write_lineup
            write_lineup("2026-05-09", "745453", self.FAKE_PAYLOAD)
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"])
        assert isinstance(body["home_lineup"], list)
        assert body["home_lineup"][0]["batting_order"] == 1
        assert body["home_lineup"][0]["player_id"] == 660271


# =============================================================================
# EPIC 2: MLB API Calls
# =============================================================================

class TestFetchBoxscore:
    def test_fetch_boxscore_returns_teams_dict(self):
        with patch("handler.statsapi.get", return_value=FAKE_BOXSCORE_CONFIRMED) as mock_get:
            from handler import fetch_boxscore
            result = fetch_boxscore(745453)
        mock_get.assert_called_once_with("game_boxscore", {"gamePk": 745453})
        assert result == FAKE_BOXSCORE_CONFIRMED


class TestIsLineupConfirmed:
    def test_confirmed_when_both_batting_orders_nonempty(self):
        from handler import is_lineup_confirmed
        assert is_lineup_confirmed(FAKE_BOXSCORE_CONFIRMED) is True

    def test_not_confirmed_when_home_batting_order_empty(self):
        from handler import is_lineup_confirmed
        boxscore = {"teams": {"home": {"battingOrder": []}, "away": {"battingOrder": [545361]}}}
        assert is_lineup_confirmed(boxscore) is False

    def test_not_confirmed_when_away_batting_order_empty(self):
        from handler import is_lineup_confirmed
        boxscore = {"teams": {"home": {"battingOrder": [660271]}, "away": {"battingOrder": []}}}
        assert is_lineup_confirmed(boxscore) is False

    def test_not_confirmed_when_both_empty(self):
        from handler import is_lineup_confirmed
        assert is_lineup_confirmed(FAKE_BOXSCORE_EMPTY) is False


class TestFetchProbablePitchers:
    def _game_response(self, home_id=453286, away_id=592789, missing=None):
        pitchers = {}
        if missing != "home":
            pitchers["home"] = {"id": home_id}
        if missing != "away":
            pitchers["away"] = {"id": away_id}
        return {"gameData": {"probablePitchers": pitchers}}

    def test_returns_both_pitcher_ids_when_present(self):
        with patch("handler.statsapi.get", return_value=self._game_response()):
            from handler import fetch_probable_pitchers
            result = fetch_probable_pitchers(745453)
        assert result == {"home": 453286, "away": 592789}

    def test_returns_none_for_home_when_home_missing(self):
        with patch("handler.statsapi.get", return_value=self._game_response(missing="home")):
            from handler import fetch_probable_pitchers
            result = fetch_probable_pitchers(745453)
        assert result["home"] is None
        assert result["away"] == 592789

    def test_returns_none_for_away_when_away_missing(self):
        with patch("handler.statsapi.get", return_value=self._game_response(missing="away")):
            from handler import fetch_probable_pitchers
            result = fetch_probable_pitchers(745453)
        assert result["home"] == 453286
        assert result["away"] is None

    def test_returns_none_both_when_probable_pitchers_key_missing(self):
        with patch("handler.statsapi.get", return_value={"gameData": {}}):
            from handler import fetch_probable_pitchers
            result = fetch_probable_pitchers(745453)
        assert result == {"home": None, "away": None}


class TestBuildLineupPayload:
    SCHEDULE_GAME = {"home_team_id": 111, "away_team_id": 147}

    def test_build_lineup_payload_returns_correct_schema(self):
        from handler import build_lineup_payload
        result = build_lineup_payload(
            745453, "2026-05-09", self.SCHEDULE_GAME, FAKE_BOXSCORE_CONFIRMED, FAKE_PITCHERS
        )
        for key in [
            "game_pk", "date", "confirmed_at", "home_team_id", "away_team_id",
            "home_lineup", "away_lineup", "home_pitcher_id", "away_pitcher_id",
        ]:
            assert key in result

    def test_build_lineup_payload_batting_order_is_one_based(self):
        from handler import build_lineup_payload
        result = build_lineup_payload(
            745453, "2026-05-09", self.SCHEDULE_GAME, FAKE_BOXSCORE_CONFIRMED, FAKE_PITCHERS
        )
        assert result["home_lineup"][0]["batting_order"] == 1
        assert result["away_lineup"][0]["batting_order"] == 1

    def test_build_lineup_payload_uses_player_ids_from_batting_order(self):
        from handler import build_lineup_payload
        result = build_lineup_payload(
            745453, "2026-05-09", self.SCHEDULE_GAME, FAKE_BOXSCORE_CONFIRMED, FAKE_PITCHERS
        )
        assert result["home_lineup"][0]["player_id"] == 660271
        assert result["away_lineup"][0]["player_id"] == 545361

    def test_build_lineup_payload_pitcher_ids_come_from_pitchers_arg(self):
        from handler import build_lineup_payload
        result = build_lineup_payload(
            745453, "2026-05-09", self.SCHEDULE_GAME, FAKE_BOXSCORE_CONFIRMED, FAKE_PITCHERS
        )
        assert result["home_pitcher_id"] == 453286
        assert result["away_pitcher_id"] == 592789

    def test_build_lineup_payload_confirmed_at_ends_with_Z(self):
        from handler import build_lineup_payload
        result = build_lineup_payload(
            745453, "2026-05-09", self.SCHEDULE_GAME, FAKE_BOXSCORE_CONFIRMED, FAKE_PITCHERS
        )
        assert result["confirmed_at"].endswith("Z")


# =============================================================================
# EPIC 3: finish()
# =============================================================================

class TestFinish:
    def _states(self, confirmed=3, skipped=2):
        games = {}
        for i in range(confirmed):
            games[str(700000 + i)] = "CONFIRMED"
        for i in range(skipped):
            games[str(800000 + i)] = "SKIPPED"
        return {"date": "2026-05-09", "last_checked": None, "games": games}

    def test_finish_disables_eventbridge_rule_by_name(self):
        mock_events = MagicMock()
        with patch("handler.boto3.client", return_value=mock_events), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import finish
            finish("2026-05-09", self._states())
        mock_events.disable_rule.assert_called_once_with(Name="mlbdk-daily-lineup-fetch")

    def test_finish_calls_write_status_with_correct_counts(self):
        with patch("handler.boto3.client"), \
             patch("handler.write_status") as mock_ws, \
             patch.dict("os.environ", ENV):
            from handler import finish
            finish("2026-05-09", self._states(confirmed=3, skipped=2))
        kw = mock_ws.call_args.kwargs
        assert kw["games_processed"] == {"scheduled": 5, "confirmed": 3, "skipped": 2}

    def test_finish_still_writes_status_if_disable_rule_raises(self):
        mock_events = MagicMock()
        mock_events.disable_rule.side_effect = Exception("permission denied")
        with patch("handler.boto3.client", return_value=mock_events), \
             patch("handler.write_status") as mock_ws, \
             patch.dict("os.environ", ENV):
            from handler import finish
            finish("2026-05-09", self._states())  # must not raise
        mock_ws.assert_called_once()

    def test_finish_write_status_function_name_is_correct(self):
        with patch("handler.boto3.client"), \
             patch("handler.write_status") as mock_ws, \
             patch.dict("os.environ", ENV):
            from handler import finish
            finish("2026-05-09", self._states())
        assert mock_ws.call_args.kwargs["function_name"] == "daily_lineup_fetch"


# =============================================================================
# EPIC 4: Main Handler Logic
# =============================================================================

class TestHandlerEventParsing:
    def test_handler_uses_date_from_event(self):
        mock_rgs = MagicMock(return_value=dict(FAKE_STATES))
        with patch("handler.read_game_states", mock_rgs), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_rgs.assert_called_once_with("2026-05-09")

    def test_handler_defaults_to_today_when_no_date(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        mock_rgs = MagicMock(return_value=dict(FAKE_STATES))
        with patch("handler.read_game_states", mock_rgs), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({}, {})
        mock_rgs.assert_called_once_with(today)

    def test_handler_returns_200_on_normal_invocation(self):
        with patch("handler.read_game_states", return_value=dict(FAKE_STATES)), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            result = handler({"date": "2026-05-09"}, {})
        assert result["statusCode"] == 200


class TestHandlerMissingStates:
    def test_handler_returns_500_when_game_states_missing(self):
        with patch("handler.read_game_states", side_effect=_client_error()), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            result = handler({"date": "2026-05-09"}, {})
        assert result["statusCode"] == 500

    def test_handler_writes_failure_status_when_game_states_missing(self):
        with patch("handler.read_game_states", side_effect=_client_error()), \
             patch("handler.write_status") as mock_ws, \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_ws.assert_called_once()
        assert mock_ws.call_args.kwargs["status"] == "failed"

    def test_handler_does_not_call_fetch_boxscore_when_states_missing(self):
        mock_fb = MagicMock()
        with patch("handler.read_game_states", side_effect=_client_error()), \
             patch("handler.write_status"), \
             patch("handler.fetch_boxscore", mock_fb), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_fb.assert_not_called()


class TestHandlerHardCutoff:
    def _pending_states(self, pks=("745453", "745454")):
        return {"date": "2026-05-09", "last_checked": None, "games": {pk: "PENDING" for pk in pks}}

    def test_handler_marks_pending_as_skipped_when_cutoff_reached(self):
        mock_wgs = MagicMock()
        with patch("handler.read_game_states", return_value=self._pending_states()), \
             patch("handler.read_schedule", return_value=PAST_SCHEDULE), \
             patch("handler.write_game_states", mock_wgs), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        written = mock_wgs.call_args.args[1]
        assert written["games"]["745453"] == "SKIPPED"
        assert written["games"]["745454"] == "SKIPPED"

    def test_handler_calls_finish_when_cutoff_reached(self):
        mock_finish = MagicMock()
        with patch("handler.read_game_states", return_value=self._pending_states(("745453",))), \
             patch("handler.read_schedule", return_value=PAST_SCHEDULE), \
             patch("handler.write_game_states"), \
             patch("handler.finish", mock_finish), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_finish.assert_called_once()

    def test_handler_does_not_poll_boxscores_when_cutoff_reached(self):
        mock_fb = MagicMock()
        with patch("handler.read_game_states", return_value=self._pending_states(("745453",))), \
             patch("handler.read_schedule", return_value=PAST_SCHEDULE), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch("handler.fetch_boxscore", mock_fb), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_fb.assert_not_called()


class TestHandlerPollingLoop:
    def test_handler_skips_already_confirmed_games(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "CONFIRMED"}}
        mock_fb = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", mock_fb), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_fb.assert_not_called()

    def test_handler_skips_already_skipped_games(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "SKIPPED"}}
        mock_fb = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", mock_fb), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_fb.assert_not_called()

    def test_handler_confirms_game_when_both_batting_orders_nonempty(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        mock_wgs = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_CONFIRMED), \
             patch("handler.fetch_probable_pitchers", return_value=FAKE_PITCHERS), \
             patch("handler.write_lineup"), \
             patch("handler.write_game_states", mock_wgs), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        written = mock_wgs.call_args.args[1]
        assert written["games"]["745453"] == "CONFIRMED"

    def test_handler_fetches_pitchers_only_after_batting_order_confirms(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        mock_fp = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.fetch_probable_pitchers", mock_fp), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_fp.assert_not_called()

    def test_handler_writes_lineup_on_confirmation(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        mock_wl = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_CONFIRMED), \
             patch("handler.fetch_probable_pitchers", return_value=FAKE_PITCHERS), \
             patch("handler.write_lineup", mock_wl), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_wl.assert_called_once()
        assert mock_wl.call_args.args[1] == "745453"

    def test_handler_leaves_game_pending_when_boxscore_raises(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        mock_wgs = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", side_effect=Exception("API timeout")), \
             patch("handler.write_game_states", mock_wgs), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            result = handler({"date": "2026-05-09"}, {})
        assert result["statusCode"] == 200
        written = mock_wgs.call_args.args[1]
        assert written["games"]["745453"] == "PENDING"

    def test_handler_leaves_game_pending_when_batting_orders_empty(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        mock_wgs = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.write_game_states", mock_wgs), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        written = mock_wgs.call_args.args[1]
        assert written["games"]["745453"] == "PENDING"


class TestHandlerStatePersistence:
    def test_handler_updates_last_checked_timestamp(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        mock_wgs = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.write_game_states", mock_wgs), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        written = mock_wgs.call_args.args[1]
        assert written["last_checked"] is not None

    def test_handler_calls_finish_when_all_games_confirmed(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        mock_finish = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_CONFIRMED), \
             patch("handler.fetch_probable_pitchers", return_value=FAKE_PITCHERS), \
             patch("handler.write_lineup"), \
             patch("handler.write_game_states"), \
             patch("handler.finish", mock_finish), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_finish.assert_called_once()

    def test_handler_calls_finish_when_all_games_skipped_or_confirmed(self):
        states = {
            "date": "2026-05-09",
            "last_checked": None,
            "games": {"745453": "CONFIRMED", "745454": "SKIPPED"},
        }
        mock_finish = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.write_game_states"), \
             patch("handler.finish", mock_finish), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_finish.assert_called_once()

    def test_handler_does_not_call_finish_when_games_still_pending(self):
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        mock_finish = MagicMock()
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.write_game_states"), \
             patch("handler.finish", mock_finish), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_finish.assert_not_called()


class TestHandlerWriteStatus:
    def test_handler_returns_200_on_success(self):
        with patch("handler.read_game_states", return_value=dict(FAKE_STATES)), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_EMPTY), \
             patch("handler.write_game_states"), \
             patch("handler.finish"), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            result = handler({"date": "2026-05-09"}, {})
        assert result["statusCode"] == 200

    def test_handler_returns_500_on_unrecoverable_exception(self):
        with patch("handler.read_game_states", return_value=dict(FAKE_STATES)), \
             patch("handler.read_schedule", side_effect=Exception("S3 down")), \
             patch("handler.write_status"), \
             patch.dict("os.environ", ENV):
            from handler import handler
            result = handler({"date": "2026-05-09"}, {})
        assert result["statusCode"] == 500

    def test_handler_writes_failure_status_on_unrecoverable_exception(self):
        with patch("handler.read_game_states", return_value=dict(FAKE_STATES)), \
             patch("handler.read_schedule", side_effect=Exception("S3 down")), \
             patch("handler.write_status") as mock_ws, \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_ws.assert_called_once()
        assert mock_ws.call_args.kwargs["status"] == "failed"
        assert "S3 down" in mock_ws.call_args.kwargs["error"]

    def test_handler_write_status_games_processed_shape_on_finish(self):
        # finish() is not patched — it runs for real so write_status is called via finish()
        states = {"date": "2026-05-09", "last_checked": None, "games": {"745453": "PENDING"}}
        with patch("handler.read_game_states", return_value=states), \
             patch("handler.read_schedule", return_value=FUTURE_SCHEDULE), \
             patch("handler.fetch_boxscore", return_value=FAKE_BOXSCORE_CONFIRMED), \
             patch("handler.fetch_probable_pitchers", return_value=FAKE_PITCHERS), \
             patch("handler.write_lineup"), \
             patch("handler.write_game_states"), \
             patch("handler.boto3.client"), \
             patch("handler.write_status") as mock_ws, \
             patch.dict("os.environ", ENV):
            from handler import handler
            handler({"date": "2026-05-09"}, {})
        mock_ws.assert_called_once()
        gp = mock_ws.call_args.kwargs["games_processed"]
        assert "scheduled" in gp
        assert "confirmed" in gp
        assert "skipped" in gp
