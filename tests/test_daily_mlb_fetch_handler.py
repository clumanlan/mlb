import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

HANDLER_DIR = os.path.join(os.path.dirname(__file__), "../src/lambdas/daily_mlb_fetch")


@pytest.fixture(autouse=True)
def handler_path():
    sys.path.insert(0, HANDLER_DIR)
    sys.modules.pop("handler", None)
    yield
    sys.modules.pop("handler", None)
    if HANDLER_DIR in sys.path:
        sys.path.remove(HANDLER_DIR)


def _game_df(gamepks, pk_col="gamePk"):
    return pd.DataFrame({pk_col: gamepks})


class TestMlbFetchHandlerWriteStatus:
    def test_handler_calls_write_status_on_success(self):
        schedule_df = pd.DataFrame({"game_id": [1, 2], "gamePk": [1, 2]})
        with patch("handler.fetch_schedule_df", return_value=(schedule_df, [])), \
             patch("handler.fetch_game_info", return_value=(_game_df([1, 2]), [])), \
             patch("handler.fetch_batter_pitcher_boxscore",
                   return_value=(_game_df([1, 2]), _game_df([1, 2]), [])), \
             patch("handler.fetch_playbyplay_data",
                   return_value=(_game_df([1, 2], pk_col="gamepk"), [])), \
             patch("handler.ReferenceFetcher") as mock_ref, \
             patch("handler.send_metrics_to_cloudwatch"), \
             patch("handler.write_status") as mock_ws:
            mock_ref.return_value.upsert_players.return_value = []
            from handler import lambda_handler
            lambda_handler({"date": "2026-05-03"}, {})
        mock_ws.assert_called_once()
        call_kwargs = mock_ws.call_args.kwargs
        assert call_kwargs["status"] == "success"
        assert call_kwargs["error"] is None
        assert call_kwargs["function_name"] == "daily_mlb_fetch"
        assert call_kwargs["run_date"] == "2026-05-03"

    def test_handler_calls_write_status_on_failure(self):
        with patch("handler.fetch_schedule_df", side_effect=RuntimeError("API down")), \
             patch("handler.write_status") as mock_ws:
            from handler import lambda_handler
            with pytest.raises(RuntimeError):
                lambda_handler({"date": "2026-05-03"}, {})
        mock_ws.assert_called_once()
        call_kwargs = mock_ws.call_args.kwargs
        assert call_kwargs["status"] == "failed"
        assert "API down" in call_kwargs["error"]

    def test_handler_write_status_games_processed_shape(self):
        schedule_df = pd.DataFrame({"game_id": [1, 2, 3], "gamePk": [1, 2, 3]})
        with patch("handler.fetch_schedule_df", return_value=(schedule_df, [])), \
             patch("handler.fetch_game_info", return_value=(_game_df([1, 2, 3]), [])), \
             patch("handler.fetch_batter_pitcher_boxscore",
                   return_value=(_game_df([1, 2]), _game_df([1, 2, 3]), [])), \
             patch("handler.fetch_playbyplay_data",
                   return_value=(_game_df([1, 2, 3], pk_col="gamepk"), [])), \
             patch("handler.ReferenceFetcher") as mock_ref, \
             patch("handler.send_metrics_to_cloudwatch"), \
             patch("handler.write_status") as mock_ws:
            mock_ref.return_value.upsert_players.return_value = []
            from handler import lambda_handler
            lambda_handler({"date": "2026-05-03"}, {})
        gp = mock_ws.call_args.kwargs["games_processed"]
        assert "schedule" in gp
        assert "game_info" in gp
        assert "batter_boxscore" in gp
        assert "pitcher_boxscore" in gp
        assert "playbyplay" in gp
        assert gp["schedule"] == 3
        assert gp["game_info"] == 3
        assert gp["batter_boxscore"] == 2
        assert gp["pitcher_boxscore"] == 3
        assert gp["playbyplay"] == 3
