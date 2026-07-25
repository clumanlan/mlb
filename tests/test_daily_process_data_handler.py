import sys
import os
import pytest
from unittest.mock import patch, MagicMock

HANDLER_DIR = os.path.join(os.path.dirname(__file__), "../src/lambdas/daily_process_data")


@pytest.fixture(autouse=True)
def handler_path():
    sys.path.insert(0, HANDLER_DIR)
    sys.modules.pop("handler", None)
    yield
    sys.modules.pop("handler", None)
    if HANDLER_DIR in sys.path:
        sys.path.remove(HANDLER_DIR)


def _make_df(gamepks):
    import pandas as pd
    return pd.DataFrame({"gamepk": gamepks})


class TestProcessDataHandlerWriteStatus:
    def test_handler_calls_write_status_on_success(self):
        with patch("handler._read_s3", return_value=MagicMock()), \
             patch("handler.process_schedule", return_value=_make_df([1, 2])), \
             patch("handler.process_game_info", return_value=MagicMock()), \
             patch("handler.process_batter_boxscore", return_value=MagicMock()), \
             patch("handler.process_pitcher_boxscore", return_value=MagicMock()), \
             patch("handler.process_player_info", return_value=MagicMock()), \
             patch("handler.prepare_batter_boxscore", return_value=_make_df([1, 2])), \
             patch("handler.prepare_pitcher_boxscore", return_value=_make_df([1, 2])), \
             patch("handler.prepare_playbyplay", return_value=_make_df([1, 2])), \
             patch("handler._write_s3"), \
             patch("handler.write_status") as mock_ws:
            from handler import lambda_handler
            lambda_handler({"date": "2026-05-03"}, {})
        mock_ws.assert_called_once()
        call_kwargs = mock_ws.call_args.kwargs
        assert call_kwargs["status"] == "success"
        assert call_kwargs["error"] is None
        assert call_kwargs["function_name"] == "daily_process_data"
        assert call_kwargs["run_date"] == "2026-05-03"

    def test_handler_calls_write_status_on_failure(self):
        with patch("handler._read_s3", side_effect=RuntimeError("S3 read failed")), \
             patch("handler.write_status") as mock_ws:
            from handler import lambda_handler
            with pytest.raises(RuntimeError):
                lambda_handler({"date": "2026-05-03"}, {})
        mock_ws.assert_called_once()
        call_kwargs = mock_ws.call_args.kwargs
        assert call_kwargs["status"] == "failed"
        assert "S3 read failed" in call_kwargs["error"]

    def test_handler_write_status_games_processed_shape(self):
        with patch("handler._read_s3", return_value=MagicMock()), \
             patch("handler.process_schedule", return_value=_make_df([1, 2, 3])), \
             patch("handler.process_game_info", return_value=MagicMock()), \
             patch("handler.process_batter_boxscore", return_value=MagicMock()), \
             patch("handler.process_pitcher_boxscore", return_value=MagicMock()), \
             patch("handler.process_player_info", return_value=MagicMock()), \
             patch("handler.prepare_batter_boxscore", return_value=_make_df([1, 2, 3])), \
             patch("handler.prepare_pitcher_boxscore", return_value=_make_df([1, 2])), \
             patch("handler.prepare_playbyplay", return_value=_make_df([1, 2, 3])), \
             patch("handler._write_s3"), \
             patch("handler.write_status") as mock_ws:
            from handler import lambda_handler
            lambda_handler({"date": "2026-05-03"}, {})
        gp = mock_ws.call_args.kwargs["games_processed"]
        assert "schedule" in gp
        assert "batter_prepared" in gp
        assert "pitcher_prepared" in gp
        assert "playbyplay_prepared" in gp
        assert gp["schedule"] == 3
        assert gp["batter_prepared"] == 3
        assert gp["pitcher_prepared"] == 2
        assert gp["playbyplay_prepared"] == 3
