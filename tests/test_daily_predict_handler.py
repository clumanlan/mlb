import json
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch

import pandas as pd

_handler_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/lambdas/daily_predict/handler.py")
)

_heavy_mocks = {
    "boto3": MagicMock(),
    "feast": MagicMock(),
    "model_registry": MagicMock(),
    "status_writer": MagicMock(),
    "predict": MagicMock(),
}

with patch.dict(sys.modules, _heavy_mocks):
    _spec = importlib.util.spec_from_file_location("daily_predict_handler", _handler_path)
    daily_predict_handler = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(daily_predict_handler)


def _fake_result_df(qualifies=(True, False, True)):
    return pd.DataFrame({
        "personId": [str(i) for i in range(len(qualifies))],
        "gamepk": ["g1"] * len(qualifies),
        "game_date": ["2025-04-15"] * len(qualifies),
        "batting_order": list(range(1, len(qualifies) + 1)),
        "batter_n_pa_roll_season_avg_n_pa_per_game": [4.0] * len(qualifies),
        "predicted_probability": [0.9, 0.1, 0.87],
        "qualifies": list(qualifies),
        "model_version": ["2026-09-22T00-00-00"] * len(qualifies),
    })


def test_today_returns_date_string():
    import re
    result = daily_predict_handler._today()
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", result)


def test_handler_defaults_to_today_when_no_date_given():
    with patch.object(daily_predict_handler, 'predict_for_date', return_value=_fake_result_df()) as mock_predict, \
         patch.object(daily_predict_handler, '_today', return_value='2026-09-22'), \
         patch.object(daily_predict_handler, '_write_predictions', return_value='predictions/n_pa_predictor_low_pa/2026/2026-09-22.parquet'):
        daily_predict_handler.lambda_handler({}, {})

    assert mock_predict.call_args.args[0] == '2026-09-22'


def test_handler_uses_explicit_date_when_passed():
    with patch.object(daily_predict_handler, 'predict_for_date', return_value=_fake_result_df()) as mock_predict, \
         patch.object(daily_predict_handler, '_write_predictions', return_value='predictions/n_pa_predictor_low_pa/2025/2025-04-15.parquet'):
        daily_predict_handler.lambda_handler({"date": "2025-04-15"}, {})

    assert mock_predict.call_args.args[0] == '2025-04-15'


def test_write_predictions_uses_expected_s3_key():
    mock_s3 = MagicMock()
    with patch.object(daily_predict_handler, 's3', mock_s3):
        daily_predict_handler._write_predictions(_fake_result_df(), "2025-04-15")

    key = mock_s3.put_object.call_args.kwargs["Key"]
    assert key == "predictions/n_pa_predictor_low_pa/2025/2025-04-15.parquet"


def test_handler_calls_write_status_on_success_with_expected_games_processed():
    with patch.object(daily_predict_handler, 'predict_for_date', return_value=_fake_result_df()), \
         patch.object(daily_predict_handler, '_write_predictions', return_value='some/key.parquet'), \
         patch.object(daily_predict_handler, 'write_status') as mock_write_status:
        daily_predict_handler.lambda_handler({"date": "2025-04-15"}, {})

    kwargs = mock_write_status.call_args.kwargs
    assert kwargs["status"] == "success"
    assert kwargs["games_processed"] == {"batters_predicted": 3, "qualifying_predictions": 2}
    assert kwargs["error"] is None


def test_handler_calls_write_status_on_failure_and_reraises():
    with patch.object(daily_predict_handler, 'predict_for_date', side_effect=RuntimeError("boom")), \
         patch.object(daily_predict_handler, 'write_status') as mock_write_status:
        try:
            daily_predict_handler.lambda_handler({"date": "2025-04-15"}, {})
            assert False, "expected RuntimeError to propagate"
        except RuntimeError:
            pass

    kwargs = mock_write_status.call_args.kwargs
    assert kwargs["status"] == "failed"
    assert "boom" in kwargs["error"]


def test_handler_returns_200_with_row_count_on_success():
    with patch.object(daily_predict_handler, 'predict_for_date', return_value=_fake_result_df()), \
         patch.object(daily_predict_handler, '_write_predictions', return_value='some/key.parquet'), \
         patch.object(daily_predict_handler, 'write_status'):
        response = daily_predict_handler.lambda_handler({"date": "2025-04-15"}, {})

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["rows"] == 3
