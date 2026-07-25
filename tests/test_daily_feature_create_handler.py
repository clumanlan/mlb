import json
import sys
import os
import re
import importlib.util
from unittest.mock import MagicMock, patch

_handler_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/lambdas/daily_feature_create/handler.py")
)

_heavy_mocks = {
    "boto3": MagicMock(),
    "feast": MagicMock(),
    "team_batter_base": MagicMock(),
    "player_batter_base": MagicMock(),
    "sp_features": MagicMock(),
    "bullpen_features": MagicMock(),
    "pandas": MagicMock(),
}

with patch.dict(sys.modules, _heavy_mocks):
    _spec = importlib.util.spec_from_file_location("feature_create_handler", _handler_path)
    feature_create_handler = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(feature_create_handler)


def test_yesterday_returns_date_string():
    result = feature_create_handler._yesterday()
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", result), f"Expected YYYY-MM-DD, got: {result!r}"


def test_team_batter_base_key_is_interpolated():
    with patch.object(feature_create_handler, '_get_snapshot') as mock_snap, \
         patch.object(feature_create_handler, '_write_snapshot') as mock_write:
        mock_snap.return_value = MagicMock()
        feature_create_handler._run_team_batter_base("2025", "2025-04-15")

    key = mock_write.call_args[1]['key']
    assert "{year}" not in key, "f-string prefix missing — key contains literal {year}"
    assert "{date}" not in key, "f-string prefix missing — key contains literal {date}"
    assert "2025" in key
    assert "2025-04-15" in key


def test_materialization_error_stored_under_correct_key():
    with patch.object(feature_create_handler, '_run_team_batter_base', return_value='ok'), \
         patch.object(feature_create_handler, '_run_player_batter_base', return_value='ok'), \
         patch.object(feature_create_handler, '_run_sp_base', return_value='ok'), \
         patch.object(feature_create_handler, '_run_bullpen_base', return_value='ok'):
        mock_store = MagicMock()
        mock_store.materialize_incremental.side_effect = RuntimeError("feast down")
        with patch.object(feature_create_handler, 'FeatureStore', return_value=mock_store):
            response = feature_create_handler.lambda_handler({"date": "2025-04-15"}, {})

    body = json.loads(response['body'])
    assert 'materialzie' not in body['results'], "typo key must not appear"
    assert 'materialize' in body['results']
    assert body['results']['materialize'].startswith('error:')
