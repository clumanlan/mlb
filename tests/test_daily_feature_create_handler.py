import json
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch

_handler_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/lambdas/daily_feature_create/handler.py")
)

_heavy_mocks = {
    "boto3": MagicMock(),
    "batter_lineup": MagicMock(),
    "batter_pa_volume": MagicMock(),
    "pandas": MagicMock(),
}

with patch.dict(sys.modules, _heavy_mocks):
    _spec = importlib.util.spec_from_file_location("feature_create_handler", _handler_path)
    feature_create_handler = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(feature_create_handler)


def test_handler_module_never_imports_feast():
    assert not hasattr(feature_create_handler, 'FeatureStore'), (
        "daily_feature_create must not import feast — materialization moved to "
        "daily_feature_materialize (a container-image Lambda) because feast's real "
        "dependency closure doesn't fit Lambda's zip+layers size cap."
    )


def test_today_returns_date_string():
    import re
    result = feature_create_handler._today()
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", result), f"Expected YYYY-MM-DD, got: {result!r}"


def test_run_batter_lineup_writes_expected_s3_key():
    with patch.object(feature_create_handler, 'compute_batter_lineup_features_for_date') as mock_compute, \
         patch.object(feature_create_handler, '_write_snapshot') as mock_write:
        mock_df = MagicMock()
        mock_df.empty = False
        mock_compute.return_value = mock_df

        result = feature_create_handler._run_batter_lineup("2026-09-16")

    assert result == "ok"
    key = mock_write.call_args[1]['key'] if 'key' in mock_write.call_args[1] else mock_write.call_args[0][1]
    assert key == "feast/features/batter_lineup/2026/2026-09-16.parquet"


def test_run_batter_lineup_returns_empty_when_no_rows():
    with patch.object(feature_create_handler, 'compute_batter_lineup_features_for_date') as mock_compute, \
         patch.object(feature_create_handler, '_write_snapshot') as mock_write:
        mock_df = MagicMock()
        mock_df.empty = True
        mock_compute.return_value = mock_df

        result = feature_create_handler._run_batter_lineup("2026-09-16")

    assert result == "empty"
    mock_write.assert_not_called()


def test_run_batter_pa_volume_writes_expected_s3_key():
    with patch.object(feature_create_handler, 'compute_batter_pa_volume_features_for_date') as mock_compute, \
         patch.object(feature_create_handler, '_write_snapshot') as mock_write:
        mock_df = MagicMock()
        mock_df.empty = False
        mock_compute.return_value = mock_df

        result = feature_create_handler._run_batter_pa_volume("2026-09-16")

    assert result == "ok"
    key = mock_write.call_args[1]['key'] if 'key' in mock_write.call_args[1] else mock_write.call_args[0][1]
    assert key == "feast/features/batter_pa_volume/2026/2026-09-16.parquet"


def test_run_batter_pa_volume_returns_empty_when_no_rows():
    with patch.object(feature_create_handler, 'compute_batter_pa_volume_features_for_date') as mock_compute, \
         patch.object(feature_create_handler, '_write_snapshot') as mock_write:
        mock_df = MagicMock()
        mock_df.empty = True
        mock_compute.return_value = mock_df

        result = feature_create_handler._run_batter_pa_volume("2026-09-16")

    assert result == "empty"
    mock_write.assert_not_called()


def test_handler_defaults_to_today_not_yesterday():
    with patch.object(feature_create_handler, '_run_batter_lineup', return_value='ok') as mock_lineup, \
         patch.object(feature_create_handler, '_run_batter_pa_volume', return_value='ok'), \
         patch.object(feature_create_handler, '_today', return_value='2026-09-16'):
        feature_create_handler.lambda_handler({}, {})

    mock_lineup.assert_called_once_with('2026-09-16')


def test_handler_uses_explicit_date_when_passed():
    with patch.object(feature_create_handler, '_run_batter_lineup', return_value='ok') as mock_lineup, \
         patch.object(feature_create_handler, '_run_batter_pa_volume', return_value='ok'):
        feature_create_handler.lambda_handler({"date": "2025-04-15"}, {})

    mock_lineup.assert_called_once_with('2025-04-15')


def test_handler_returns_207_when_a_transform_fails():
    with patch.object(feature_create_handler, '_run_batter_lineup', side_effect=RuntimeError("boom")), \
         patch.object(feature_create_handler, '_run_batter_pa_volume', return_value='ok'), \
         patch.object(feature_create_handler, 'lambda_client'):
        response = feature_create_handler.lambda_handler({"date": "2025-04-15"}, {})

    body = json.loads(response['body'])
    assert response['statusCode'] == 207
    assert body['failed'] == ['batter_lineup']
    assert body['succeeded'] == ['batter_pa_volume']
    assert 'materialize' not in body['results']


def _s3_event(key):
    return {"Records": [{"s3": {"object": {"key": key}}}]}


def test_handler_extracts_date_from_s3_event_key():
    event = _s3_event("raw_data/games/lineups/2026/2026-09-18/745123.json")
    with patch.object(feature_create_handler, '_run_batter_lineup', return_value='ok') as mock_lineup, \
         patch.object(feature_create_handler, '_run_batter_pa_volume', return_value='ok'), \
         patch.object(feature_create_handler, 'lambda_client'):
        feature_create_handler.lambda_handler(event, {})

    mock_lineup.assert_called_once_with('2026-09-18')


def test_handler_skips_states_file_events():
    """daily_lineup_fetch writes a states-tracking file to the same
    raw_data/games/lineups/ prefix on every 15-min poll, whether or not
    anything new confirmed that cycle — S3's prefix/suffix-only filters
    can't exclude it, so the handler itself must recognize and skip it
    rather than recomputing features for a non-event."""
    event = _s3_event("raw_data/games/lineups/states/2026/2026-09-18.json")
    with patch.object(feature_create_handler, '_run_batter_lineup') as mock_lineup, \
         patch.object(feature_create_handler, '_run_batter_pa_volume') as mock_pa_volume, \
         patch.object(feature_create_handler, 'lambda_client') as mock_lambda_client:
        response = feature_create_handler.lambda_handler(event, {})

    mock_lineup.assert_not_called()
    mock_pa_volume.assert_not_called()
    mock_lambda_client.invoke.assert_not_called()
    assert response['statusCode'] == 200


def test_handler_still_supports_direct_invoke_date():
    with patch.object(feature_create_handler, '_run_batter_lineup', return_value='ok') as mock_lineup, \
         patch.object(feature_create_handler, '_run_batter_pa_volume', return_value='ok'), \
         patch.object(feature_create_handler, 'lambda_client'):
        feature_create_handler.lambda_handler({"date": "2025-04-15"}, {})

    mock_lineup.assert_called_once_with('2025-04-15')


def test_handler_invokes_materialize_lambda_on_success():
    with patch.object(feature_create_handler, '_run_batter_lineup', return_value='ok'), \
         patch.object(feature_create_handler, '_run_batter_pa_volume', return_value='ok'), \
         patch.object(feature_create_handler, 'lambda_client') as mock_lambda_client:
        feature_create_handler.lambda_handler({"date": "2025-04-15"}, {})

    mock_lambda_client.invoke.assert_called_once_with(
        FunctionName='daily_feature_materialize', InvocationType='Event',
    )


def test_handler_does_not_invoke_materialize_when_all_failed():
    with patch.object(feature_create_handler, '_run_batter_lineup', side_effect=RuntimeError("boom")), \
         patch.object(feature_create_handler, '_run_batter_pa_volume', side_effect=RuntimeError("boom")), \
         patch.object(feature_create_handler, 'lambda_client') as mock_lambda_client:
        feature_create_handler.lambda_handler({"date": "2025-04-15"}, {})

    mock_lambda_client.invoke.assert_not_called()
