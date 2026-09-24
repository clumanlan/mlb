import json
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch

_handler_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/lambdas/daily_feature_materialize/handler.py")
)

_heavy_mocks = {
    "feast": MagicMock(),
    "boto3": MagicMock(),
}

with patch.dict(sys.modules, _heavy_mocks):
    _spec = importlib.util.spec_from_file_location("feature_materialize_handler", _handler_path)
    feature_materialize_handler = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(feature_materialize_handler)


def test_feast_path_resolves_relative_to_handler_file():
    expected = os.path.dirname(os.path.abspath(feature_materialize_handler.__file__))
    assert feature_materialize_handler.FEAST_PATH == expected


def test_handler_calls_materialize_incremental():
    mock_store = MagicMock()
    with patch.object(feature_materialize_handler, 'FeatureStore', return_value=mock_store) as mock_fs:
        response = feature_materialize_handler.lambda_handler({}, {})

    mock_fs.assert_called_once_with(repo_path=feature_materialize_handler.FEAST_PATH)
    mock_store.materialize_incremental.assert_called_once()
    assert response['statusCode'] == 200
    body = json.loads(response['body'])
    assert body['status'] == 'ok'


def test_handler_returns_500_on_materialize_error():
    mock_store = MagicMock()
    mock_store.materialize_incremental.side_effect = RuntimeError("feast down")
    with patch.object(feature_materialize_handler, 'FeatureStore', return_value=mock_store):
        response = feature_materialize_handler.lambda_handler({}, {})

    assert response['statusCode'] == 500
    body = json.loads(response['body'])
    assert 'feast down' in body['error']


def test_handler_invokes_predict_lambda_on_success():
    mock_store = MagicMock()
    with patch.object(feature_materialize_handler, 'FeatureStore', return_value=mock_store), \
         patch.object(feature_materialize_handler, 'lambda_client') as mock_lambda_client:
        feature_materialize_handler.lambda_handler({}, {})

    mock_lambda_client.invoke.assert_called_once_with(
        FunctionName='daily_predict', InvocationType='Event',
    )


def test_handler_does_not_invoke_predict_when_materialize_fails():
    mock_store = MagicMock()
    mock_store.materialize_incremental.side_effect = RuntimeError("feast down")
    with patch.object(feature_materialize_handler, 'FeatureStore', return_value=mock_store), \
         patch.object(feature_materialize_handler, 'lambda_client') as mock_lambda_client:
        feature_materialize_handler.lambda_handler({}, {})

    mock_lambda_client.invoke.assert_not_called()
