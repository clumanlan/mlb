import json
import datetime as real_datetime
from datetime import timedelta
from unittest.mock import patch, MagicMock

from feast import Entity, Field, FeatureView, FileSource
from feast.data_format import ParquetFormat
from feast.types import Int64, Float32
from feast.value_type import ValueType
import xgboost as xgb


def _train_tiny_xgb_model():
    X = [[0, 1], [1, 0], [1, 1], [0, 0]] * 3
    y = [0, 1, 1, 0] * 3
    model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
    model.fit(X, y)
    return model


FIXED_NOW = real_datetime.datetime(2026, 9, 20, 14, 30, 0, tzinfo=real_datetime.timezone.utc)


class _FixedDatetime(real_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW


def _make_feature_view(name, fields):
    entity = Entity(name=f"{name}_entity", join_keys=[f"{name}_id"], value_type=ValueType.STRING)
    source = FileSource(
        name=f"{name}_source",
        path=f"s3://mlbdk/fake/{name}/",
        file_format=ParquetFormat(),
        timestamp_field="event_timestamp",
    )
    return FeatureView(
        name=name,
        entities=[entity],
        ttl=timedelta(days=7),
        schema=fields,
        source=source,
        online=True,
    )


class TestSnapshotFeatureSchema:
    def test_extracts_name_and_dtype_per_feature_view(self):
        fv_one = _make_feature_view("fv_one", [Field(name="a", dtype=Int64)])
        fv_two = _make_feature_view(
            "fv_two",
            [Field(name="b", dtype=Float32), Field(name="c", dtype=Int64)],
        )

        from model_registry import snapshot_feature_schema
        result = snapshot_feature_schema([fv_one, fv_two])

        # Feast doesn't guarantee field order within a schema, so compare
        # each feature view's fields as a set, not a fixed-order list.
        assert result.keys() == {"fv_one", "fv_two"}
        assert {tuple(sorted(f.items())) for f in result["fv_one"]} == {
            (("dtype", "Int64"), ("name", "a")),
        }
        assert {tuple(sorted(f.items())) for f in result["fv_two"]} == {
            (("dtype", "Float32"), ("name", "b")),
            (("dtype", "Int64"), ("name", "c")),
        }

    def test_empty_feature_view_list_returns_empty_dict(self):
        from model_registry import snapshot_feature_schema
        assert snapshot_feature_schema([]) == {}


class TestRegisterModel:
    def _call(self, mock_s3, **overrides):
        kwargs = dict(
            model_name="n_pa_predictor_low_pa",
            model=_train_tiny_xgb_model(),
            feature_schema={"fv_one": [{"name": "a", "dtype": "Int64"}]},
            operating_threshold=0.85,
            metrics={"precision": 0.647, "n": 116},
        )
        kwargs.update(overrides)
        with patch("model_registry.boto3.client", return_value=mock_s3), \
             patch("model_registry.datetime", _FixedDatetime):
            from model_registry import register_model
            return register_model(**kwargs)

    def test_writes_versioned_key_under_model_name(self):
        mock_s3 = MagicMock()
        self._call(mock_s3)
        keys = [c.kwargs["Key"] for c in mock_s3.put_object.call_args_list]
        assert "models/n_pa_predictor_low_pa/2026-09-20T14-30-00.json" in keys

    def test_also_writes_latest_alias_with_same_body(self):
        mock_s3 = MagicMock()
        self._call(mock_s3)
        calls = {c.kwargs["Key"]: c.kwargs["Body"] for c in mock_s3.put_object.call_args_list}
        assert "models/n_pa_predictor_low_pa/latest.json" in calls
        versioned_body = calls["models/n_pa_predictor_low_pa/2026-09-20T14-30-00.json"]
        assert calls["models/n_pa_predictor_low_pa/latest.json"] == versioned_body

    def test_payload_contains_expected_metadata_fields(self):
        mock_s3 = MagicMock()
        self._call(mock_s3)
        body = json.loads(mock_s3.put_object.call_args_list[0].kwargs["Body"])
        assert body["model_name"] == "n_pa_predictor_low_pa"
        assert body["version"] == "2026-09-20T14-30-00"
        assert body["feature_schema"] == {"fv_one": [{"name": "a", "dtype": "Int64"}]}
        assert body["operating_threshold"] == 0.85
        assert body["metrics"] == {"precision": 0.647, "n": 116}

    def test_embeds_real_xgboost_model_json_not_a_string(self):
        mock_s3 = MagicMock()
        self._call(mock_s3)
        body = json.loads(mock_s3.put_object.call_args_list[0].kwargs["Body"])
        assert isinstance(body["model"], dict)
        assert "learner" in body["model"]

    def test_returns_the_version_string(self):
        mock_s3 = MagicMock()
        version = self._call(mock_s3)
        assert version == "2026-09-20T14-30-00"


class TestLoadModel:
    def _registered_body(self, **overrides):
        """Real bytes register_model would have written to latest.json —
        built by actually calling register_model against a mocked S3 client,
        not hand-constructed, so load_model is tested against the real
        payload shape, not a guess at it."""
        mock_s3 = MagicMock()
        kwargs = dict(
            model_name="n_pa_predictor_low_pa",
            model=_train_tiny_xgb_model(),
            feature_schema={"fv_one": [{"name": "a", "dtype": "Int64"}]},
            operating_threshold=0.85,
            metrics={"precision": 0.647, "n": 116},
        )
        kwargs.update(overrides)
        with patch("model_registry.boto3.client", return_value=mock_s3), \
             patch("model_registry.datetime", _FixedDatetime):
            from model_registry import register_model
            register_model(**kwargs)
        return mock_s3.put_object.call_args_list[0].kwargs["Body"]

    def _call(self, mock_s3, **overrides):
        with patch("model_registry.boto3.client", return_value=mock_s3):
            from model_registry import load_model
            return load_model(model_name="n_pa_predictor_low_pa", **overrides)

    def test_reads_latest_json_for_the_given_model_name(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: self._registered_body())}
        self._call(mock_s3)
        key = mock_s3.get_object.call_args.kwargs["Key"]
        assert key == "models/n_pa_predictor_low_pa/latest.json"

    def test_returns_metadata_matching_what_was_registered(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: self._registered_body())}
        result = self._call(mock_s3)
        assert result["version"] == "2026-09-20T14-30-00"
        assert result["operating_threshold"] == 0.85
        assert result["feature_schema"] == {"fv_one": [{"name": "a", "dtype": "Int64"}]}
        assert result["metrics"] == {"precision": 0.647, "n": 116}

    def test_returns_a_model_that_predicts_identically_to_the_original(self):
        X = [[0, 1], [1, 0], [1, 1], [0, 0]] * 3
        original = _train_tiny_xgb_model()
        mock_s3 = MagicMock()
        body = self._registered_body(model=original)
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}

        result = self._call(mock_s3)

        import numpy as np
        assert np.array_equal(result["model"].predict_proba(X), original.predict_proba(X))
