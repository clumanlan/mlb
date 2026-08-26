import io
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest
from botocore.exceptions import ClientError

import s3_client


def _make_s3_object(key, last_modified, body_bytes):
    return {
        "Key": key,
        "LastModified": last_modified,
        "Body": io.BytesIO(body_bytes),
    }


# ---------------------------------------------------------------------------
# get_latest_s3_json
# ---------------------------------------------------------------------------

class TestGetLatestS3Json:
    def test_returns_parsed_json(self):
        payload = {"status": "success"}
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "prefix/2026-05-08.json", "LastModified": datetime(2026, 5, 8, tzinfo=timezone.utc)}]
        }
        mock_s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(payload).encode())}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            result = s3_client.get_latest_s3_json("mlbdk", "prefix/")

        assert result == payload

    def test_returns_most_recent_when_multiple_objects(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/2026-05-06.json", "LastModified": datetime(2026, 5, 6, tzinfo=timezone.utc)},
                {"Key": "prefix/2026-05-08.json", "LastModified": datetime(2026, 5, 8, tzinfo=timezone.utc)},
                {"Key": "prefix/2026-05-07.json", "LastModified": datetime(2026, 5, 7, tzinfo=timezone.utc)},
            ]
        }
        mock_s3.get_object.return_value = {"Body": io.BytesIO(b"{}")}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            s3_client.get_latest_s3_json("mlbdk", "prefix/")

        mock_s3.get_object.assert_called_once_with(Bucket="mlbdk", Key="prefix/2026-05-08.json")

    def test_raises_file_not_found_when_prefix_empty(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {"Contents": []}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            with pytest.raises(FileNotFoundError):
                s3_client.get_latest_s3_json("mlbdk", "prefix/")

    def test_raises_file_not_found_when_no_contents_key(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {}  # no 'Contents' key — empty prefix in S3

        with patch("s3_client.boto3.client", return_value=mock_s3):
            with pytest.raises(FileNotFoundError):
                s3_client.get_latest_s3_json("mlbdk", "prefix/")


# ---------------------------------------------------------------------------
# get_s3_json (exact key)
# ---------------------------------------------------------------------------

class TestGetS3Json:
    def test_returns_parsed_json_for_exact_key(self):
        payload = [{"game_pk": 1}]
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(payload).encode())}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            result = s3_client.get_s3_json("mlbdk", "raw_data/schedule/2026/2026-05-08.json")

        assert result == payload

    def test_raises_file_not_found_on_no_such_key(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
        )

        with patch("s3_client.boto3.client", return_value=mock_s3):
            with pytest.raises(FileNotFoundError):
                s3_client.get_s3_json("mlbdk", "missing.json")


# ---------------------------------------------------------------------------
# get_s3_parquet
# ---------------------------------------------------------------------------

class TestGetLatestS3Parquet:
    def _parquet_bytes(self, df):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        return buf.read()

    def test_returns_most_recent_parquet_as_dataframe(self):
        df = pd.DataFrame({"home_name": ["Blue Jays"], "away_name": ["Angels"]})
        raw = self._parquet_bytes(df)
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/2026-05-08.parquet", "LastModified": datetime(2026, 5, 8, tzinfo=timezone.utc)},
                {"Key": "prefix/2026-05-09.parquet", "LastModified": datetime(2026, 5, 9, tzinfo=timezone.utc)},
            ]
        }
        mock_s3.get_object.return_value = {"Body": io.BytesIO(raw)}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            result = s3_client.get_latest_s3_parquet("mlbdk", "prefix/")

        mock_s3.get_object.assert_called_once_with(Bucket="mlbdk", Key="prefix/2026-05-09.parquet")
        assert isinstance(result, pd.DataFrame)

    def test_raises_file_not_found_when_prefix_empty(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            with pytest.raises(FileNotFoundError):
                s3_client.get_latest_s3_parquet("mlbdk", "prefix/")


class TestGetS3Parquet:
    def _parquet_bytes(self, df):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        return buf.read()

    def test_returns_dataframe(self):
        df = pd.DataFrame({"home_team": ["Toronto Blue Jays"], "away_team": ["LA Angels"]})
        raw = self._parquet_bytes(df)
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": io.BytesIO(raw)}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            result = s3_client.get_s3_parquet("mlbdk", "some/key.parquet")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["home_team", "away_team"]

    def test_raises_file_not_found_on_missing_key(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
        )

        with patch("s3_client.boto3.client", return_value=mock_s3):
            with pytest.raises(FileNotFoundError):
                s3_client.get_s3_parquet("mlbdk", "missing.parquet")


class TestReadS3ParquetSeason:
    def _parquet_bytes(self, df):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        return buf.read()

    def test_concatenates_all_files_under_prefix(self):
        df1 = pd.DataFrame({"gamepk": [1, 2]})
        df2 = pd.DataFrame({"gamepk": [3]})
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/2026-04-01.parquet", "LastModified": datetime(2026, 4, 1, tzinfo=timezone.utc)},
                {"Key": "prefix/2026-04-02.parquet", "LastModified": datetime(2026, 4, 2, tzinfo=timezone.utc)},
            ]
        }
        mock_s3.get_object.side_effect = [
            {"Body": io.BytesIO(self._parquet_bytes(df1))},
            {"Body": io.BytesIO(self._parquet_bytes(df2))},
        ]

        with patch("s3_client.boto3.client", return_value=mock_s3):
            result = s3_client.read_s3_parquet_season("mlbdk", "prefix/")

        assert sorted(result["gamepk"].tolist()) == [1, 2, 3]

    def test_returns_empty_dataframe_when_prefix_has_no_objects(self):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            result = s3_client.read_s3_parquet_season("mlbdk", "prefix/")

        assert result.empty

    def test_columns_param_restricts_returned_columns(self):
        df = pd.DataFrame({"gamepk": [1], "extra_col": ["x"]})
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "prefix/2026-04-01.parquet", "LastModified": datetime(2026, 4, 1, tzinfo=timezone.utc)}]
        }
        mock_s3.get_object.return_value = {"Body": io.BytesIO(self._parquet_bytes(df))}

        with patch("s3_client.boto3.client", return_value=mock_s3):
            result = s3_client.read_s3_parquet_season("mlbdk", "prefix/", columns=["gamepk"])

        assert list(result.columns) == ["gamepk"]
