import json
from unittest.mock import patch, MagicMock


class TestWriteStatus:
    def _call(self, **kwargs):
        defaults = {
            "function_name": "daily_mlb_fetch",
            "run_date": "2026-05-03",
            "status": "success",
            "started_at": "2026-05-03T10:00:00.000000",
            "completed_at": "2026-05-03T10:01:23.000000",
            "duration_seconds": 83.0,
            "games_processed": {"schedule": 15, "game_info": 15},
        }
        defaults.update(kwargs)
        from status_writer import write_status
        return write_status(**defaults)

    def test_write_status_uses_correct_s3_key(self):
        mock_s3 = MagicMock()
        with patch("status_writer.boto3.client", return_value=mock_s3):
            self._call(function_name="daily_mlb_fetch", run_date="2026-05-03")
        key = mock_s3.put_object.call_args.kwargs["Key"]
        assert key == "lambdas/status/daily_mlb_fetch/2026-05-03.json"

    def test_write_status_payload_contains_all_fields(self):
        mock_s3 = MagicMock()
        with patch("status_writer.boto3.client", return_value=mock_s3):
            self._call()
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"])
        for field in ("function_name", "run_date", "status", "started_at",
                      "completed_at", "duration_seconds", "games_processed", "error"):
            assert field in body, f"missing field: {field}"

    def test_write_status_error_is_null_when_not_provided(self):
        mock_s3 = MagicMock()
        with patch("status_writer.boto3.client", return_value=mock_s3):
            self._call()
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"])
        assert body["error"] is None

    def test_write_status_error_is_string_when_set(self):
        mock_s3 = MagicMock()
        with patch("status_writer.boto3.client", return_value=mock_s3):
            self._call(error="something exploded")
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"])
        assert body["error"] == "something exploded"

    def test_write_status_games_processed_stored_as_dict(self):
        mock_s3 = MagicMock()
        games = {"schedule": 15, "batter_boxscore": 14}
        with patch("status_writer.boto3.client", return_value=mock_s3):
            self._call(games_processed=games)
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"])
        assert body["games_processed"] == games

    def test_write_status_uses_mlbdk_bucket_by_default(self):
        mock_s3 = MagicMock()
        with patch("status_writer.boto3.client", return_value=mock_s3):
            self._call()
        bucket = mock_s3.put_object.call_args.kwargs["Bucket"]
        assert bucket == "mlbdk"

    def test_write_status_content_type_is_json(self):
        mock_s3 = MagicMock()
        with patch("status_writer.boto3.client", return_value=mock_s3):
            self._call()
        ct = mock_s3.put_object.call_args.kwargs["ContentType"]
        assert ct == "application/json"
