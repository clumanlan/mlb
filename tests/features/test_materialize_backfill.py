import io
from unittest.mock import patch

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from materialize_backfill import write_batter_lineup_backfill, write_batter_pa_volume_backfill

BUCKET = "mlbdk"


@pytest.fixture()
def aws_infra(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
        )
        yield s3


def test_write_batter_lineup_backfill_writes_expected_s3_key(aws_infra):
    """Feast's FileSource for batter_lineup needs real Parquet at a real S3
    path with event_timestamp present — compute_batter_lineup_features on
    its own only ever returns a DataFrame, it never writes anywhere."""
    fake_df = pd.DataFrame([
        {"personId": "100", "gamepk": "1", "event_timestamp": pd.Timestamp("2024-04-01"),
         "batting_order": 3},
    ])
    with patch("materialize_backfill.compute_batter_lineup_features", return_value=fake_df):
        key = write_batter_lineup_backfill("2024")

    assert key == "feast/features/batter_lineup/2024/backfill.parquet"
    obj = aws_infra.get_object(Bucket=BUCKET, Key=key)
    result = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    pd.testing.assert_frame_equal(result, fake_df)


def test_write_batter_pa_volume_backfill_writes_expected_s3_key(aws_infra):
    fake_df = pd.DataFrame([
        {"personId": "100", "gamepk": "1", "event_timestamp": pd.Timestamp("2024-04-01"),
         "batter_pa_roll_season_games_n": 3, "batter_pa_roll_season_avg_n_pa_per_game": 4.0},
    ])
    with patch("materialize_backfill.compute_batter_pa_volume_features", return_value=fake_df):
        key = write_batter_pa_volume_backfill("2024")

    assert key == "feast/features/batter_pa_volume/2024/backfill.parquet"
    obj = aws_infra.get_object(Bucket=BUCKET, Key=key)
    result = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    pd.testing.assert_frame_equal(result, fake_df)
