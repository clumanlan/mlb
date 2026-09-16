import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import io

import boto3
import pandas as pd

from batter_lineup import compute_batter_lineup_features
from batter_pa_volume import compute_batter_pa_volume_features

BUCKET = "mlbdk"


def _write_parquet_to_s3(df: pd.DataFrame, key: str) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())


def write_batter_lineup_backfill(year: str) -> str:
    df = compute_batter_lineup_features(year)
    key = f"feast/features/batter_lineup/{year}/backfill.parquet"
    _write_parquet_to_s3(df, key)
    return key


def write_batter_pa_volume_backfill(year: str) -> str:
    df = compute_batter_pa_volume_features(year)
    key = f"feast/features/batter_pa_volume/{year}/backfill.parquet"
    _write_parquet_to_s3(df, key)
    return key


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2024"
    print(write_batter_lineup_backfill(year))
    print(write_batter_pa_volume_backfill(year))
