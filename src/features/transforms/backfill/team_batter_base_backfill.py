import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import pandas as pd
import statsapi
import boto3
import io

from team_batter_base import compute_team_batter_base_features

BUCKET = "mlbdk"
REGION = "us-east-2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3", region_name=REGION)


def run_feature_backfill(season: int, write: bool = True):
    season_lookup = statsapi.get("season", {"seasonId": season, "sportId": 1})["seasons"][0]
    start_date = season_lookup["regularSeasonStartDate"]
    end_date = season_lookup["regularSeasonEndDate"]
    logger.info(f"Season {season}: {start_date} -> {end_date}")

    team_base_features = compute_team_batter_base_features(str(season))

    date_window = team_base_features["game_date"].between(start_date, end_date)
    team_base_features = team_base_features[date_window]
    logger.info(f"Filtered to {len(team_base_features)} rows within season window")

    for date in sorted(team_base_features["game_date"].unique()):
        snapshot = team_base_features[team_base_features["game_date"] == date].copy()
        snapshot["event_timestamp"] = pd.to_datetime(snapshot["game_date"])

        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        year = date_str[:4]
        key = f"feast/features/team_batter_base/{year}/{date_str}.parquet"

        if write:
            buffer = io.BytesIO()
            snapshot.to_parquet(buffer, index=False)
            buffer.seek(0)
            s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
            logger.info(f"Wrote {len(snapshot)} rows to s3://{BUCKET}/{key}")
        else:
            logger.info(f"DRY RUN — would write {len(snapshot)} rows to {key}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="team base backfill")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--write", action="store_true", help='Actually write to s3')
    args = parser.parse_args()

    run_feature_backfill(season=args.season, write=args.write)
