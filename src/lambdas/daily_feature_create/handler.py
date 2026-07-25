import io
import json
import logging
from datetime import datetime, timedelta, timezone
import boto3
import pandas as pd
from feast import FeatureStore
 
from team_batter_base import compute_team_batter_base_features
from player_batter_base import compute_player_batter_base_features
from sp_features import compute_sp_rolling_features
from bullpen_features import compute_bullpen_features
 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
 
BUCKET      = "mlbdk"
REGION      = "us-east-2"
FEAST_PATH  = "/opt/features/"
 
s3 = boto3.client("s3", region_name=REGION)
 


# ----------------------- HELPERS -----------------------------------

def _yesterday():
    return (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')

def _write_snapshot(df: pd.DataFrame, key:str) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    logger.info(f"Wrote {len(df)} rows to s3://{BUCKET}/{key}")
 
def _get_snapshot(df: pd.DataFrame, date: str) -> pd.DataFrame:
    snapshot = df[df["game_date"] == date].copy()
    snapshot["event_timestamp"] = pd.to_datetime(snapshot["game_date"])
    return snapshot

# ----------------------- INDIVIDUAL TRANSFORMS -----------------------------------

def _run_team_batter_base(year:str, date:str) -> str:
    features = compute_team_batter_base_features(year)
    snapshot = _get_snapshot(features, date)
    if snapshot.empty:
        logger.warning(f"[team_batter_base] no rows for {date}")
    _write_snapshot(snapshot, key=f"feast/features/team_batter_base/{year}/{date}.parquet")

    return "ok"


def _run_player_batter_base(year: str, date: str) -> str:
    features = compute_player_batter_base_features(year)
    snapshot = _get_snapshot(features, date)
    if snapshot.empty:
        logger.warning(f"[player_batter_base] no rows for {date}")
        return "empty"
    _write_snapshot(snapshot, f"feast/features/player_batter_base/{year}/{date}.parquet")
    return "ok"
 
 
def _run_sp_base(year: str, date: str) -> str:
    features = compute_sp_rolling_features(year)
    snapshot = _get_snapshot(features, date)
    if snapshot.empty:
        logger.warning(f"[sp_base] no rows for {date}")
        return "empty"
    _write_snapshot(snapshot, f"feast/features/starting_pitcher_base/{year}/{date}.parquet")
    return "ok"
 
 
def _run_bullpen_base(year: str, date: str) -> str:
    features = compute_bullpen_features(year)
    snapshot = _get_snapshot(features, date)
    if snapshot.empty:
        logger.warning(f"[bullpen_base] no rows for {date}")
        return "empty"
    _write_snapshot(snapshot, f"feast/features/bullpen_pitcher_base/{year}/{date}.parquet")
    return "ok"
 
 # ----------------------- HANDLER  -----------------------------------

def lambda_handler(event, context):
    date = event.get("date") or _yesterday() # will default to yesterday if date not passed
    year = date[:4]
    logger.info(f"Feature pipeline starting for {date}")

    transforms = {
        "team_batter_base": _run_team_batter_base,
        "player_batter_base": _run_player_batter_base,
        "sp_base": _run_sp_base,
        "bullpen_base": _run_bullpen_base
    }

    results = {}

    for name, fn in transforms.items():
        try:
            results[name] = fn(year,date) 
            logger.info(f"[{name}] {results[name]} written to S3")
        except Exception as e:
            logger.error(f"[{name}] FAILED: {type(e).__name__}: {e}")
            results[name] = f"error: {e}"


    # only materialize features that worked
    succeeded = [k for k, v in results.items() if v == 'ok']
    failed = [k for k, v in results.items() if v.startswith('error')]

    if succeeded:
        try:
            store = FeatureStore(repo_path=FEAST_PATH)
            store.materialize_incremental(end_date=datetime.now(timezone.utc))
            results['materialize'] = 'ok'
            logger.info("materialization complete")

        except Exception as e:
            logger.error(f"Materialization failed: {e}")
            results['materialize'] = f"error: {e}"

    else:
        logger.info("All transforms failed -- skipping materialization")

        results['materialize'] = "skipped"

    logger.info(f"Pipeline results: {json.dumps(results)}")
    

    return {
        "statusCode": 200 if not failed else 207,
        "body": json.dumps({
            "date":date,
            "succeeded":succeeded,
            "failed":failed,
            "results": results,
        }) 
    }