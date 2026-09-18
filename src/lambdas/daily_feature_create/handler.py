import io
import json
import logging
from datetime import datetime, timezone
import boto3
import pandas as pd

from batter_lineup import compute_batter_lineup_features_for_date
from batter_pa_volume import compute_batter_pa_volume_features_for_date

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET      = "mlbdk"
REGION      = "us-east-2"
MATERIALIZE_FUNCTION_NAME = "daily_feature_materialize"

s3 = boto3.client("s3", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)


# ----------------------- HELPERS -----------------------------------

def _today():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def _date_from_event(event):
    """Direct-invoke shape ({"date": "..."}, used for manual smoke tests —
    see CLAUDE.md) is unchanged. S3-event shape (this Lambda's real trigger,
    fired by daily_lineup_fetch's per-game lineup writes) carries the date
    in the object key: raw_data/games/lineups/{year}/{date}/{game_pk}.json.
    Returns (date, is_states_file) — daily_lineup_fetch also writes a
    states-tracking file to the same prefix on every poll cycle regardless
    of whether anything new confirmed; S3's prefix/suffix-only filters
    can't exclude it, so this must be detected and skipped here instead."""
    records = event.get("Records")
    if not records:
        return event.get("date") or _today(), False

    key = records[0]["s3"]["object"]["key"]
    parts = key.split("/")
    # raw_data/games/lineups/{year}/{date}/{game_pk}.json
    # raw_data/games/lineups/states/{year}/{date}.json
    if parts[3] == "states":
        return None, True
    return parts[4], False

def _write_snapshot(df: pd.DataFrame, key: str) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    logger.info(f"Wrote {len(df)} rows to s3://{BUCKET}/{key}")

# ----------------------- INDIVIDUAL TRANSFORMS -----------------------------------

def _run_batter_lineup(date: str) -> str:
    year = date[:4]
    features = compute_batter_lineup_features_for_date(date)
    if features.empty:
        logger.warning(f"[batter_lineup] no rows for {date}")
        return "empty"
    _write_snapshot(features, key=f"feast/features/batter_lineup/{year}/{date}.parquet")
    return "ok"


def _run_batter_pa_volume(date: str) -> str:
    year = date[:4]
    features = compute_batter_pa_volume_features_for_date(date)
    if features.empty:
        logger.warning(f"[batter_pa_volume] no rows for {date}")
        return "empty"
    _write_snapshot(features, key=f"feast/features/batter_pa_volume/{year}/{date}.parquet")
    return "ok"

# ----------------------- HANDLER  -----------------------------------

def lambda_handler(event, context):
    date, is_states_file = _date_from_event(event)
    if is_states_file:
        logger.info("Skipping states-tracking file event, not a lineup confirmation")
        return {"statusCode": 200, "body": json.dumps({"skipped": "states file, not a lineup"})}

    logger.info(f"Feature pipeline starting for {date}")

    transforms = {
        "batter_lineup": _run_batter_lineup,
        "batter_pa_volume": _run_batter_pa_volume,
    }

    results = {}
    for name, fn in transforms.items():
        try:
            results[name] = fn(date)
            logger.info(f"[{name}] {results[name]} written to S3")
        except Exception as e:
            logger.error(f"[{name}] FAILED: {type(e).__name__}: {e}")
            results[name] = f"error: {e}"

    succeeded = [k for k, v in results.items() if v == 'ok']
    failed = [k for k, v in results.items() if v.startswith('error')]

    if succeeded:
        try:
            lambda_client.invoke(FunctionName=MATERIALIZE_FUNCTION_NAME, InvocationType='Event')
            logger.info(f"Invoked {MATERIALIZE_FUNCTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to invoke {MATERIALIZE_FUNCTION_NAME}: {type(e).__name__}: {e}")

    logger.info(f"Pipeline results: {json.dumps(results)}")
    return {
        "statusCode": 200 if not failed else 207,
        "body": json.dumps({
            "date": date,
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
        })
    }
