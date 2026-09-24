import io
import json
import logging
import os
from datetime import datetime, timezone

import boto3
from feast import FeatureStore

from model_registry import load_model
from status_writer import write_status
from predict import predict_for_date

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = "mlbdk"
REGION = "us-east-2"
MODEL_NAME = "n_pa_predictor_low_pa"
FEAST_PATH = os.path.dirname(os.path.abspath(__file__))

s3 = boto3.client("s3", region_name=REGION)


def _today():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def _write_predictions(df, date):
    year = date[:4]
    key = f"predictions/{MODEL_NAME}/{year}/{date}.parquet"
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    return key


def lambda_handler(event, context):
    date = event.get("date") or _today()
    started_at = datetime.utcnow()
    games_processed = {}
    try:
        feature_store = FeatureStore(repo_path=FEAST_PATH)
        model_info = load_model(MODEL_NAME)
        result = predict_for_date(date, feature_store, model_info)
        key = _write_predictions(result, date)

        games_processed = {
            "batters_predicted": len(result),
            "qualifying_predictions": int(result["qualifies"].sum()) if len(result) else 0,
        }

        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_predict", run_date=date, status="success",
            started_at=started_at.isoformat(), completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed=games_processed, error=None,
        )
        logger.info(f"Wrote {len(result)} predictions to s3://{BUCKET}/{key}")
        return {
            "statusCode": 200,
            "body": json.dumps({"date": date, "rows": len(result), "key": key}),
        }
    except Exception as e:
        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_predict", run_date=date, status="failed",
            started_at=started_at.isoformat(), completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed=games_processed, error=str(e),
        )
        logger.error(f"daily_predict FAILED: {type(e).__name__}: {e}")
        raise
