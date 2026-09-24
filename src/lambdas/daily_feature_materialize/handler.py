import json
import logging
import os
from datetime import datetime, timezone
import boto3
from feast import FeatureStore

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = "us-east-2"
FEAST_PATH = os.path.dirname(os.path.abspath(__file__))
PREDICT_FUNCTION_NAME = "daily_predict"

lambda_client = boto3.client("lambda", region_name=REGION)


def lambda_handler(event, context):
    try:
        store = FeatureStore(repo_path=FEAST_PATH)
        store.materialize_incremental(end_date=datetime.now(timezone.utc))
        logger.info("materialization complete")

        try:
            lambda_client.invoke(FunctionName=PREDICT_FUNCTION_NAME, InvocationType='Event')
            logger.info(f"Invoked {PREDICT_FUNCTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to invoke {PREDICT_FUNCTION_NAME}: {type(e).__name__}: {e}")

        return {
            "statusCode": 200,
            "body": json.dumps({"status": "ok"})
        }
    except Exception as e:
        logger.error(f"Materialization failed: {type(e).__name__}: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "error": str(e)})
        }
