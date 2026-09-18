import json
import logging
import os
from datetime import datetime, timezone
from feast import FeatureStore

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FEAST_PATH = os.path.dirname(os.path.abspath(__file__))


def lambda_handler(event, context):
    try:
        store = FeatureStore(repo_path=FEAST_PATH)
        store.materialize_incremental(end_date=datetime.now(timezone.utc))
        logger.info("materialization complete")
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
