import json
from datetime import datetime, timezone

import boto3
import xgboost as xgb

BUCKET = "mlbdk"


def snapshot_feature_schema(feature_views):
    return {
        fv.name: [{"name": field.name, "dtype": str(field.dtype)} for field in fv.schema]
        for fv in feature_views
    }


def register_model(model_name, model, feature_schema, operating_threshold, metrics, bucket=BUCKET):
    version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    model_json = json.loads(model.get_booster().save_raw(raw_format="json"))

    payload = {
        "model_name": model_name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema": feature_schema,
        "operating_threshold": operating_threshold,
        "metrics": metrics,
        "model": model_json,
    }
    body = json.dumps(payload)

    s3 = boto3.client("s3")
    for key in (
        f"models/{model_name}/{version}.json",
        f"models/{model_name}/latest.json",
    ):
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

    return version


def load_model(model_name, bucket=BUCKET):
    s3 = boto3.client("s3")
    key = f"models/{model_name}/latest.json"
    payload = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())

    model = xgb.XGBClassifier()
    model.load_model(bytearray(json.dumps(payload["model"]).encode("utf-8")))

    return {
        "model": model,
        "version": payload["version"],
        "operating_threshold": payload["operating_threshold"],
        "feature_schema": payload["feature_schema"],
        "metrics": payload["metrics"],
    }
