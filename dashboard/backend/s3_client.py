import io
import json

import boto3
import pandas as pd
from botocore.exceptions import ClientError


def get_latest_s3_json(bucket: str, prefix: str) -> dict | list:
    """List all objects under prefix, return parsed JSON from the most recently modified one."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])
    if not contents:
        raise FileNotFoundError(f"No objects found at s3://{bucket}/{prefix}")
    latest = max(contents, key=lambda o: o["LastModified"])
    obj = s3.get_object(Bucket=bucket, Key=latest["Key"])
    return json.loads(obj["Body"].read())


def get_s3_json(bucket: str, key: str) -> dict | list:
    """Fetch and parse JSON from an exact S3 key."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise FileNotFoundError(f"Not found: s3://{bucket}/{key}") from e
        raise


def get_latest_s3_parquet(bucket: str, prefix: str) -> pd.DataFrame:
    """List all objects under prefix, fetch the most recently modified one as a DataFrame."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])
    if not contents:
        raise FileNotFoundError(f"No objects found at s3://{bucket}/{prefix}")
    latest = max(contents, key=lambda o: o["LastModified"])
    obj = s3.get_object(Bucket=bucket, Key=latest["Key"])
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def get_s3_parquet(bucket: str, key: str) -> pd.DataFrame:
    """Fetch a Parquet file from S3 and return it as a DataFrame."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise FileNotFoundError(f"Not found: s3://{bucket}/{key}") from e
        raise
