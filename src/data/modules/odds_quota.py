import boto3
from botocore.exceptions import ClientError


def get_monthly_usage(year, month):
    ssm = boto3.client("ssm")
    try:
        response = ssm.get_parameter(Name=f"/mlb/odds-api/requests-used/{year}/{month}")
        return int(response["Parameter"]["Value"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "ParameterNotFound":
            return 0
        raise


def check_quota(year, month, limit=500, warn_threshold=0.9):
    current = get_monthly_usage(year, month)
    stop = current >= limit
    warn = current >= limit * warn_threshold
    return {"current": current, "warn": warn, "stop": stop}


def increment_monthly_usage(year, month, current_count):
    ssm = boto3.client("ssm")
    ssm.put_parameter(
        Name=f"/mlb/odds-api/requests-used/{year}/{month}",
        Value=str(current_count + 1),
        Type="String",
        Overwrite=True,
    )
