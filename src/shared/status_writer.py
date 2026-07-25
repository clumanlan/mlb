import json
import boto3

BUCKET = "mlbdk"


def write_status(
    function_name,
    run_date,
    status,
    started_at,
    completed_at,
    duration_seconds,
    games_processed,
    error=None,
    bucket=BUCKET,
):
    key = f"lambdas/status/{function_name}/{run_date}.json"
    payload = {
        "function_name": function_name,
        "run_date": run_date,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "games_processed": games_processed,
        "error": error,
    }
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )
