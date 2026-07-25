import logging
import boto3
import awswrangler as wr
import pandas as pd
from datetime import datetime


from odds_quota import check_quota, increment_monthly_usage
from odds_fetch import get_all_player_props, get_team_odds
from status_writer import write_status

QUOTA_LIMIT = 500
WARN_THRESHOLD = 0.9
S3_BUCKET = "mlbdk"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_api_key():
    ssm = boto3.client("ssm")
    response = ssm.get_parameter(Name="/mlb/odds-api/api-key", WithDecryption=True)
    return response["Parameter"]["Value"]


def fetch_and_store(date, api_key, current_count):
    year, month = date[:4], date[5:7]

    team_odds = get_team_odds(date, api_key=api_key)
    wr.s3.to_parquet(
        df=pd.DataFrame(team_odds),
        path=f"s3://{S3_BUCKET}/raw_data/odds/team_odds/{year}/{date}.parquet",
    )

    player_props = get_all_player_props(date, api_key=api_key)
    wr.s3.to_parquet(
        df=pd.DataFrame(player_props),
        path=f"s3://{S3_BUCKET}/raw_data/odds/player_props/{year}/{date}.parquet",
    )

    increment_monthly_usage(year=year, month=month, current_count=current_count)
    logger.info(f"Requests used this month: {current_count + 1}/{QUOTA_LIMIT}")

    return {
        "team_odds": len(team_odds),
        "player_props": len(player_props),
    }


def run(date):
    year, month = date[:4], date[5:7]

    quota = check_quota(year=year, month=month, limit=QUOTA_LIMIT, warn_threshold=WARN_THRESHOLD)
    if quota["stop"]:
        raise Exception(f"Monthly quota reached ({quota['current']}/{QUOTA_LIMIT}). Aborting.")
    if quota["warn"]:
        logger.warning(f"quota at {quota['current']}/{QUOTA_LIMIT} — approaching limit.")

    api_key = get_api_key()
    return fetch_and_store(date, api_key, current_count=quota["current"])


def handler(event, context):
    date = event.get("date") or datetime.utcnow().strftime("%Y-%m-%d")
    started_at = datetime.utcnow()
    games_processed = {}
    try:
        games_processed = run(date)
        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_odds_fetch",
            run_date=date,
            status="success",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed=games_processed,
            error=None,
        )
        return {"statusCode": 200, "body": f"Done for {date}"}
    except Exception as e:
        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_odds_fetch",
            run_date=date,
            status="failed",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed=games_processed,
            error=str(e),
        )
        return {"statusCode": 500, "body": str(e)}
