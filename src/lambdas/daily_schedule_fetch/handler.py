import json
import logging
import os
import boto3
import requests
from datetime import datetime
from status_writer import write_status


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
S3_BUCKET = "mlbdk"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def fetch_schedule(date):
    response = requests.get(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": date, "hydrate": "venue"},
    )
    response.raise_for_status()
    data = response.json()
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            games.append({
                "game_pk": game["gamePk"],
                "home_team_id": game["teams"]["home"]["team"]["id"],
                "home_team_name": game["teams"]["home"]["team"]["name"],
                "away_team_id": game["teams"]["away"]["team"]["id"],
                "away_team_name": game["teams"]["away"]["team"]["name"],
                "game_time_utc": game["gameDate"],
                "venue_name": game["venue"]["name"],
                "status": game["status"]["detailedState"],
            })
    return games


def run(date):
    year = date[:4]
    games = fetch_schedule(date)
    if len(games) == 0:
        logger.warning(f"No games found for {date} — off day or All-Star break.")
    # S3 path for downstream consumers: raw_data/schedule/{year}/{date}.json
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"raw_data/schedule/{year}/{date}.json",
        Body=json.dumps(games),
        ContentType="application/json",
    )
    # Write initial game states for lineup fetch
    states = {
        "date": date,
        "last_checked": None,
        "games": {str(game["game_pk"]): "PENDING" for game in games},
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"raw_data/games/lineups/states/{year}/{date}.json",
        Body=json.dumps(states),
        ContentType="application/json",
    )
    logger.info(f"game states written: {len(states['games'])} games PENDING")
    # Enables lineup polling rule as final step — do not move this before the S3 writes
    events_client = boto3.client("events")
    events_client.enable_rule(Name=os.environ["LINEUP_EVENTBRIDGE_RULE_NAME"])
    logger.info("lineup polling rule enabled")
    return {"scheduled_games": len(games)}

def handler(event, context):
    date = event.get("date") or datetime.utcnow().strftime("%Y-%m-%d")
    started_at = datetime.utcnow()
    games_processed = {}
    try:
        games_processed = run(date)
        completed_at = datetime.utcnow()

        if games_processed["scheduled_games"] == 0:
            write_status(
                function_name="daily_schedule_fetch",
                run_date=date,
                status="no_games",
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                duration_seconds=(completed_at - started_at).total_seconds(),
                games_processed=games_processed,
                error=None,
            )
            return {"statusCode": 200, "body": f"No games on {date} — lineup rule not enabled"}

        write_status(
            function_name="daily_schedule_fetch",
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
            function_name="daily_schedule_fetch",
            run_date=date,
            status="failed",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed=games_processed,
            error=str(e),
        )
        return {"statusCode": 500, "body": str(e)}
    
