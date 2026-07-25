import json
import logging
import os
import boto3
import statsapi
from datetime import datetime, timedelta
from status_writer import write_status

logger = logging.getLogger()
logger.setLevel(logging.INFO)

S3_BUCKET = "mlbdk"
# IAM required on this Lambda's role:
#   s3:GetObject      — read schedule + game_states
#   s3:PutObject      — write lineups + game_states + status file
#   events:DisableRule — disable own EventBridge rule on completion
HARD_CUTOFF_MINUTES = int(os.environ.get("HARD_CUTOFF_MINUTES", 30))

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def read_schedule(date):
    s3 = boto3.client("s3")
    year = date[:4]
    obj = s3.get_object(Bucket=S3_BUCKET, Key=f"raw_data/schedule/{year}/{date}.json")
    return json.loads(obj["Body"].read())


def read_game_states(date):
    s3 = boto3.client("s3")
    year = date[:4]
    obj = s3.get_object(Bucket=S3_BUCKET, Key=f"raw_data/games/lineups/states/{year}/{date}.json")
    return json.loads(obj["Body"].read())


def write_game_states(date, states):
    s3 = boto3.client("s3")
    year = date[:4]
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"raw_data/games/lineups/states/{year}/{date}.json",
        Body=json.dumps(states),
        ContentType="application/json",
    )


def write_lineup(date, game_pk, payload):
    s3 = boto3.client("s3")
    year = date[:4]
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"raw_data/games/lineups/{year}/{date}/{game_pk}.json",
        Body=json.dumps(payload),
        ContentType="application/json",
    )


def fetch_boxscore(game_pk):
    return statsapi.get("game_boxscore", {"gamePk": game_pk})


def is_lineup_confirmed(boxscore):
    home = boxscore["teams"]["home"]["battingOrder"]
    away = boxscore["teams"]["away"]["battingOrder"]
    return bool(home) and bool(away)


def fetch_probable_pitchers(game_pk):
    game = statsapi.get("game", {"gamePk": game_pk})
    pitchers = game.get("gameData", {}).get("probablePitchers", {})
    return {
        "home": pitchers.get("home", {}).get("id"),
        "away": pitchers.get("away", {}).get("id"),
    }


def build_lineup_payload(game_pk, date, schedule_game, boxscore, pitchers):
    home_order = boxscore["teams"]["home"]["battingOrder"]
    away_order = boxscore["teams"]["away"]["battingOrder"]
    return {
        "game_pk": game_pk,
        "date": date,
        "confirmed_at": datetime.utcnow().isoformat() + "Z",
        "home_team_id": schedule_game["home_team_id"],
        "away_team_id": schedule_game["away_team_id"],
        "home_lineup": [
            {"batting_order": i + 1, "player_id": pid} for i, pid in enumerate(home_order)
        ],
        "away_lineup": [
            {"batting_order": i + 1, "player_id": pid} for i, pid in enumerate(away_order)
        ],
        "home_pitcher_id": pitchers["home"],
        "away_pitcher_id": pitchers["away"],
    }


def finish(date, states):
    try:
        events_client = boto3.client("events")
        events_client.disable_rule(Name=os.environ["LINEUP_EVENTBRIDGE_RULE_NAME"])
        logger.info("lineup polling rule disabled — all games resolved")
    except Exception as e:
        logger.error(f"failed to disable EventBridge rule: {e}")

    confirmed = sum(1 for s in states["games"].values() if s == "CONFIRMED")
    skipped = sum(1 for s in states["games"].values() if s == "SKIPPED")
    now = datetime.utcnow().isoformat()
    write_status(
        function_name="daily_lineup_fetch",
        run_date=date,
        status="success",
        started_at=now,
        completed_at=now,
        duration_seconds=0,
        games_processed={
            "scheduled": len(states["games"]),
            "confirmed": confirmed,
            "skipped": skipped,
        },
        error=None,
    )


def handler(event, context):
    date = event.get("date") or datetime.utcnow().strftime("%Y-%m-%d")
    started_at = datetime.utcnow()

    try:
        states = read_game_states(date)
    except Exception as e:
        logger.error(f"failed to read game states for {date}: {e}")
        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_lineup_fetch",
            run_date=date,
            status="failed",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed={},
            error=str(e),
        )
        return {"statusCode": 500, "body": str(e)}

    try:
        schedule = read_schedule(date)
        schedule_by_pk = {str(g["game_pk"]): g for g in schedule}

        first_pitch_utc = min(
            datetime.strptime(g["game_time_utc"], "%Y-%m-%dT%H:%M:%SZ") for g in schedule
        )
        cutoff = first_pitch_utc - timedelta(minutes=HARD_CUTOFF_MINUTES)

        if datetime.utcnow() >= cutoff:
            for game_pk, state in states["games"].items():
                if state == "PENDING":
                    states["games"][game_pk] = "SKIPPED"
                    logger.info(f"cutoff reached, skipping: {game_pk}")
            write_game_states(date, states)
            finish(date, states)
            return {"statusCode": 200, "body": "cutoff reached — all pending games skipped"}

        for game_pk, state in states["games"].items():
            if state != "PENDING":
                continue
            try:
                boxscore = fetch_boxscore(game_pk)
                if is_lineup_confirmed(boxscore):
                    pitchers = fetch_probable_pitchers(game_pk)
                    payload = build_lineup_payload(
                        int(game_pk), date, schedule_by_pk[game_pk], boxscore, pitchers
                    )
                    write_lineup(date, game_pk, payload)
                    states["games"][game_pk] = "CONFIRMED"
                    logger.info(f"lineup confirmed: {game_pk}")
            except Exception as e:
                logger.warning(f"boxscore fetch failed for {game_pk}: {e}")

        states["last_checked"] = datetime.utcnow().isoformat()
        write_game_states(date, states)

        if all(s != "PENDING" for s in states["games"].values()):
            finish(date, states)

        return {"statusCode": 200, "body": f"Done for {date}"}

    except Exception as e:
        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_lineup_fetch",
            run_date=date,
            status="failed",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed={},
            error=str(e),
        )
        return {"statusCode": 500, "body": str(e)}
