from datetime import date, timedelta, datetime
from typing import Optional
import zoneinfo

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import completeness_audit
import s3_client

CT = zoneinfo.ZoneInfo("America/Chicago")


def utc_to_ct(utc_str: str) -> str:
    """Convert a UTC ISO datetime string to a Central Time h:MM AM/PM string."""
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    ct = dt.astimezone(CT)
    hour = ct.hour % 12 or 12
    ampm = "AM" if ct.hour < 12 else "PM"
    return f"{hour}:{ct.strftime('%M')} {ampm}"

load_dotenv()

app = FastAPI(title="MLB Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

S3_BUCKET = "mlbdk"


def _normalize_schedule_parquet(df) -> list:
    """Convert legacy daily_mlb_fetch schedule Parquet columns to the standard game dict shape."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "game_pk": int(row["game_id"]),
            "home_team_name": row["home_name"],
            "away_team_name": row["away_name"],
            "game_time_utc": row["game_datetime"],
            "venue_name": row["venue_name"],
            "status": row["status"],
        })
    return rows


def _load_schedule(year: str, date_str: str, use_latest: bool) -> list:
    """
    Load schedule games from S3 for an exact date. Never falls back to a different
    date — if today's file isn't there yet, return [] so the UI can say "not ready."

    Two legacy naming conventions exist in S3 (both tried):
      - raw_data/games/schedule/{year}/{date}.parquet        (newer)
      - raw_data/games/schedule/{year}/schedule_{date}.parquet  (older)
    """
    # New JSON path (daily_schedule_fetch Lambda)
    try:
        if use_latest:
            return s3_client.get_latest_s3_json(S3_BUCKET, f"raw_data/schedule/{year}/")
        return s3_client.get_s3_json(S3_BUCKET, f"raw_data/schedule/{year}/{date_str}.json")
    except FileNotFoundError:
        pass

    # Legacy Parquet fallback — try both naming conventions for the exact date
    legacy_prefix = f"raw_data/games/schedule/{year}/"
    for key in [
        f"{legacy_prefix}{date_str}.parquet",
        f"{legacy_prefix}schedule_{date_str}.parquet",
    ]:
        try:
            df = s3_client.get_s3_parquet(S3_BUCKET, key)
            return _normalize_schedule_parquet(df)
        except FileNotFoundError:
            continue
    return []

CARD_CONFIGS = [
    {"key": "daily_mlb_fetch",    "title": "Raw — MLB fetch",   "staleness": "yesterday"},
    {"key": "daily_process_data", "title": "Processed data",     "staleness": "yesterday"},
    {"key": "daily_odds_fetch",   "title": "Odds fetch",         "staleness": "today"},
]


def is_stale_mlb_fetch(run_date: str, today: date) -> bool:
    return date.fromisoformat(run_date) != (today - timedelta(days=1))


def is_stale_odds_fetch(run_date: str, today: date) -> bool:
    return date.fromisoformat(run_date) != today


def _build_card(config: dict, today: date) -> dict:
    try:
        data = s3_client.get_latest_s3_json(S3_BUCKET, f"lambdas/status/{config['key']}/")
        run_date = data.get("run_date", "")
        if config["staleness"] == "yesterday":
            stale = is_stale_mlb_fetch(run_date, today)
        else:
            stale = is_stale_odds_fetch(run_date, today)
        return {
            "title": config["title"],
            "run_date": run_date,
            "completed_at": data.get("completed_at"),
            "duration_seconds": data.get("duration_seconds"),
            "is_stale": stale,
            "games_processed": data.get("games_processed", {}),
            "error": data.get("error"),
        }
    except FileNotFoundError as e:
        return {
            "title": config["title"],
            "run_date": None,
            "completed_at": None,
            "duration_seconds": None,
            "is_stale": True,
            "games_processed": {},
            "error": str(e),
        }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/today-slate")
def today_slate(date_param: Optional[str] = Query(default=None, alias="date")):
    """
    Return today's game slate with lineup and odds status.
    Pass ?date=YYYY-MM-DD to query a specific historical date (useful for debugging).
    """
    target_date = date_param or str(date.today())
    year = target_date[:4]

    use_latest = date_param is None
    schedule = _load_schedule(year, target_date, use_latest)

    try:
        if use_latest:
            lineup_states = s3_client.get_latest_s3_json(S3_BUCKET, f"raw_data/games/lineups/states/{year}/")
        else:
            lineup_states = s3_client.get_s3_json(S3_BUCKET, f"raw_data/games/lineups/states/{year}/{target_date}.json")
    except FileNotFoundError:
        lineup_states = {}

    try:
        odds_df = s3_client.get_s3_parquet(S3_BUCKET, f"raw_data/odds/team_odds/{year}/{target_date}.parquet")
        odds_pairs = set(zip(odds_df["home_team"], odds_df["away_team"]))
    except FileNotFoundError:
        odds_pairs = set()

    games = []
    for game in sorted(schedule, key=lambda g: g["game_time_utc"]):
        game_pk_str = str(game["game_pk"])
        lineup_status = lineup_states.get("games", {}).get(game_pk_str, "PENDING")
        home = game["home_team_name"]
        away = game["away_team_name"]
        games.append({
            "game_pk": game["game_pk"],
            "away_team": away,
            "home_team": home,
            "game_time_utc": game["game_time_utc"],
            "game_time_ct": utc_to_ct(game["game_time_utc"]),
            "venue": game["venue_name"],
            "lineup_status": lineup_status,
            "has_odds": (home, away) in odds_pairs,
            "prediction": None,
        })
        
    lineup_last_checked = lineup_states.get("last_checked") if isinstance(lineup_states, dict) else None
    return {"date": target_date, "games": games, "games_sorted_by_time": True, "lineup_last_checked": lineup_last_checked}


@app.get("/api/season-completeness")
def season_completeness(year: Optional[str] = Query(default=None)):
    """
    Audit a season's schedule vs. the batter_boxscore/pitcher_boxscore/playbyplay
    tables k_predictor depends on. Defaults to the current year.
    """
    target_year = year or str(date.today().year)
    return completeness_audit.run_season_completeness_audit(S3_BUCKET, target_year)


@app.get("/api/pipeline-status")
def pipeline_status():
    today = date.today()
    cards = [_build_card(cfg, today) for cfg in CARD_CONFIGS]
    ok_count = sum(1 for c in cards if not c["is_stale"] and not c["error"])
    stale_count = sum(1 for c in cards if c["is_stale"])
    return {"cards": cards, "summary": {"ok_count": ok_count, "stale_count": stale_count}}
