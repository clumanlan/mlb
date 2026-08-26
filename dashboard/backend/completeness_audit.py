from datetime import datetime, timezone

import s3_client

TABLE_PREFIXES = {
    "batter_boxscore": "processed_data/prepared/batter_boxscore/{year}/",
    "pitcher_boxscore": "processed_data/prepared/pitcher_boxscore/{year}/",
    "playbyplay": "processed_data/prepared/playbyplay/{year}/",
}

SCHEDULE_PREFIX = "processed_data/games/schedule/{year}/"


def diff_game_coverage(scheduled_pks: set, table_pks: dict) -> dict:
    """Compare a set of scheduled gamepks against each table's gamepk set."""
    result = {}
    all_complete = True
    for table_name, pks in table_pks.items():
        missing = sorted(scheduled_pks - pks)
        result[table_name] = {"missing_count": len(missing), "missing_gamepks": missing}
        if missing:
            all_complete = False
    result["is_complete"] = all_complete
    return result


def run_season_completeness_audit(bucket: str, year: str, season_type: str = "R") -> dict:
    """
    Compare the regular-season schedule against each table k_predictor depends on
    (batter_boxscore, pitcher_boxscore, playbyplay, all at processed_data/prepared/)
    and report missing games per table.
    """
    schedule = s3_client.read_s3_parquet_season(
        bucket, SCHEDULE_PREFIX.format(year=year), columns=["gamepk", "game_type", "game_date"]
    )
    schedule = schedule[schedule["game_type"] == season_type]
    scheduled_pks = set(schedule["gamepk"].tolist())
    game_dates = dict(zip(schedule["gamepk"], schedule["game_date"]))

    table_pks = {}
    for table_name, prefix_template in TABLE_PREFIXES.items():
        df = s3_client.read_s3_parquet_season(bucket, prefix_template.format(year=year), columns=["gamepk"])
        table_pks[table_name] = set(df["gamepk"].tolist())

    diff = diff_game_coverage(scheduled_pks, table_pks)

    tables = {}
    for table_name in TABLE_PREFIXES:
        missing_pks = diff[table_name]["missing_gamepks"]
        tables[table_name] = {
            "missing_count": diff[table_name]["missing_count"],
            "missing_games": [{"gamepk": pk, "game_date": game_dates.get(pk)} for pk in missing_pks],
        }

    return {
        "year": year,
        "total_scheduled_games": len(scheduled_pks),
        "tables": tables,
        "is_complete": diff["is_complete"],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
