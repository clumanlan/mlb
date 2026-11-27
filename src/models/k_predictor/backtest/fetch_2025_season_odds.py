"""
Full-season 2025 pitcher_strikeouts historical odds pull, for a larger-n
k_predictor edge backtest than fetch_2025_odds.py's 7-date sample.

Credits reset on the 1st of every month and don't roll over, so this is
meant to be re-run each month during the off-season: it skips dates
already saved IN S3 (the record of what's been paid for, so a rerun from
any machine never re-buys a date) and stops before dipping into RESERVE,
picking up where the last run left off. Each date is written to S3 as soon
as it's fetched, so a crash or lost laptop costs at most the in-flight day.

Per date: list games (10 credits), cross-check the count against
raw_data/games/schedule/2025/ BEFORE spending on props, then 10 credits
per game for props. A day whose listing doesn't match the schedule is
logged and skipped, not paid for.

One-off backtest script, not a Lambda. Run from repo root:
    python src/models/k_predictor/backtest/fetch_2025_season_odds.py
"""
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../data/modules"))

PLAN_LIMIT = 20_000
RESERVE = 500
COMPLETED_STATUSES = {"Final", "Completed Early", "Completed Early: Rain"}
BUCKET = "mlbdk"
S3_PREFIX = "raw_data/odds/historical/pitcher_strikeouts/2025/"

logger = logging.getLogger(__name__)


def expected_games_by_date(schedule):
    done = schedule[(schedule.game_type == "R") & (schedule.status.isin(COMPLETED_STATUSES))]
    done = done.drop_duplicates("game_id", keep="last")
    return {str(d): int(n) for d, n in done.groupby("game_date").size().items()}


def dates_to_fetch(expected, already_saved):
    return sorted(d for d in expected if d not in already_saved)


def day_cost(n_games):
    return 10 + 10 * n_games


def has_budget_for(n_games, used, limit=PLAN_LIMIT, reserve=RESERVE):
    return limit - used - day_cost(n_games) >= reserve


def listing_matches_schedule(n_listed, n_expected):
    return n_listed >= n_expected - 1


PROP_COLUMNS = ["date", "event_id", "away_team", "home_team", "commence_time",
                "pitcher_name", "side", "line", "price", "bookmaker"]


def with_prop_schema(df):
    return df if len(df.columns) else pd.DataFrame(columns=PROP_COLUMNS)


def s3_key_for(date):
    return f"{S3_PREFIX}{date}.parquet"


def saved_dates_from_keys(keys):
    return {k.rsplit("/", 1)[-1].removesuffix(".parquet") for k in keys}


def main():
    import awswrangler as wr
    from fetch_2025_odds import flatten_props, get_api_key
    from odds_fetch import get_all_historical_player_props, get_historical_games, get_last_usage

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    api_key = get_api_key()

    schedule = wr.s3.read_parquet(
        "s3://mlbdk/raw_data/games/schedule/2025/",
        columns=["game_id", "game_date", "game_type", "status"],
    )
    expected = expected_games_by_date(schedule)
    saved = saved_dates_from_keys(wr.s3.list_objects(f"s3://{BUCKET}/{S3_PREFIX}"))
    todo = dates_to_fetch(expected, saved)
    logger.info(f"{len(expected)} season dates, {len(saved)} already saved, {len(todo)} to go")

    used = None
    flagged = []
    for date in todo:
        n_expected = expected[date]
        if used is not None and not has_budget_for(n_expected, used):
            logger.info(f"Budget stop before {date}: used={used}, next day ~{day_cost(n_expected)}")
            break

        games = get_historical_games(date, f"{date}T11:00:00Z", api_key)
        used = get_last_usage()
        if not listing_matches_schedule(len(games), n_expected):
            logger.warning(f"{date}: listing has {len(games)} games, schedule has {n_expected} -- skipping props")
            flagged.append((date, len(games), n_expected))
            continue
        if not has_budget_for(len(games), used):
            logger.info(f"Budget stop at {date}: used={used}")
            break

        game_props = get_all_historical_player_props(date, api_key, games=games)
        df = with_prop_schema(flatten_props(date, game_props))
        wr.s3.to_parquet(df, f"s3://{BUCKET}/{s3_key_for(date)}", index=False)
        used = get_last_usage()
        logger.info(f"{date}: {len(games)}/{n_expected} listed, {df.event_id.nunique() if len(df) else 0} with K lines, "
                    f"{len(df)} rows, used={used}")

    logger.info(f"Done. used={used}. Flagged (not paid for): {flagged}")


if __name__ == "__main__":
    main()
