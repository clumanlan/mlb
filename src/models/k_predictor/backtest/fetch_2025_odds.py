"""
k_predictor production backtest -- Epic 1: pull real 2025 pitcher_strikeouts
odds for a small, user-approved sample of dates.

No 2025 odds data exists anywhere in S3 (the live daily_odds_fetch pipeline
only started ~2026-05), so this uses The Odds API's historical endpoint
(10x the normal markets x regions credit rate -- see odds_fetch.py's
get_historical_* functions) instead. Scoped to exactly 7 dates and one
market (pitcher_strikeouts) to keep the one-time cost small: 1 beginning-of-
season date, 1 mid-season date, and 5 in September (the current live
betting window), all confirmed as normal full slates against
raw_data/games/schedule/2025/ before picking them.

This is a one-off backtest script, not a daily Lambda -- no S3 production
path, no status_writer, no Terraform. Run from src/models/k_predictor/ with:
    python backtest/fetch_2025_odds.py
Requires AWS credentials with read access to SSM parameter
/mlb/odds-api/api-key (us-east-2).
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../data/modules"))

import boto3
import pandas as pd

from odds_fetch import get_all_historical_player_props, get_last_usage

DATES = [
    "2025-07-09",  # middle of season
    "2025-09-03",  # September (current betting window) x5
    "2025-09-09",
    "2025-09-15",
    "2025-09-21",
    "2025-09-27",
]

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_api_key():
    ssm = boto3.client("ssm", region_name="us-east-2")
    return ssm.get_parameter(Name="/mlb/odds-api/api-key", WithDecryption=True)["Parameter"]["Value"]


def flatten_props(date, game_props):
    """One row per (pitcher, over/under side) -- the flat shape the edge
    report (Epic 4) actually needs, rather than the nested API envelope."""
    rows = []
    for entry in game_props:
        game = entry["game"]
        props = entry["props"]
        for bookmaker in props.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "pitcher_strikeouts":
                    continue
                for outcome in market["outcomes"]:
                    rows.append({
                        "date": date,
                        "event_id": game["id"],
                        "away_team": game["away_team"],
                        "home_team": game["home_team"],
                        "commence_time": game["commence_time"],
                        "pitcher_name": outcome["description"],
                        "side": outcome["name"],
                        "line": outcome["point"],
                        "price": outcome["price"],
                        "bookmaker": bookmaker["key"],
                    })
    return pd.DataFrame(rows)


def main():
    api_key = get_api_key()
    total_used_start = None

    for date in DATES:
        print(f"\nFetching historical pitcher_strikeouts odds for {date}...")
        game_props = get_all_historical_player_props(date, api_key)
        df = flatten_props(date, game_props)

        out_path = OUT_DIR / f"player_props_{date}.parquet"
        df.to_parquet(out_path, index=False)

        used = get_last_usage()
        if total_used_start is None:
            total_used_start = used
        print(f"  {len(game_props)} games, {len(df)} prop rows -> {out_path}")
        print(f"  requests used this month so far: {used}")

    print(f"\nDone. Wrote {len(DATES)} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
