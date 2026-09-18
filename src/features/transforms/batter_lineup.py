import json

import boto3
import pandas as pd

from data_readers import read_batter_boxscore
from data.modules.preprocessing import create_batting_order

BUCKET = "mlbdk"

# Passthrough feature group — no aggregation. batting_order already exists
# verbatim in the raw source; the only job here is producing one consistent
# schema (personId, gamepk, event_timestamp, game_date, batting_order)
# whether the source is a realized box score (backfill) or an announced
# lineup (incremental, see daily_lineup_fetch). event_timestamp and
# game_date deliberately diverge for incremental (see parse_lineup_payload)
# — event_timestamp is the real Feast-facing freshness signal,
# materialize_incremental()'s watermark reads it directly, so it must be the
# real confirmation moment, not the calendar date, or a second same-day
# materialize call would see it as already-covered and skip it (see
# DECISIONS.md's 2026-09-17 entry). game_date is the calendar date,
# consumed by batter_pa_volume.py's point-in-time-safe rolling logic.
OUTPUT_COLUMNS = ["personId", "gamepk", "event_timestamp", "game_date", "batting_order"]


def _normalize_batter_boxscore_lineup(df: pd.DataFrame) -> pd.DataFrame:
    # Delegates the null-filter + dedup decision to the shared
    # create_batting_order (data/modules/preprocessing.py) — the same
    # function n_pa_predictor's own training pipeline calls
    # (processing/pipeline.py::build_batter_game_frame), moved out of
    # hit_predictor 2026-09-15 so this feature-store layer isn't depending
    # on one specific model's internals (see DECISIONS.md) — rather than
    # reimplementing it here. That function renames personId to batter_id
    # and drops game_date for its own (unrelated) purposes, so this wrapper
    # renames back and re-attaches game_date from a gamepk-only lookup, but
    # never re-decides which rows count. No real "confirmation moment"
    # concept exists for a realized box score, so event_timestamp is just
    # game_date here too.
    batting_order = create_batting_order(df).rename(columns={"batter_id": "personId"})
    game_dates = df[["gamepk", "game_date"]].drop_duplicates("gamepk")
    out = batting_order.merge(game_dates, on="gamepk", how="left")
    out["event_timestamp"] = out["game_date"]
    out = out[OUTPUT_COLUMNS].copy()
    out["batting_order"] = out["batting_order"].astype("int64")
    return out.reset_index(drop=True)


def compute_batter_lineup_features(year: str) -> pd.DataFrame:
    batter_boxscore = read_batter_boxscore(year)
    return _normalize_batter_boxscore_lineup(batter_boxscore)


def parse_lineup_payload(payload: dict) -> pd.DataFrame:
    """One row per player in a single game's announced lineup (daily_lineup_fetch's
    build_lineup_payload output) — the incremental-mode counterpart to
    _normalize_batter_boxscore_lineup, same output contract. event_timestamp
    comes from confirmed_at (real confirmation moment), not date — see
    OUTPUT_COLUMNS's comment above. tz-stripped to naive UTC so its dtype
    matches backfill's already-naive game_date-based timestamps; a mixed
    tz-aware/naive column breaks once both sources land in the same Parquet
    file."""
    event_timestamp = pd.Timestamp(payload["confirmed_at"]).tz_localize(None)
    game_date = pd.Timestamp(payload["date"])
    rows = [
        {
            "personId": str(player["player_id"]),
            "gamepk": str(payload["game_pk"]),
            "event_timestamp": event_timestamp,
            "game_date": game_date,
            "batting_order": player["batting_order"],
        }
        for side in ("home_lineup", "away_lineup")
        for player in payload[side]
    ]
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out["batting_order"] = out["batting_order"].astype("int64")
    return out


def compute_batter_lineup_features_for_date(date: str) -> pd.DataFrame:
    """Every announced lineup daily_lineup_fetch confirmed for `date`,
    combined into the same output contract as backfill mode."""
    year = date[:4]
    prefix = f"raw_data/games/lineups/{year}/{date}/"
    s3 = boto3.client("s3")
    keys = [
        obj["Key"]
        for obj in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])
    ]
    frames = []
    for key in keys:
        payload = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
        frames.append(parse_lineup_payload(payload))
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)
