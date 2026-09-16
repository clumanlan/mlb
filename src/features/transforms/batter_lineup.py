import json

import boto3
import pandas as pd

from data_readers import read_batter_boxscore
from models.hit_predictor.processing.pipeline import _create_batting_order

BUCKET = "mlbdk"

# Passthrough feature group — no aggregation. batting_order already exists
# verbatim in the raw source; the only job here is producing one consistent
# schema (personId, gamepk, event_timestamp, batting_order) whether the
# source is a realized box score (backfill) or an announced lineup
# (incremental, see daily_lineup_fetch).
OUTPUT_COLUMNS = ["personId", "gamepk", "event_timestamp", "batting_order"]


def _normalize_batter_boxscore_lineup(df: pd.DataFrame) -> pd.DataFrame:
    # Delegates the null-filter + dedup decision to hit_predictor's
    # _create_batting_order — the same function n_pa_predictor's own
    # training pipeline calls (processing/pipeline.py::build_batter_game_frame)
    # — rather than reimplementing it here. That function renames personId to
    # batter_id and drops game_date for its own (unrelated) purposes, so this
    # wrapper renames back and re-attaches event_timestamp from a
    # gamepk-only lookup, but never re-decides which rows count.
    batting_order = _create_batting_order(df).rename(columns={"batter_id": "personId"})
    game_dates = (
        df[["gamepk", "game_date"]]
        .drop_duplicates("gamepk")
        .rename(columns={"game_date": "event_timestamp"})
    )
    out = batting_order.merge(game_dates, on="gamepk", how="left")[OUTPUT_COLUMNS].copy()
    out["batting_order"] = out["batting_order"].astype("int64")
    return out.reset_index(drop=True)


def compute_batter_lineup_features(year: str) -> pd.DataFrame:
    batter_boxscore = read_batter_boxscore(year)
    return _normalize_batter_boxscore_lineup(batter_boxscore)


def parse_lineup_payload(payload: dict) -> pd.DataFrame:
    """One row per player in a single game's announced lineup (daily_lineup_fetch's
    build_lineup_payload output) — the incremental-mode counterpart to
    _normalize_batter_boxscore_lineup, same output contract."""
    event_timestamp = pd.Timestamp(payload["date"])
    rows = [
        {
            "personId": str(player["player_id"]),
            "gamepk": str(payload["game_pk"]),
            "event_timestamp": event_timestamp,
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
