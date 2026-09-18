import numpy as np
import pandas as pd

from data_readers import read_batter_boxscore, read_playbyplay, read_schedule
from batter_lineup import compute_batter_lineup_features_for_date

from models.n_pa_predictor.processing.pipeline import build_batter_game_frame
from models.n_pa_predictor.processing.features.batter_playing_time import build_batter_pa_rolling_stats

# Wraps the already-tested build_batter_game_frame + build_batter_pa_rolling_stats
# rather than reimplementing the point-in-time-safe rolling logic — same
# delegate-don't-redefine rule batter_lineup.py follows via
# create_batting_order (data/modules/preprocessing.py). Neither wrapped
# function is modified here; this module only normalizes their output into
# the store's schema (personId, event_timestamp — not batter_id, game_date)
# and names the columns per this repo's established
# {entity}_roll_{window}_{stat} convention.
OUTPUT_COLUMNS = [
    "personId", "gamepk", "event_timestamp",
    "batter_pa_roll_season_games_n", "batter_pa_roll_season_avg_n_pa_per_game",
]

# build_batter_game_frame's build_n_pa_label only ever reads these 4 columns
# out of the prepared playbyplay table's 70 (mostly wide Statcast float
# columns) — reading the full table for a season inside a Lambda OOM'd
# twice in a row (2026-09-18, 512MB then 2048MB, maxed out both times).
PBP_COLUMNS_NEEDED = ["gamepk", "play_id", "batter_id", "play_result"]


def _normalize_batter_pa_rolling(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={
        "batter_id": "personId",
        "game_date": "event_timestamp",
        "batter_n_pa_roll_season_games_n": "batter_pa_roll_season_games_n",
        "batter_n_pa_roll_season_avg_n_pa_per_game": "batter_pa_roll_season_avg_n_pa_per_game",
    })[OUTPUT_COLUMNS].copy()
    out["batter_pa_roll_season_games_n"] = out["batter_pa_roll_season_games_n"].astype("int64")
    return out


def compute_batter_pa_volume_features(year: str) -> pd.DataFrame:
    pbp = read_playbyplay(year, columns=PBP_COLUMNS_NEEDED)
    batter_boxscore = read_batter_boxscore(year)
    schedule = read_schedule(year)

    batter_game = build_batter_game_frame(pbp, batter_boxscore, schedule)
    batter_game["game_season"] = batter_game["game_date"].dt.year

    rolled = build_batter_pa_rolling_stats(batter_game, window="season")
    return _normalize_batter_pa_rolling(rolled)


def _append_phantom_rows(batter_game: pd.DataFrame, today_players: pd.DataFrame) -> pd.DataFrame:
    """One placeholder row per today_players entry (n_pa unknown/irrelevant
    — build_batter_pa_rolling_stats's shift(1) never reads a row's own
    n_pa, only prior rows'), appended to real history so the SAME rolling
    function can be reused unmodified for the "as of right now" snapshot.
    today_players is compute_batter_lineup_features_for_date's own output shape —
    reused directly, not a separate lookup."""
    # np.nan, not pd.NA — pd.NA forces the whole n_pa column to object
    # dtype once concatenated with real int rows, which then propagates
    # through the rolling computation and breaks dtype parity with
    # backfill (caught by test_backfill_and_incremental_produce_identical_schema).
    phantoms = pd.DataFrame({
        "batter_id": today_players["personId"],
        "gamepk": today_players["gamepk"],
        "game_date": today_players["game_date"],
        "game_season": today_players["game_date"].dt.year,
        "n_pa": np.nan,
    })
    return pd.concat([batter_game, phantoms], ignore_index=True)


def compute_batter_pa_volume_features_for_date(date: str) -> pd.DataFrame:
    today_players = compute_batter_lineup_features_for_date(date)
    if today_players.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    year = date[:4]
    pbp = read_playbyplay(year, columns=PBP_COLUMNS_NEEDED)
    batter_boxscore = read_batter_boxscore(year)
    schedule = read_schedule(year)

    batter_game = build_batter_game_frame(pbp, batter_boxscore, schedule)
    batter_game["game_season"] = batter_game["game_date"].dt.year

    with_phantoms = _append_phantom_rows(batter_game, today_players)
    rolled = build_batter_pa_rolling_stats(with_phantoms, window="season")

    today_ts = pd.Timestamp(date)
    todays_rows = rolled[rolled["game_date"] == today_ts]
    out = _normalize_batter_pa_rolling(todays_rows)

    # Fold in the real confirmation moment from today_players — _normalize's
    # game_date->event_timestamp rename is correct for backfill (no real
    # "confirmed_at" concept) but wrong here: materialize_incremental()'s
    # watermark reads event_timestamp, so incremental output must carry the
    # real confirmation time, not midnight (see OUTPUT_COLUMNS comment in
    # batter_lineup.py and DECISIONS.md's 2026-09-17 entry).
    confirmed_at = today_players[["personId", "gamepk", "event_timestamp"]]
    out = out.drop(columns=["event_timestamp"]).merge(confirmed_at, on=["personId", "gamepk"], how="left")
    return out[OUTPUT_COLUMNS]
