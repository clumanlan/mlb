"""
One batter's plate appearances in a single game, showing how the new v14/v15
features (platoon_matchup, expected_times_through_order,
estimated_team_pa_position, pitcher_projected_pitches_before_pa) evolve PA by
PA against the same starter -- a concrete illustration of the "same_hand
matchup" and "in-game workload" ideas discussed this session, rather than
aggregate metrics.

Loads only 2023-2024 (2023 supplies "last season" stats and early-2024
trailing-3-game carryover) instead of the full multi-season pull the real
experiments use -- much faster, and sufficient for a single illustrative
game. Picks the 2024 batter-game with the most PAs against a single starter
(so times_through_order actually reaches its cap of 3, making the new
uncapped features' extra granularity visible) rather than a hand-picked
example, so it's not cherry-picked to make a point.

Run from src/models/k_predictor/ with: python experiments/v15_pa_workload/case_study.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
import awswrangler as wr
import boto3
import numpy as np
import pandas as pd

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import expected_role

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_projected_workload

BUCKET = "mlbdk"
REGION = "us-east-2"
SEASONS = [2023, 2024]
SHORT_PITCHER_WINDOW = 3

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)

boto_session = boto3.Session(region_name=REGION)


def read_parquet_seasons(path_tpl, seasons, chunked=False):
    frames = []
    for season in seasons:
        path = path_tpl.format(bucket=BUCKET, season=season)
        print(f"  {path}")
        if chunked:
            for chunk in wr.s3.read_parquet(path=path, chunked=True, boto3_session=boto_session):
                if "spin_direction" in chunk.columns:
                    chunk["spin_direction"] = chunk["spin_direction"].astype("float64")
                frames.append(chunk)
        else:
            frames.append(wr.s3.read_parquet(path=path, boto3_session=boto_session))
    return pd.concat(frames, ignore_index=True)


print("\nLoading play-by-play (2023-2024)...")
pbp = read_parquet_seasons("s3://{bucket}/processed_data/prepared/playbyplay/{season}/", SEASONS, chunked=True)
print("\nLoading schedule...")
schedule = read_parquet_seasons("s3://{bucket}/processed_data/games/schedule/{season}/", SEASONS)
print("\nLoading game info...")
game_info = read_parquet_seasons("s3://{bucket}/processed_data/games/game_info/{season}/", SEASONS)
print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", SEASONS)
print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", SEASONS)
print("\nLoading player info...")
player_info = wr.s3.read_parquet(path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session)

print("\nBuilding PA-grain DataFrame...")
schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome_strikeout(pbp, batter_boxscore, game_info, schedule)
pa_outcome["game_season"] = pa_outcome["game_date"].dt.year

pitcher_start_depth_stats = season_stats.build_pitcher_start_depth_stats(pbp)
league_avg_start_depth = season_stats.build_league_avg_start_depth(pitcher_start_depth_stats)
pa_outcome = expected_role.assign_expected_pitcher_role(pa_outcome, pitcher_start_depth_stats, league_avg_start_depth)

pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_pa_strikeout_rate"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})
pa_outcome = pa_outcome.merge(
    pitcher_role_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

pitcher_pbp_rolling3 = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window=SHORT_PITCHER_WINDOW)
pbp_rolling3_cols = [
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_total", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_strikeout_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_command_swinging_strike_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean",
]
pa_outcome = pa_outcome.merge(
    pitcher_pbp_rolling3.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + pbp_rolling3_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

# Pace must come from the STARTER'S OWN trailing-3g stat, not the
# expected_pitcher_key_id/expected_pitcher_role-keyed merge above -- that one
# is a pre-game estimate that falls back to team-pooled bullpen stats once a
# PA is projected past the starter's own expected depth. Every row here is
# already a REALIZED sp PA (create_pa_outcome_strikeout's own scoping), so
# join on starting_pitcher_id instead to keep this starter's own pace.
starter_pace_col = f"starting_pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean"
sp_only_pace = pitcher_pbp_rolling3[pitcher_pbp_rolling3["pitcher_role"] == "sp"][
    ["pitcher_key_id", "gamepk", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean"]
].rename(columns={
    "pitcher_key_id": "starting_pitcher_id",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean": starter_pace_col,
})
pa_outcome["starting_pitcher_id"] = pa_outcome["starting_pitcher_id"].astype(str)
pa_outcome["gamepk"] = pa_outcome["gamepk"].astype(str)
sp_only_pace["starting_pitcher_id"] = sp_only_pace["starting_pitcher_id"].astype(str)
sp_only_pace["gamepk"] = sp_only_pace["gamepk"].astype(str)
pa_outcome = pa_outcome.merge(sp_only_pace, on=["gamepk", "starting_pitcher_id"], how="left")

pa_outcome = build_pitcher_projected_workload(pa_outcome, pace_col=starter_pace_col)

pa_2024 = pa_outcome[pa_outcome["game_season"] == 2024].copy()

# ── Pick the batter-game with the most PAs against a single starter ─────────
counts = pa_2024.groupby(["gamepk", "batter_id"]).size().sort_values(ascending=False)
top_gamepk, top_batter_id = counts.index[0]
n_pa = counts.iloc[0]
print(f"\nMost PAs by one batter against a starter in 2024: {n_pa} "
      f"(gamepk={top_gamepk}, batter_id={top_batter_id})")

example = pa_2024[(pa_2024["gamepk"] == top_gamepk) & (pa_2024["batter_id"] == top_batter_id)].sort_values("batter_pa_number")

DISPLAY_COLS = [
    "batter_pa_number", "game_date", "pitcher_name", "batter_name", "is_strikeout",
    "pitcher_throw_hand", "batter_bat_side", "platoon_matchup",
    "expected_times_through_order", "estimated_team_pa_position",
    starter_pace_col,
    "pitcher_projected_pitches_before_pa",
]
DISPLAY_COLS = [c for c in DISPLAY_COLS if c in example.columns]

print(f"\n{'#' * 100}")
print(f"# {example['batter_name'].iloc[0]} vs {example['pitcher_name'].iloc[0]} -- {example['game_date'].iloc[0].date()} "
      f"(gamepk={top_gamepk})")
print(f"{'#' * 100}\n")
print(example[DISPLAY_COLS].to_string(index=False))

print("\nDone.")
