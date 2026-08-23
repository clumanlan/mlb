"""
Pitcher case study: one real, genuinely dominant 2024 starter's actual
game-by-game log, to make the matchup-extremity grid (matchup_extremity_check.py)
concrete instead of only aggregate. Identifies the pitcher FROM THE DATA
(lowest realized 2024 hit-rate-allowed among starters with >=15 starts) --
not a guessed name -- then shows the shrinkage baseline's (batter-only)
predicted probability against this pitcher's actual per-start results across
the season.

Run from models/hit_predictor/ with: python -m baseline.statistical.pitcher_case_study
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Minimal S3 pull vs the other baseline scripts: only 2024 (target season) +
2023 (feeds last_season_ba) are needed, not the full 7-season history.
"""
import json
import yaml
from datetime import datetime
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd

import models.hit_predictor.processing.pipeline as pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.baseline.statistical.shrinkage import add_shrinkage_component

BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET = cfg["bucket"]
REGION = cfg["region"]
TARGET = cfg["target_column"]
TARGET_SEASON = cfg["val_season"]  # 2024
FEATURE_SEASON = TARGET_SEASON - 1  # 2023

SHRINKAGE_K = 100.0
MIN_STARTS = 15
MIN_PA_FACED = 250

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


print("\nLoading play-by-play (2024 only)...")
pbp = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/playbyplay/{season}/", [TARGET_SEASON], chunked=True,
)

print("\nLoading schedule (2024)...")
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/", [TARGET_SEASON],
)

print("\nLoading game info (2024)...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/", [TARGET_SEASON],
)

print("\nLoading batter boxscore (2023-2024, 2023 feeds last_season_ba)...")
batter_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", [FEATURE_SEASON, TARGET_SEASON],
)

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)

game_info = pipeline.process_game_info(game_info)
schedule = pipeline.process_schedule(schedule)
pbp = pipeline.build_pbp_features(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

batter_season_stats = season_stats.build_batter_stats(batter_boxscore).rename(
    columns={'personId': 'batter_id'}
)[['batter_id', 'game_season', 'batter_last_season_ba']].rename(
    columns={'batter_last_season_ba': 'last_season_ba'}
)
pa_outcome = pa_outcome.merge(batter_season_stats, on=['batter_id', 'game_season'], how='left')
pa_outcome = add_shrinkage_component(pa_outcome, k=SHRINKAGE_K)
pa_outcome['pred_clipped'] = np.clip(pa_outcome['shrinkage_pred'], 1e-6, 1 - 1e-6)

# ── Identify a genuinely dominant 2024 starter FROM THE DATA ─────────────────
sp_pa = pa_outcome[pa_outcome['pitcher_role'] == 'sp'].copy()
pitcher_totals = (
    sp_pa.groupby(['pitcher_id', 'pitcher_name'])
    .agg(n_starts=('gamepk', 'nunique'), n_pa=(TARGET, 'size'), hits=(TARGET, 'sum'))
    .reset_index()
)
pitcher_totals['hit_rate_allowed'] = pitcher_totals['hits'] / pitcher_totals['n_pa']
qualified = pitcher_totals[
    (pitcher_totals['n_starts'] >= MIN_STARTS) & (pitcher_totals['n_pa'] >= MIN_PA_FACED)
].sort_values('hit_rate_allowed')

print(f"\n{len(qualified)} pitchers qualify (>= {MIN_STARTS} starts, >= {MIN_PA_FACED} PA faced) in {TARGET_SEASON}")
print("\nTop 5 most dominant by realized 2024 hit rate allowed:")
print(qualified.head(5)[['pitcher_name', 'n_starts', 'n_pa', 'hit_rate_allowed']].to_string(index=False))

chosen = qualified.iloc[0]
pitcher_id = chosen['pitcher_id']
pitcher_name = chosen['pitcher_name']
print(f"\nChosen: {pitcher_name} (pitcher_id={pitcher_id}) — "
      f"{chosen['n_starts']:.0f} starts, {chosen['n_pa']:.0f} PA faced, "
      f"{chosen['hit_rate_allowed']:.3f} hit rate allowed")

league_avg_hit_rate = pa_outcome[TARGET].mean()

# ── Per-game log for the chosen pitcher ───────────────────────────────────────
pitcher_pa = sp_pa[sp_pa['pitcher_id'] == pitcher_id].copy()
pitcher_pa['game_date'] = pd.to_datetime(pitcher_pa['game_date'])

per_game = (
    pitcher_pa.groupby(['gamepk', 'game_date'])
    .agg(
        n_pa=(TARGET, 'size'),
        hits=(TARGET, 'sum'),
        mean_pred=('pred_clipped', 'mean'),
        opponent=('batter_team_name', lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
    )
    .reset_index()
    .sort_values('game_date')
    .reset_index(drop=True)
)
per_game['hit_rate'] = per_game['hits'] / per_game['n_pa']
per_game['cum_hit_rate'] = per_game['hits'].cumsum() / per_game['n_pa'].cumsum()
per_game['start_number'] = range(1, len(per_game) + 1)

print(f"\n{pitcher_name} — {len(per_game)} starts logged in {TARGET_SEASON}")
print(per_game[['start_number', 'game_date', 'opponent', 'n_pa', 'hit_rate', 'mean_pred', 'cum_hit_rate']].to_string(index=False))

# ── Write results ─────────────────────────────────────────────────────────────
OUT_DIR = BASE_DIR / "baseline" / "statistical"
per_game_out = per_game.copy()
per_game_out['game_date'] = per_game_out['game_date'].dt.strftime('%Y-%m-%d')
result = {
    "pitcher_id": str(pitcher_id),
    "pitcher_name": pitcher_name,
    "season": TARGET_SEASON,
    "n_starts": int(chosen['n_starts']),
    "n_pa_faced": int(chosen['n_pa']),
    "season_hit_rate_allowed": float(chosen['hit_rate_allowed']),
    "league_avg_hit_rate": float(league_avg_hit_rate),
    "qualified_field_size": int(len(qualified)),
    "per_game": per_game_out.to_dict(orient='records'),
}
with open(OUT_DIR / "pitcher_case_study.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nSaved {OUT_DIR / 'pitcher_case_study.json'}")
print("\nDone.")
