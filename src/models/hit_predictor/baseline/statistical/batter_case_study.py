"""
Batter case study, mirroring pitcher_case_study.py from the other side: one
real, genuinely bad 2024 batter and one real, genuinely good one, identified
FROM THE DATA (realized 2024 PA-based hit rate, not batting average --
walks/HBP are in this pipeline's denominator throughout, same convention as
everywhere else in this analysis), among batters with >=400 PA.

For each, builds a per-game log (predicted vs actual, same shape as the
Snell chart) plus a PA-level worked table for a selected mid-season stretch
showing the matchup-blended formula against each different pitcher faced.

Run from models/hit_predictor/ with: python -m baseline.statistical.batter_case_study
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
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
from models.hit_predictor.baseline.statistical.shrinkage import (
    add_matchup_shrinkage_component_blended,
    add_shrinkage_component,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET = cfg["bucket"]
REGION = cfg["region"]
TARGET = cfg["target_column"]
TARGET_SEASON = cfg["val_season"]  # 2024
FEATURE_SEASON = TARGET_SEASON - 1  # 2023
SHRINKAGE_K = 100.0
MIN_PA_QUALIFIED = 400
STRETCH_N_PA = 18  # length of the mid-season worked-table stretch, per batter

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
pbp = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/playbyplay/{season}/", [FEATURE_SEASON, TARGET_SEASON], chunked=True,
)
print("\nLoading schedule (2023-2024)...")
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/", [FEATURE_SEASON, TARGET_SEASON],
)
print("\nLoading game info (2024)...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/", [TARGET_SEASON],
)
print("\nLoading batter boxscore (2023-2024)...")
batter_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", [FEATURE_SEASON, TARGET_SEASON],
)
print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)

game_info = pipeline.process_game_info(game_info)
schedule = pipeline.process_schedule(schedule)
pbp_full = pipeline.build_pbp_features(pbp, schedule, player_info)
pbp_2024 = pbp_full[pbp_full["game_date"].astype(str).str.startswith(str(TARGET_SEASON))]

pa_outcome = pipeline.create_pa_outcome(pbp_2024, batter_boxscore, game_info, schedule)

batter_season_stats = season_stats.build_batter_stats(batter_boxscore).rename(
    columns={'personId': 'batter_id'}
)[['batter_id', 'game_season', 'batter_last_season_ba']].rename(
    columns={'batter_last_season_ba': 'last_season_ba'}
)
pa_outcome = pa_outcome.merge(batter_season_stats, on=['batter_id', 'game_season'], how='left')

print("\nBuilding pitcher role-tagged season stats...")
pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp_full)[
    ['pitcher_key_id', 'game_season', 'pitcher_role', 'pitcher_last_season_pa_hit_rate']
]
pa_outcome['realized_pitcher_key_id'] = np.where(
    pa_outcome['pitcher_role'] == 'sp', pa_outcome['pitcher_id'], pa_outcome['pitcher_team_id']
)
pa_outcome = pa_outcome.merge(
    pitcher_role_season_stats.rename(columns={'pitcher_key_id': 'realized_pitcher_key_id'}),
    on=['realized_pitcher_key_id', 'game_season', 'pitcher_role'],
    how='left',
)

pa_outcome = add_shrinkage_component(pa_outcome, k=SHRINKAGE_K)
pa_outcome = add_matchup_shrinkage_component_blended(pa_outcome, k=SHRINKAGE_K)

val_df = pa_outcome.dropna(subset=['shrinkage_pred', 'matchup_shrinkage_blended_pred']).copy()
val_df['game_date'] = pd.to_datetime(val_df['game_date'])
val_df['pred_clipped'] = np.clip(val_df['matchup_shrinkage_blended_pred'], 1e-6, 1 - 1e-6)

league_avg_2024 = val_df[TARGET].mean()

# ── Identify a genuinely bad and genuinely good 2024 batter FROM THE DATA ────
batter_totals = (
    val_df.groupby(['batter_id', 'batter_name'])
    .agg(n_pa=(TARGET, 'size'), hits=(TARGET, 'sum'))
    .reset_index()
)
batter_totals['hit_rate'] = batter_totals['hits'] / batter_totals['n_pa']
qualified = batter_totals[batter_totals['n_pa'] >= MIN_PA_QUALIFIED].sort_values('hit_rate')

print(f"\n{len(qualified)} batters qualify (>= {MIN_PA_QUALIFIED} PA) in {TARGET_SEASON}")
print("\nWorst 3 by realized 2024 PA-based hit rate:")
print(qualified.head(3)[['batter_name', 'n_pa', 'hit_rate']].to_string(index=False))
print("\nBest 3 by realized 2024 PA-based hit rate:")
print(qualified.tail(3)[['batter_name', 'n_pa', 'hit_rate']].to_string(index=False))

bad_batter = qualified.iloc[0]
good_batter = qualified.iloc[-1]


def build_case_study(batter_row, label):
    batter_id = batter_row['batter_id']
    batter_name = batter_row['batter_name']
    bpa = val_df[val_df['batter_id'] == batter_id].copy().sort_values(['game_date', 'gamepk', 'play_id'])

    per_game = (
        bpa.groupby(['gamepk', 'game_date'])
        .agg(
            n_pa=(TARGET, 'size'), hits=(TARGET, 'sum'),
            mean_pred=('pred_clipped', 'mean'),
            opponent=('pitcher_name', lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        )
        .reset_index().sort_values('game_date').reset_index(drop=True)
    )
    per_game['hit_rate'] = per_game['hits'] / per_game['n_pa']
    per_game['cum_hit_rate'] = per_game['hits'].cumsum() / per_game['n_pa'].cumsum()
    per_game['game_number'] = range(1, len(per_game) + 1)

    print(f"\n{label}: {batter_name} — {len(per_game)} games, "
          f"{int(batter_row['n_pa'])} PA, {batter_row['hit_rate']:.3f} season hit rate")
    print(per_game[['game_number', 'game_date', 'opponent', 'n_pa', 'hit_rate', 'mean_pred', 'cum_hit_rate']].to_string(index=False))

    # Mid-season stretch of individual PAs for the worked table.
    mid_start = len(bpa) // 2 - STRETCH_N_PA // 2
    stretch = bpa.iloc[mid_start: mid_start + STRETCH_N_PA]

    worked = []
    for _, row in stretch.iterrows():
        b = row['last_season_ba']
        p = row['pitcher_last_season_pa_hit_rate']
        L = league_avg_2024
        worked.append({
            "game_date": row['game_date'].strftime('%Y-%m-%d'),
            "pitcher_name": row['pitcher_name'],
            "pitcher_role": row['pitcher_role'],
            "batter_last_season_ba": None if pd.isna(b) else round(float(b), 3),
            "pitcher_last_season_hit_rate": None if pd.isna(p) else round(float(p), 3),
            "league_avg_2024": round(float(L), 3),
            "batter_only_pred": round(float(row['shrinkage_pred']), 4),
            "matchup_blended_pred": round(float(row['matchup_shrinkage_blended_pred']), 4),
            "actual_is_hit": int(row[TARGET]),
        })

    per_game_out = per_game.copy()
    per_game_out['game_date'] = per_game_out['game_date'].dt.strftime('%Y-%m-%d')
    return {
        "label": label,
        "batter_id": str(batter_id),
        "batter_name": batter_name,
        "season": TARGET_SEASON,
        "n_pa": int(batter_row['n_pa']),
        "season_hit_rate": float(batter_row['hit_rate']),
        "league_avg_hit_rate": float(league_avg_2024),
        "qualified_field_size": int(len(qualified)),
        "per_game": per_game_out.to_dict(orient='records'),
        "worked_examples": worked,
    }


bad_result = build_case_study(bad_batter, "Worst qualified batter")
good_result = build_case_study(good_batter, "Best qualified batter")

OUT_DIR = BASE_DIR / "baseline" / "statistical"
with open(OUT_DIR / "batter_case_study.json", "w") as f:
    json.dump({"bad_batter": bad_result, "good_batter": good_result}, f, indent=2, default=str)
print(f"\nSaved {OUT_DIR / 'batter_case_study.json'}")
print("\nDone.")
