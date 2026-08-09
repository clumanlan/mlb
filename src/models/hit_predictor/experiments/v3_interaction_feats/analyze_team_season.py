"""Pull one team's season stats two ways and put them side by side for a sanity check.

Loads the same S3 tables and runs the same processing steps as train.py
(process_schedule, process_pitcher_boxscore, build_pbp_features, season_stats,
rolling_stats) but scoped down to a single team + season instead of the full
multi-season training set. No model is trained here — this is a validation
script, not an experiment variant.

Two comparisons come out of it:

  1. RAW vs PIPELINE — a season total computed directly from box score sums
     (independent of season_stats.py, so it can't share a bug with it) next to
     the same total computed by season_stats.build_batter_stats /
     build_pitcher_stats. These should match almost exactly (pipeline values
     are rounded to 2 decimals, raw are not) — anything outside that is worth
     investigating in season_stats.py itself.
  2. RAW vs online — the RAW tables use the same box-score counting stats
     (H, HR, RBI, SB, AVG, ERA, ...) shown on a site like baseball-reference.com's
     team season page, so they can be eyeballed against that directly.

Edit TEAM_NAME / TEAM_ID / SEASON below and run the same way train.py is run.
"""

from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import yaml

import models.hit_predictor.processing.pipeline as pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET = cfg["bucket"]
REGION = cfg["region"]

# ── Edit these to check a different team/season ────────────────────────────
TEAM_NAME = "Cleveland Guardians"
TEAM_ID   = "114"
SEASON    = 2024

# Tolerance for RAW-vs-PIPELINE rate-stat comparisons — pipeline values are
# rounded to 2 decimals in season_stats.py, RAW values here are not. iso is
# slg - avg computed from two independently-rounded pipeline values, so its
# rounding error can compound to ~2x a single stat's — hence the wider bound.
RATE_TOLERANCE = 0.011

OUT_DIR = BASE_DIR / "plots" / "v3_interaction_feats" / "team_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

boto_session = boto3.Session(region_name=REGION)


def read_parquet(path_tpl, chunked=False):
    path = path_tpl.format(bucket=BUCKET, season=SEASON)
    print(f"  {path}")
    if not chunked:
        return wr.s3.read_parquet(path=path, boto3_session=boto_session)
    frames = list(wr.s3.read_parquet(path=path, chunked=True, boto3_session=boto_session))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------- #
#                                 READ + FILTER                                #
# ---------------------------------------------------------------------------- #
print(f"\nLoading {TEAM_NAME} ({SEASON})...")

print("schedule...")
schedule = pipeline.process_schedule(read_parquet("s3://{bucket}/processed_data/games/schedule/{season}/"))
schedule = schedule[
    ((schedule["home_id"] == TEAM_ID) | (schedule["away_id"] == TEAM_ID))
    & (schedule["game_type"] == "R")
].copy()
guardians_gamepks = set(schedule["gamepk"])
print(f"  {len(schedule)} regular-season games")

print("batter boxscore...")
batter_boxscore = read_parquet("s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/")
batter_boxscore = batter_boxscore[
    (batter_boxscore["team_id"] == TEAM_ID) & (batter_boxscore["gamepk"].isin(guardians_gamepks))
].copy()
print(f"  {len(batter_boxscore)} batter-game rows, {batter_boxscore['personId'].nunique()} players")

print("pitcher boxscore...")
pitcher_boxscore = read_parquet("s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/")
pitcher_boxscore = pitcher_boxscore[
    (pitcher_boxscore["team_id"] == TEAM_ID) & (pitcher_boxscore["gamepk"].isin(guardians_gamepks))
].copy()
pitcher_boxscore = pipeline.process_pitcher_boxscore(pitcher_boxscore)
print(f"  {len(pitcher_boxscore)} pitcher-game rows, {pitcher_boxscore['personId'].nunique()} players")

print("play-by-play (filtered to this team's games)...")
pbp = read_parquet("s3://{bucket}/processed_data/prepared/playbyplay/{season}/", chunked=True)
pbp = pbp[pbp["gamepk"].isin(guardians_gamepks)].copy()
print(f"  {len(pbp)} pitch rows")

print("player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)

pbp = pipeline.build_pbp_features(pbp, schedule, player_info)


# ---------------------------------------------------------------------------- #
#                            BATTER: RAW vs PIPELINE                           #
# ---------------------------------------------------------------------------- #
BATTER_SUM_COLS = ['ab', 'h', 'r', 'doubles', 'triples', 'hr', 'rbi', 'sb', 'bb', 'k',
                    'plate_appearances', 'total_bases_from_h']

# RAW — plain box-score sums, computed independently of season_stats.py.
raw_batter = (
    batter_boxscore
    .groupby(['personId', 'player_name'])
    .agg(**{c: (c, 'sum') for c in BATTER_SUM_COLS}, games=('gamepk', 'nunique'))
    .reset_index()
    .assign(
        avg=lambda x: x['h'] / x['ab'].replace(0, np.nan),
        slg=lambda x: x['total_bases_from_h'] / x['ab'].replace(0, np.nan),
        # approximate — HBP/SF aren't in the box score schema, so this is (H+BB)/(AB+BB)
        # rather than the true OBP denominator. Fine for a sanity check, not exact.
        obp_approx=lambda x: (x['h'] + x['bb']) / (x['ab'] + x['bb']).replace(0, np.nan),
    )
    .assign(
        iso=lambda x: x['slg'] - x['avg'],
        babip=lambda x: (x['h'] - x['hr']) / (x['ab'] - x['k'] - x['hr']).replace(0, np.nan),
    )
    .sort_values('plate_appearances', ascending=False)
    .reset_index(drop=True)
)

# PIPELINE — same season_stats.build_batter_stats() call train.py uses. It shifts
# game_season forward one year (built for joining onto NEXT season as a lookup
# feature), so this season's true totals land on game_season == SEASON + 1.
pipeline_batter = season_stats.build_batter_stats(batter_boxscore)
pipeline_batter = pipeline_batter[pipeline_batter['game_season'] == SEASON + 1].copy()
pipeline_batter = pipeline_batter.rename(columns=lambda c: c.replace('batter_last_season_', 'pipeline_'))
pipeline_batter = pipeline_batter.drop(columns=['game_season'])

batter_compare = raw_batter.merge(pipeline_batter, on='personId', how='left')
for stat in ['ba', 'slg', 'iso', 'babip']:
    raw_col = {'ba': 'avg'}.get(stat, stat)
    batter_compare[f'diff_{stat}'] = batter_compare[raw_col] - batter_compare[f'pipeline_{stat}']
mismatch_cols = [f'diff_{s}' for s in ['ba', 'slg', 'iso', 'babip']]
batter_compare['match'] = (batter_compare[mismatch_cols].abs().max(axis=1) < RATE_TOLERANCE) | (
    batter_compare[mismatch_cols].isna().all(axis=1)
)

print(f"\n=== {TEAM_NAME} {SEASON} — BATTERS (raw box score vs season_stats.build_batter_stats) ===")
print(batter_compare[[
    'player_name', 'games', 'plate_appearances', 'ab', 'h', 'hr', 'r', 'rbi', 'sb', 'bb', 'k',
    'avg', 'pipeline_ba', 'slg', 'pipeline_slg', 'obp_approx', 'match',
]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

n_batter_mismatch = (~batter_compare['match']).sum()
if n_batter_mismatch:
    print(f"\n!! {n_batter_mismatch} batter(s) outside {RATE_TOLERANCE} tolerance — check season_stats.py")

batter_compare.to_csv(OUT_DIR / f"{SEASON}_batters.csv", index=False)


# ---------------------------------------------------------------------------- #
#                           PITCHER: RAW vs PIPELINE                           #
# ---------------------------------------------------------------------------- #
PITCHER_SUM_COLS = ['h', 'r', 'er', 'bb', 'hr', 'k', 'p', 's', 'ip']

# RAW — includes ERA, which season_stats.py doesn't compute at all, specifically
# because it's the stat most people will recognize on a stats site to cross-check.
raw_pitcher = (
    pitcher_boxscore
    .groupby(['personId', 'player_name'])
    .agg(**{c: (c, 'sum') for c in PITCHER_SUM_COLS}, games=('gamepk', 'nunique'))
    .reset_index()
    .assign(
        whip=lambda x: (x['bb'] + x['h']) / x['ip'].replace(0, np.nan),
        k_rate=lambda x: x['k'] / x['ip'].replace(0, np.nan),
        bb_rate=lambda x: x['bb'] / x['ip'].replace(0, np.nan),
        era=lambda x: x['er'] * 9 / x['ip'].replace(0, np.nan),
    )
    .sort_values('ip', ascending=False)
    .reset_index(drop=True)
)

# PIPELINE — season_stats.build_pitcher_stats(), the role-agnostic version (same
# per-pitcher total as build_pitcher_stats_all_roles' 'sp' rows would give if this
# pitcher only ever started, but summed across ALL his appearances regardless of
# role — the fair comparison against RAW's every-appearance sum above).
pipeline_pitcher = season_stats.build_pitcher_stats(pitcher_boxscore, entity_col='personId')
pipeline_pitcher = pipeline_pitcher[pipeline_pitcher['game_season'] == SEASON + 1].copy()
pipeline_pitcher = pipeline_pitcher.rename(columns=lambda c: c.replace('pitcher_last_season_', 'pipeline_'))
pipeline_pitcher = pipeline_pitcher.drop(columns=['game_season'])

pitcher_compare = raw_pitcher.merge(pipeline_pitcher, on='personId', how='left')
for stat in ['whip', 'k_rate', 'bb_rate']:
    pitcher_compare[f'diff_{stat}'] = pitcher_compare[stat] - pitcher_compare[f'pipeline_{stat}']
mismatch_cols = [f'diff_{s}' for s in ['whip', 'k_rate', 'bb_rate']]
pitcher_compare['match'] = (pitcher_compare[mismatch_cols].abs().max(axis=1) < RATE_TOLERANCE) | (
    pitcher_compare[mismatch_cols].isna().all(axis=1)
)

print(f"\n=== {TEAM_NAME} {SEASON} — PITCHERS (raw box score vs season_stats.build_pitcher_stats) ===")
print(pitcher_compare[[
    'player_name', 'games', 'ip', 'h', 'r', 'er', 'bb', 'k', 'hr',
    'era', 'whip', 'pipeline_whip', 'k_rate', 'pipeline_k_rate', 'match',
]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

n_pitcher_mismatch = (~pitcher_compare['match']).sum()
if n_pitcher_mismatch:
    print(f"\n!! {n_pitcher_mismatch} pitcher(s) outside {RATE_TOLERANCE} tolerance — check season_stats.py")

pitcher_compare.to_csv(OUT_DIR / f"{SEASON}_pitchers.csv", index=False)


# ---------------------------------------------------------------------------- #
#              INFORMATIONAL: role-split + rolling-season snapshot             #
# ---------------------------------------------------------------------------- #
# What train.py actually joins into model_df: sp rows per individual pitcher,
# bullpen rows pooled by team (pitcher_key_id == TEAM_ID for those rows — not a
# real player id). Shown for context, not compared against RAW above.
role_split = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)
role_split = role_split[role_split['game_season'] == SEASON + 1].copy()
print(f"\n=== {TEAM_NAME} {SEASON} — pipeline role-split pitcher stats (sp vs pooled bullpen) ===")
print(role_split.rename(columns=lambda c: c.replace('pitcher_last_season_', ''))
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# Rolling 'season' window excludes each row's own game (point-in-time safety —
# see rolling_stats.py), so a player's LAST game of the season reflects his
# stats through the second-to-last game, not the full season. Close to, but
# not exactly, the RAW batter table above — that gap is expected, not a bug.
batter_rolling = (
    rolling_stats.build_batter_rolling_stats(batter_boxscore, window='season')
    .sort_values(['personId', 'game_date'])
    .groupby('personId')
    .tail(1)
)
batter_rolling = batter_rolling.merge(
    batter_boxscore[['personId', 'player_name']].drop_duplicates(), on='personId', how='left'
)
print(f"\n=== {TEAM_NAME} {SEASON} — rolling 'season' stats, last game of season "
      "(excludes that game itself — expect slightly below full-season RAW totals) ===")
print(batter_rolling[[
    'player_name', 'game_date', 'batter_roll_season_ba', 'batter_roll_season_slg',
]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


print(f"\nSaved comparison CSVs to {OUT_DIR}")
print("Next: spot-check the top few players by PA/IP above against "
      f"baseball-reference.com's {SEASON} {TEAM_NAME} team stats pages.")
