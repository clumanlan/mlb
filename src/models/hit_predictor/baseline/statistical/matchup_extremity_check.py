"""
Matchup-extremity slice check for the statistical shrinkage baseline.
Run from models/hit_predictor/ with: python -m baseline.statistical.matchup_extremity_check
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Per BENCHMARKS.md §2's noise-at-n=1 caveat: instead of checking one
"dominant pitcher vs weak batter" game (still n=1, still noise-dominated),
this pools EVERY PA in the val season matching each (batter strength,
pitcher dominance) archetype and reports the aggregate observed hit rate
against the shrinkage baseline's mean predicted probability for that group.
See utils/matchup_slicing.py::slice_by_matchup_extremity for the pure logic.

Matchup axes, both already-existing features, no new ingestion:
  - batter strength: last_season_ba (same feature the shrinkage baseline
    itself uses as its shrink target)
  - pitcher dominance: pitcher_last_season_pa_hit_rate, from
    season_stats.build_pbp_pitcher_feats_all_roles (role-tagged: sp rows per
    individual pitcher, bullpen rows pooled by team). Joined onto pa_outcome
    via REALIZED pitcher_role/pitcher_id -- fine for this post-hoc diagnostic
    (unlike a production feature, there's no point-in-time leakage concern
    here since nothing is being predicted, only historical predictions
    already made are being sliced for review).
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
from models.hit_predictor.utils.matchup_slicing import slice_by_matchup_extremity

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── 1. Config ────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET          = cfg["bucket"]
REGION          = cfg["region"]
TRAIN_SEASONS   = cfg["train_seasons"]
FEATURE_SEASONS = cfg["feature_seasons"]
TARGET          = cfg["target_column"]
VAL_SEASON      = cfg["val_season"]
TEST_SEASON     = cfg["test_season"]

SHRINKAGE_K = 100.0
N_BINS = 3
MIN_N = 200

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))


# ── 2. Load data from S3 ─────────────────────────────────────────────────────
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


print("\nLoading play-by-play...")
pbp = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/playbyplay/{season}/", TRAIN_SEASONS, chunked=True,
)

print("\nLoading schedule...")
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/", all_boxscore_seasons,
)

print("\nLoading game info...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/", TRAIN_SEASONS,
)

print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", all_boxscore_seasons,
)

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)

game_info = pipeline.process_game_info(game_info)
schedule = pipeline.process_schedule(schedule)
pbp = pipeline.build_pbp_features(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

# ── 3. Batter-side + pitcher-side matchup features ────────────────────────────
batter_season_stats = season_stats.build_batter_stats(batter_boxscore).rename(
    columns={'personId': 'batter_id'}
)[['batter_id', 'game_season', 'batter_last_season_ba']].rename(
    columns={'batter_last_season_ba': 'last_season_ba'}
)
pa_outcome = pa_outcome.merge(batter_season_stats, on=['batter_id', 'game_season'], how='left')

pa_outcome = add_shrinkage_component(pa_outcome, k=SHRINKAGE_K)

print("\nBuilding pitcher role-tagged season stats...")
pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ['pitcher_key_id', 'game_season', 'pitcher_role', 'pitcher_last_season_pa_hit_rate']
]

# Realized pitcher key: sp PAs key on the individual pitcher_id (already
# equal to starting_pitcher_id for sp-role PAs); bullpen PAs key on
# pitcher_team_id, matching how the bullpen half of pitcher_role_season_stats
# is pooled. Using REALIZED role/id here (not the pre-game-estimated role
# used elsewhere in this repo) is fine -- see module docstring.
pa_outcome['realized_pitcher_key_id'] = np.where(
    pa_outcome['pitcher_role'] == 'sp', pa_outcome['pitcher_id'], pa_outcome['pitcher_team_id']
)
pa_outcome = pa_outcome.merge(
    pitcher_role_season_stats.rename(columns={'pitcher_key_id': 'realized_pitcher_key_id'}),
    on=['realized_pitcher_key_id', 'game_season', 'pitcher_role'],
    how='left',
)

# ── 4. Slice val season ────────────────────────────────────────────────────────
val_df = pa_outcome[pa_outcome['game_season'] == VAL_SEASON].copy()
val_df['pred_clipped'] = np.clip(val_df['shrinkage_pred'], 1e-6, 1 - 1e-6)

print(f"\nVal season {VAL_SEASON}: {len(val_df):,} PA rows before dropping missing matchup features")

result = slice_by_matchup_extremity(
    val_df, batter_col='last_season_ba', pitcher_col='pitcher_last_season_pa_hit_rate',
    outcome_col=TARGET, pred_col='pred_clipped', n_bins=N_BINS, min_n=MIN_N,
)
result = result.sort_values(['batter_bin', 'pitcher_bin']).reset_index(drop=True)

print("\n" + "#" * 75)
print("# MATCHUP-EXTREMITY SLICE — VAL SEASON, GAME-GRAIN QUESTION ANSWERED AT PA GRAIN")
print("# (pooled across many PAs per cell -- not a single-game read)")
print("#" * 75)
print(result.to_string(index=False))

n_dropped = len(val_df) - int(result['n'].sum())
print(f"\nDropped {n_dropped:,} rows with missing last_season_ba or pitcher_last_season_pa_hit_rate "
      f"(rookie batters / pitchers with no last-season pbp)")

# ── 5. Write results ─────────────────────────────────────────────────────────
OUT_DIR = BASE_DIR / "baseline" / "statistical"
result.to_json(OUT_DIR / "matchup_extremity_results.json", orient='records', indent=2)

md_lines = [
    "# Matchup-Extremity Slice Check — batter_hit_predictor",
    "",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
    f"**Question:** does the shrinkage baseline (and reality) actually predict near-zero "
    f"hit probability for a weak batter facing a dominant pitcher, once pooled across "
    f"every such PA in the val season (not read off one game)?  ",
    f"**Val season:** {VAL_SEASON} — {len(val_df):,} PA rows, {n_dropped:,} dropped "
    f"(missing last_season_ba or pitcher_last_season_pa_hit_rate)  ",
    "",
    "## Batter strength × pitcher dominance grid",
    "",
    "| Batter | Pitcher | N | Obs hit rate | Mean pred (shrinkage) | Reliable (n≥200) |",
    "|--------|---------|---|--------------|------------------------|-------------------|",
]
for _, row in result.iterrows():
    md_lines.append(
        f"| {row['batter_bin']} | {row['pitcher_bin']} | {row['n']:,} | "
        f"{row['obs_rate']:.3f} | {row['mean_pred']:.3f} | {row['reliable']} |"
    )

md_lines += [
    "",
    "## Setup",
    "",
    "- batter axis: last_season_ba (same feature the shrinkage baseline shrinks toward)",
    "- pitcher axis: pitcher_last_season_pa_hit_rate (role-tagged: sp per-individual, "
    "bullpen pooled by team), joined via REALIZED pitcher_role/pitcher_id -- a post-hoc "
    "diagnostic, not a production feature, so no point-in-time leakage concern",
    f"- n_bins = {N_BINS}, min_n = {MIN_N}",
]

with open(OUT_DIR / "matchup_extremity_results.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"\nSaved {OUT_DIR / 'matchup_extremity_results.md'}")
print(f"Saved {OUT_DIR / 'matchup_extremity_results.json'}")
print("\nDone.")
