"""
Matchup (log5 pitcher-augmented) shrinkage baseline vs the current
batter-only shrinkage baseline. Verifies whether adding pitcher signal is a
`real_improvement` per utils/eval.py::summarize_verdict(), not just "ROC-AUC
moved" -- see BENCHMARKS.md §2. Also dumps Blake Snell's real 2024 worked
examples (same pitcher identified in pitcher_case_study.py) for a lavish
artifact walkthrough of the formula against several real batters.

Run from models/hit_predictor/ with: python -m baseline.statistical.run_matchup_baseline
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Minimal S3 pull (2023-2024 only), same rationale as pitcher_case_study.py:
last_season_ba and pitcher_last_season_pa_hit_rate only need one prior
season (2023) to compute for the 2024 val season -- no regression, no
fitting, both predictions are closed-form given already-known rates.
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
    add_matchup_shrinkage_component,
    add_matchup_shrinkage_component_blended,
    add_shrinkage_component,
)
from models.hit_predictor.utils.eval import run_pa_vs_game_grain_check, summarize_verdict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET = cfg["bucket"]
REGION = cfg["region"]
TARGET = cfg["target_column"]
TARGET_SEASON = cfg["val_season"]  # 2024
FEATURE_SEASON = TARGET_SEASON - 1  # 2023
SHRINKAGE_K = 100.0

# Same pitcher identified in pitcher_case_study.py (mechanically, from data —
# lowest 2024 hit rate allowed among qualified starters): Blake Snell.
CASE_STUDY_PITCHER_ID = "605483"

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

print("\nLoading schedule (2023-2024 -- 2023 needed so build_pbp_features can "
      "date-stamp 2023 pbp rows correctly before the pitcher season-stats shift)...")
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
# create_pa_outcome needs only the target season's own PA rows (schedule/
# game_info are 2024-only anyway); pbp_full still carries 2023 for the
# pitcher season-stats shift below.
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

# ── Three predictions, same frame ─────────────────────────────────────────────
pa_outcome = add_shrinkage_component(pa_outcome, k=SHRINKAGE_K)
pa_outcome = add_matchup_shrinkage_component(pa_outcome, k=SHRINKAGE_K)
pa_outcome = add_matchup_shrinkage_component_blended(pa_outcome, k=SHRINKAGE_K)

PRED_COLS = ['shrinkage_pred', 'matchup_shrinkage_pred', 'matchup_shrinkage_blended_pred']
val_df = pa_outcome.dropna(subset=PRED_COLS).copy()
y_val = val_df[TARGET]
batter_only_proba = np.clip(val_df['shrinkage_pred'].to_numpy(), 1e-6, 1 - 1e-6)
matchup_proba = np.clip(val_df['matchup_shrinkage_pred'].to_numpy(), 1e-6, 1 - 1e-6)
matchup_blended_proba = np.clip(val_df['matchup_shrinkage_blended_pred'].to_numpy(), 1e-6, 1 - 1e-6)

print(f"\n{len(val_df):,} PA rows with all three predictions available "
      f"({len(pa_outcome) - len(val_df):,} dropped, missing last_season_ba or pitcher rate)")

print("\n" + "#" * 75)
print("# BATTER-ONLY SHRINKAGE — GAME GRAIN")
print("#" * 75)
_, batter_only_game_metrics, _ = run_pa_vs_game_grain_check(
    val_df, y_val, batter_only_proba, group_cols=("batter_id", "gamepk"), verbose=False,
)
for k_ in ("reliability", "resolution", "roc_auc", "brier"):
    print(f"  {k_:12s} {batter_only_game_metrics[k_]:.4f}")

print("\n" + "#" * 75)
print("# MATCHUP (LOG5 PITCHER-AUGMENTED) SHRINKAGE — GAME GRAIN")
print("#" * 75)
_, matchup_game_metrics, _ = run_pa_vs_game_grain_check(
    val_df, y_val, matchup_proba, group_cols=("batter_id", "gamepk"), verbose=False,
)
for k_ in ("reliability", "resolution", "roc_auc", "brier"):
    print(f"  {k_:12s} {matchup_game_metrics[k_]:.4f}")

print("\n" + "#" * 75)
print("# MATCHUP, PITCHER-SIDE BLENDED (in-season cascade, not static) — GAME GRAIN")
print("#" * 75)
_, matchup_blended_game_metrics, _ = run_pa_vs_game_grain_check(
    val_df, y_val, matchup_blended_proba, group_cols=("batter_id", "gamepk"), verbose=False,
)
for k_ in ("reliability", "resolution", "roc_auc", "brier"):
    print(f"  {k_:12s} {matchup_blended_game_metrics[k_]:.4f}")

verdict_static = summarize_verdict(batter_only_game_metrics, matchup_game_metrics)
verdict_blended_vs_batter_only = summarize_verdict(batter_only_game_metrics, matchup_blended_game_metrics)
verdict_blended_vs_static = summarize_verdict(matchup_game_metrics, matchup_blended_game_metrics)

print("\n" + "#" * 75)
print("# VERDICTS")
print("#" * 75)
for label, v in [
    ("matchup (static pitcher) vs batter-only", verdict_static),
    ("matchup (blended pitcher) vs batter-only", verdict_blended_vs_batter_only),
    ("matchup (blended pitcher) vs matchup (static pitcher)", verdict_blended_vs_static),
]:
    print(f"  {label}: {v['verdict']}  "
          f"(reliability delta {v['reliability_delta']:+.4f}, resolution delta {v['resolution_delta']:+.4f})")

# ── Blake Snell worked examples ───────────────────────────────────────────────
snell_pa = val_df[
    (val_df['pitcher_id'] == CASE_STUDY_PITCHER_ID) & (val_df['pitcher_role'] == 'sp')
].copy()
snell_pa['game_date'] = pd.to_datetime(snell_pa['game_date'])
snell_pa = snell_pa.sort_values(['game_date', 'gamepk', 'play_id'])

league_avg_2024 = val_df[TARGET].mean()
snell_rate = snell_pa['pitcher_last_season_pa_hit_rate'].iloc[0] if len(snell_pa) else None

# One example PA per start, spread across the season, picking the batter
# with the most extreme last_season_ba difference from Snell's own rate
# each time so the worked examples show real variety, not near-duplicates.
examples = []
for gamepk, game_df in snell_pa.groupby('gamepk', sort=False):
    row = game_df.iloc[(game_df['last_season_ba'] - game_df['last_season_ba'].median()).abs().argmax()] \
        if len(game_df) > 1 else game_df.iloc[0]
    examples.append(row)
examples_df = pd.DataFrame(examples).sort_values('game_date')

worked = []
for _, row in examples_df.iterrows():
    b = row['last_season_ba']
    p = row['pitcher_last_season_pa_hit_rate']
    L = league_avg_2024
    matchup_target = np.clip(b * p / L, 0, 1) if pd.notna(b) and pd.notna(p) else None
    worked.append({
        "game_date": row['game_date'].strftime('%Y-%m-%d'),
        "opponent_team": row['batter_team_name'],
        "batter_name": row['batter_name'],
        "batter_last_season_ba": None if pd.isna(b) else round(float(b), 3),
        "pitcher_last_season_hit_rate": None if pd.isna(p) else round(float(p), 3),
        "league_avg_2024": round(float(L), 3),
        "matchup_target": None if matchup_target is None else round(float(matchup_target), 4),
        "batter_only_pred": round(float(row['shrinkage_pred']), 4),
        "matchup_pred": round(float(row['matchup_shrinkage_pred']), 4),
        "matchup_blended_pred": round(float(row['matchup_shrinkage_blended_pred']), 4),
        "actual_is_hit": int(row[TARGET]),
    })

print(f"\nBlake Snell worked examples ({len(worked)} starts):")
for w in worked:
    print(w)

# ── Save results ──────────────────────────────────────────────────────────────
OUT_DIR = BASE_DIR / "baseline" / "statistical"
result = {
    "date": datetime.now().strftime('%Y-%m-%d'),
    "val_season": TARGET_SEASON,
    "n_pa": len(val_df),
    "batter_only_game_metrics": {k: float(v) for k, v in batter_only_game_metrics.items() if k != 'calibration_df'},
    "matchup_game_metrics": {k: float(v) for k, v in matchup_game_metrics.items() if k != 'calibration_df'},
    "matchup_blended_game_metrics": {k: float(v) for k, v in matchup_blended_game_metrics.items() if k != 'calibration_df'},
    "verdict_static_vs_batter_only": verdict_static,
    "verdict_blended_vs_batter_only": verdict_blended_vs_batter_only,
    "verdict_blended_vs_static": verdict_blended_vs_static,
    "pitcher_name": "Blake Snell",
    "pitcher_last_season_hit_rate": None if snell_rate is None or pd.isna(snell_rate) else round(float(snell_rate), 3),
    "worked_examples": worked,
}
with open(OUT_DIR / "matchup_baseline_verdict.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nSaved {OUT_DIR / 'matchup_baseline_verdict.json'}")
print("\nDone.")
