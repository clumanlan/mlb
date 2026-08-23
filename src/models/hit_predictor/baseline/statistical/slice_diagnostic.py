"""
Slice diagnostic: "what is the model good at, what is it bad at" — answered
by pointing shared/model_dashboard's existing, tested slicing/contribution/
bootstrap tooling (built for exactly this, per CLAUDE.md's Model Layer
section, never wired to hit_predictor's actual predictions) at the best
model built this session (matchup, pitcher-blended shrinkage).

Includes times_through_order (the batter's Nth PA vs the SAME starter this
game, capped at 3 — Lichtman's TTOP research, already computed by
pipeline.py's _add_pbp_times_through_order but not carried into
create_pa_outcome's column selection) as an explicit slice, since "is the
model capturing the batter-has-seen-this-pitcher-before effect" was the
concrete example that prompted this script.

Run from models/hit_predictor/ with: python -m baseline.statistical.slice_diagnostic
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Slice-average log_loss (not resolution) is the right lens here even at PA
grain: unlike resolution, it isn't fooled by n=1 noise as long as a slice
pools ENOUGH PAs (same principle as the matchup-extremity grid) -- which is
why min_n is raised well above the dashboard's default of 50.
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
from shared.model_dashboard.logic.bootstrap import bootstrap_ci
from shared.model_dashboard.logic.contribution import compute_contribution
from shared.model_dashboard.logic.slicing import generate_single_slices

BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET = cfg["bucket"]
REGION = cfg["region"]
TARGET = cfg["target_column"]
TARGET_SEASON = cfg["val_season"]  # 2024
FEATURE_SEASON = TARGET_SEASON - 1  # 2023
SHRINKAGE_K = 100.0
MIN_N = 500  # raised well above the dashboard's default of 50 -- see module docstring
PRED_COL = "matchup_shrinkage_blended_pred"

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

val_df = pa_outcome.dropna(subset=[PRED_COL]).copy()

# ── Candidate slice columns ───────────────────────────────────────────────────
# times_through_order: same formula as pipeline.py's _add_pbp_times_through_order,
# derived here from columns create_pa_outcome already carries (batter_pa_number,
# pitcher_role) rather than plumbing it through the shared pipeline -- it's a
# post-hoc diagnostic column, not a model feature.
val_df['times_through_order'] = np.where(
    val_df['pitcher_role'] == 'sp', val_df['batter_pa_number'].clip(upper=3), np.nan
)

is_switch = (val_df['batter_bat_side'] == 'S').fillna(False).astype(bool)
is_same_hand = (val_df['batter_bat_side'] == val_df['pitcher_throw_hand']).fillna(False).astype(bool)
val_df['platoon_matchup'] = 'opposite_hand'
val_df.loc[is_same_hand, 'platoon_matchup'] = 'same_hand'
val_df.loc[is_switch, 'platoon_matchup'] = 'switch_hitter'
val_df.loc[val_df['batter_bat_side'].isna() | val_df['pitcher_throw_hand'].isna(), 'platoon_matchup'] = np.nan

SLICE_COLS = ['times_through_order', 'platoon_matchup', 'pitcher_role', 'batting_order']

slices = generate_single_slices(val_df, SLICE_COLS)
print(f"\n{len(slices)} candidate slice values across {SLICE_COLS}")

contribution_df = compute_contribution(val_df, slices, target_col=TARGET, pred_col=PRED_COL, min_n=MIN_N)

print("\n" + "#" * 75)
print(f"# SLICE CONTRIBUTION RANKING (min_n={MIN_N}) — most-hurting slice first")
print("#" * 75)
print(contribution_df.to_string(index=False))

# ── Explicit times_through_order breakout (the concrete example this script exists for) ──
tto_rows = contribution_df[contribution_df['feature'] == 'times_through_order']
print("\n" + "#" * 75)
print("# TIMES THROUGH THE ORDER — explicit breakout")
print("#" * 75)
print(tto_rows.to_string(index=False))

overall_loss = float(
    val_df.apply(lambda r: -(r[TARGET] * np.log(np.clip(r[PRED_COL], 1e-15, 1 - 1e-15))
                              + (1 - r[TARGET]) * np.log(1 - np.clip(r[PRED_COL], 1e-15, 1 - 1e-15))), axis=1).mean()
)
print(f"\nOverall log_loss (all {len(val_df):,} PAs): {overall_loss:.4f}")

# ── Bootstrap CIs: top 5 slices by |contribution|, plus every TTO value explicitly ──
check_slices = (
    contribution_df.reindex(contribution_df['contribution'].abs().sort_values(ascending=False).index).head(5)
).to_dict('records')
tto_values = [s for s in slices if s['feature'] == 'times_through_order']
for s in tto_values:
    if not any(c['feature'] == s['feature'] and c['value'] == s['value'] for c in check_slices):
        check_slices.append(s)

print("\n" + "#" * 75)
print("# BOOTSTRAP 95% CIs — is the apparent gap real, or small-sample noise?")
print("#" * 75)
ci_results = []
for s in check_slices:
    feature, value = s['feature'], s['value']
    subset = val_df[val_df[feature] == value]
    if len(subset) < MIN_N:
        continue
    point, lo, hi = bootstrap_ci(subset, target_col=TARGET, pred_col=PRED_COL)
    gap_is_real = lo > overall_loss or hi < overall_loss
    print(f"  {feature}={value}: n={len(subset):,} log_loss={point:.4f} "
          f"95% CI=[{lo:.4f}, {hi:.4f}]  overall={overall_loss:.4f}  "
          f"{'REAL GAP' if gap_is_real else 'CI overlaps overall — not distinguishable from noise'}")
    ci_results.append({
        "feature": feature, "value": str(value), "n": len(subset),
        "log_loss": point, "ci_lo": lo, "ci_hi": hi,
        "overall_log_loss": overall_loss, "gap_is_real": bool(gap_is_real),
    })

# ── Save results ──────────────────────────────────────────────────────────────
OUT_DIR = BASE_DIR / "baseline" / "statistical"
result = {
    "date": datetime.now().strftime('%Y-%m-%d'),
    "val_season": TARGET_SEASON,
    "model": PRED_COL,
    "min_n": MIN_N,
    "overall_log_loss": overall_loss,
    "n_pa": len(val_df),
    "contribution_ranking": contribution_df.to_dict('records'),
    "bootstrap_ci": ci_results,
}
with open(OUT_DIR / "slice_diagnostic.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nSaved {OUT_DIR / 'slice_diagnostic.json'}")
print("\nDone.")
