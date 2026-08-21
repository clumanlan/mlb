"""
per_game_aggregation_check.py

ROADMAP.md's top near-term priority: does aggregating existing per-PA hit
predictions to player-game grain (P(1+ hits in game), the grain a real DK
prop resolves on) reveal discrimination that per-PA evaluation is hiding —
or is the flat AUC (~0.51-0.52 across v1-v3) a real ceiling regardless of
grain? See BENCHMARKS.md §4.5 for the full reasoning.

Deliberately reuses v2_rolling_features/train.py's exact pipeline
construction and model choice (Random Forest, same seasons) unchanged --
the only thing this script adds is keeping batter_id/gamepk/game_date
attached to the val predictions so they can be aggregated afterward. If
this used a different/newer feature set, an improvement at game grain
couldn't be cleanly attributed to the grain change alone.
"""

import json
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from models.hit_predictor.utils.eval import evaluate_hit_predictor, aggregate_pa_predictions_to_game
from models.hit_predictor.utils.model_prep import add_missing_indicators, get_columns_with_nulls, impute_and_encode

import models.hit_predictor.processing.pipeline as pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import park_factors
from models.hit_predictor.processing.features import expected_role

SHORT_WINDOW_GAMES = 10
pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parent.parent
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET, REGION = cfg["bucket"], cfg["region"]
TRAIN_SEASONS, FEATURE_SEASONS = cfg["train_seasons"], cfg["feature_seasons"]
TARGET, VAL_SEASON, TEST_SEASON = cfg["target_column"], cfg["val_season"], cfg["test_season"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)  # same lagged-stats exclusion as v2

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))


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


print("Loading play-by-play...")
pbp = read_parquet_seasons("s3://{bucket}/processed_data/prepared/playbyplay/{season}/", TRAIN_SEASONS, chunked=True)
print("Loading schedule...")
schedule = read_parquet_seasons("s3://{bucket}/processed_data/games/schedule/{season}/", all_boxscore_seasons)
print("Loading game info...")
game_info = read_parquet_seasons("s3://{bucket}/processed_data/games/game_info/{season}/", TRAIN_SEASONS)
print("Loading batter boxscore...")
batter_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", all_boxscore_seasons)
print("Loading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", all_boxscore_seasons)
print("Loading player info...")
player_info = wr.s3.read_parquet(path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session)

game_info = pipeline.process_game_info(game_info)
pitcher_boxscore = pipeline.process_pitcher_boxscore(pitcher_boxscore)
schedule = pipeline.process_schedule(schedule)
pbp = pipeline.build_pbp_features(pbp, schedule, player_info)
pa_outcome = pipeline.create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

pitcher_start_depth_stats = season_stats.build_pitcher_start_depth_stats(pbp)
league_avg_start_depth = season_stats.build_league_avg_start_depth(pitcher_start_depth_stats)
pa_outcome = expected_role.assign_expected_pitcher_role(pa_outcome, pitcher_start_depth_stats, league_avg_start_depth)

batter_season_stats = season_stats.build_batter_stats(batter_boxscore).rename(columns={'personId': 'batter_id'})
pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp).rename(
    columns={'pitcher_key_id': 'expected_pitcher_key_id', 'pitcher_role': 'expected_pitcher_role'})
pitcher_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp).rename(
    columns={'pitcher_key_id': 'expected_pitcher_key_id', 'pitcher_role': 'expected_pitcher_role'})

batter_rolling_season_stats = rolling_stats.build_batter_rolling_stats(batter_boxscore, window='season').rename(columns={'personId': 'batter_id'})
batter_rolling_short_stats = rolling_stats.build_batter_rolling_stats(batter_boxscore, window=SHORT_WINDOW_GAMES).rename(columns={'personId': 'batter_id'})

ROLE_KEY_RENAME = {'pitcher_key_id': 'expected_pitcher_key_id', 'pitcher_role': 'expected_pitcher_role'}
pitcher_rolling_season_stats = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window='season').rename(columns=ROLE_KEY_RENAME)
pitcher_rolling_short_stats = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window=SHORT_WINDOW_GAMES).rename(columns=ROLE_KEY_RENAME)
pitcher_role_rolling_season_stats = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window='season').rename(columns=ROLE_KEY_RENAME)
pitcher_role_rolling_short_stats = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window=SHORT_WINDOW_GAMES).rename(columns=ROLE_KEY_RENAME)

venue_park_factors = park_factors.build_park_factors(schedule, batter_boxscore)

pitcher_tto_stats = season_stats.build_pbp_pitcher_feats_by_times_through_order(pbp).rename(columns=ROLE_KEY_RENAME)
league_tto_stats = season_stats.build_league_times_through_order_stats(pbp)

model_df = pa_outcome.merge(batter_season_stats, on=['game_season', 'batter_id'], how='left')
model_df = model_df.merge(pitcher_season_stats, on=['game_season', 'expected_pitcher_key_id', 'expected_pitcher_role'], how='left')
model_df = model_df.merge(pitcher_role_season_stats, on=['game_season', 'expected_pitcher_key_id', 'expected_pitcher_role'], how='left')
model_df = model_df.merge(venue_park_factors, on=['game_season', 'venue_id'], how='left')
model_df = model_df.merge(pitcher_tto_stats, on=['game_season', 'expected_pitcher_key_id', 'expected_pitcher_role'], how='left')
model_df = model_df.merge(league_tto_stats, on='game_season', how='left')

for rdf in (batter_rolling_season_stats, batter_rolling_short_stats):
    rdf = rdf.drop(columns=['game_date', 'game_season'])
    model_df = model_df.merge(rdf, on=['gamepk', 'batter_id'], how='left')

for rdf in (pitcher_rolling_season_stats, pitcher_rolling_short_stats, pitcher_role_rolling_season_stats, pitcher_role_rolling_short_stats):
    rdf = rdf.drop(columns=['game_date', 'game_season'])
    model_df = model_df.merge(rdf, on=['gamepk', 'expected_pitcher_key_id', 'expected_pitcher_role'], how='left')

model_df.loc[model_df['expected_pitcher_role'] == 'bullpen', 'pitcher_throw_hand'] = None

NON_FEATURE_COLS = {
    'gamepk', 'play_id', 'batter_id', 'pitcher_id', 'pitcher_team_id',
    'starting_pitcher_id', 'expected_pitcher_key_id',
    'batter_team_name', 'pitcher_name', 'batter_name',
    'venue_id', 'pitcher_role', 'game_date', 'game_season', TARGET,
}
CAT_FEATS = ['pitcher_throw_hand', 'batter_bat_side', 'weather_condition', 'expected_pitcher_role']
NUM_FEATS = [c for c in model_df.columns if c not in NON_FEATURE_COLS and c not in CAT_FEATS
             and pd.api.types.is_numeric_dtype(model_df[c])]

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df = model_df[model_df["game_season"] == VAL_SEASON].copy()

X_train_raw = train_df[NUM_FEATS + CAT_FEATS].copy()
y_train = train_df[TARGET].copy()
X_val_raw = val_df[NUM_FEATS + CAT_FEATS].copy()
y_val = val_df[TARGET].copy()

indicator_cols = get_columns_with_nulls(X_train_raw, NUM_FEATS)
indicator_col_names = [f"{c}_isnull" for c in indicator_cols]
X_train_ind = add_missing_indicators(X_train_raw, indicator_cols)
X_val_ind = add_missing_indicators(X_val_raw, indicator_cols)

X_train, X_val = impute_and_encode(X_train_ind, X_val_ind, num_cols=NUM_FEATS, cat_cols=CAT_FEATS)
X_train = pd.concat([X_train, X_train_ind[indicator_col_names]], axis=1)
X_val = pd.concat([X_val, X_val_ind[indicator_col_names]], axis=1)

print("\nTraining Random Forest (same config as v2_rolling_features)...")
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features="sqrt",
                                n_jobs=-1, oob_score=True, random_state=42)
model.fit(X_train, y_train)
print(f"OOB score: {model.oob_score_:.4f}")

proba = model.predict_proba(X_val)[:, 1]

# ---------------------------------------------------------------------------- #
#                        PA GRAIN vs GAME GRAIN COMPARISON                     #
# ---------------------------------------------------------------------------- #
print("\n" + "#" * 75)
print("# PA GRAIN (existing evaluation, sanity check against v2's table row)")
print("#" * 75)
pa_metrics = evaluate_hit_predictor(y_true=y_val, y_prob=proba, n_bins=10, min_n=500)

pa_results = val_df[['batter_id', 'gamepk', 'game_date']].copy()
pa_results['is_hit'] = y_val.values
pa_results['pred_prob'] = proba

game_results = aggregate_pa_predictions_to_game(pa_results)
print(f"\n{len(pa_results)} PA rows -> {len(game_results)} batter-game rows "
      f"(mean {pa_results.groupby(['batter_id', 'gamepk']).size().mean():.2f} PA/game)")

print("\n" + "#" * 75)
print("# GAME GRAIN (aggregated: P(1+ hits in game) vs actual any-hit)")
print("#" * 75)
game_metrics = evaluate_hit_predictor(
    y_true=game_results['game_is_hit'], y_prob=game_results['game_pred_prob'],
    n_bins=10, min_n=200, base_rate=game_results['game_is_hit'].mean(),
)

summary = {
    "pa_grain": {k: (v if not hasattr(v, "to_dict") else "see calibration_df") for k, v in pa_metrics.items()},
    "game_grain": {k: (v if not hasattr(v, "to_dict") else "see calibration_df") for k, v in game_metrics.items()},
    "n_pa_rows": len(pa_results),
    "n_game_rows": len(game_results),
}
out_path = Path(__file__).parent / "per_game_aggregation_check_results.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSaved summary to {out_path}")

print("\n" + "#" * 75)
print("# HEADLINE COMPARISON")
print("#" * 75)
for key in ("reliability", "resolution", "roc_auc", "brier", "log_loss", "ece"):
    pa_v, game_v = pa_metrics[key], game_metrics[key]
    print(f"  {key:12s}  PA={pa_v:.4f}   GAME={game_v:.4f}")
