
import yaml
from datetime import datetime
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score, ConfusionMatrixDisplay
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from models.hit_predictor.utils.eval import evaluate_hit_predictor, plot_calibration_curve

import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
from models.hit_predictor.utils.mlflow_logging import log_evaluation_to_mlflow, get_git_sha

import models.hit_predictor.processing.pipeline as pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats

STAGE = "v2-rolling-feats"

# Short rolling window size, in games. Change this one constant to try a different
# recent-form window — it flows through every build_*_rolling_stats(window=...) call below.
SHORT_WINDOW_GAMES = 10

pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── 1. Config ────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET           = cfg["bucket"]
REGION           = cfg["region"]
TRAIN_SEASONS    = cfg["train_seasons"]
FEATURE_SEASONS  = cfg["feature_seasons"]
TARGET           = cfg["target_column"]
DATE_COL         = cfg["date_column"]
TEST_SEASON      = cfg["test_season"]
VAL_SEASON       = cfg["val_season"]
MODEL_NAME       = cfg["model_name"]


# Seasons used for model fitting (everything that isn't val or test)
FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]

# have to drop 2017 bc of lagged stats - can fill in later
FIT_SEASONS.remove(2017)

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))

# ---------------------------------------------------------------------------- #
#                                 READ IN DATA                                 #
# ---------------------------------------------------------------------------- #
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
    "s3://{bucket}/processed_data/prepared/playbyplay/{season}/",
    TRAIN_SEASONS,
    chunked=True,
)

print("\nLoading schedule...")
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/",
    TRAIN_SEASONS,
)

print("\nLoading game info...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/",
    TRAIN_SEASONS,
)

print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/",
    all_boxscore_seasons,
)

print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/",
    all_boxscore_seasons,
)

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/",
    boto3_session=boto_session,
)


game_info = pipeline.process_game_info(game_info)
pitcher_boxscore = pipeline.process_pitcher_boxscore(pitcher_boxscore)
schedule = pipeline.process_schedule(schedule)

pbp = pipeline.build_pbp_features(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

batter_season_stats = season_stats.build_batter_stats(batter_boxscore).rename(columns={'personId': 'batter_id'})
sp_season_stats = season_stats.build_pbp_pitcher_feats(pbp, pitcher_role='sp')
pitcher_season_stats = season_stats.build_pitcher_stats(pitcher_boxscore).rename(columns={'personId': 'pitcher_id'})

# rolling season (expanding, resets each season) + rolling short (trailing
# SHORT_WINDOW_GAMES, carries across season boundaries) — same stat categories as
# season_stats above, updated game-by-game instead of once a year. personId is
# renamed to the entity col up front so these merge cleanly on ['gamepk', 'batter_id'
# / 'pitcher_id'] below, matching how sp_season_stats already merges on pitcher_id.
batter_rolling_season_stats = (
    rolling_stats.build_batter_rolling_stats(batter_boxscore, window='season')
    .rename(columns={'personId': 'batter_id'})
)
batter_rolling_short_stats = (
    rolling_stats.build_batter_rolling_stats(batter_boxscore, window=SHORT_WINDOW_GAMES)
    .rename(columns={'personId': 'batter_id'})
)

pitcher_rolling_season_stats = (
    rolling_stats.build_pitcher_rolling_stats(pitcher_boxscore, window='season')
    .rename(columns={'personId': 'pitcher_id'})
)
pitcher_rolling_short_stats = (
    rolling_stats.build_pitcher_rolling_stats(pitcher_boxscore, window=SHORT_WINDOW_GAMES)
    .rename(columns={'personId': 'pitcher_id'})
)

sp_rolling_season_stats = rolling_stats.build_pbp_pitcher_rolling_feats(pbp, window='season', pitcher_role='sp')
sp_rolling_short_stats = rolling_stats.build_pbp_pitcher_rolling_feats(pbp, window=SHORT_WINDOW_GAMES, pitcher_role='sp')


# ---------------------------------------------------------------------------- #
#                              ASSEMBLE MODEL FRAME                            #
# ---------------------------------------------------------------------------- #
# season_stats keyed by (game_season, entity_id) — one row per player-season, same
# number for every game that season. rolling_stats keyed by (gamepk, entity_id) —
# one row per player-game, updates every game. game_date/game_season are dropped
# from the rolling frames before merging since pa_outcome already carries those from
# game_info/schedule; keeping them would collide as *_x/*_y on every merge below.
model_df = pa_outcome.merge(
    batter_season_stats, on=['game_season', 'batter_id'], how='left'
)
model_df = model_df.merge(
    pitcher_season_stats, on=['game_season', 'pitcher_id'], how='left'
)
model_df = model_df.merge(sp_season_stats, on=['game_season', 'pitcher_id'], how='left')

for rolling_df in (
    batter_rolling_season_stats, batter_rolling_short_stats,
    pitcher_rolling_season_stats, pitcher_rolling_short_stats,
    sp_rolling_season_stats, sp_rolling_short_stats,
):
    entity_col = 'batter_id' if 'batter_id' in rolling_df.columns else 'pitcher_id'
    rolling_df = rolling_df.drop(columns=['game_date', 'game_season'])
    model_df = model_df.merge(rolling_df, on=['gamepk', entity_col], how='left')