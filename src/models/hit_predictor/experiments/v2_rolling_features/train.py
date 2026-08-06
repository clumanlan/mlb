
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

STAGE = "v2-rolling-feats"

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

pbp = pipeline.build_pbp_features(pbp, schedule)

pa_outcome = pipeline.create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

batter_season_stats = season_stats.build_batter_stats(batter_boxscore)
sp_season_stats = season_stats.build_pbp_pitcher_feats(pbp, pitcher_role='sp')
pitcher_season_stats = season_stats.build_pitcher_stats(pitcher_boxscore)

# rolling season + rolling short 
# i think i'll just do rolling + optimize for it and put it in production from ehre