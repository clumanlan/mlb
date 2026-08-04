import pandas as pd 

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


import numpy as np
import pandas as pd

from schema import PBP

# Bump by hand as the project progresses (e.g. "v0-baseline" -> "v1-tuned")
STAGE = "v0-baseline"

import sys
print(sys.path)
# need to shift play by play count data in the processing 
# get game_info's that are missing from the playbyplay df

pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parent.parent

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

all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))
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
# ---------------------------------------------------------------------------- #
#                                 PROCESS DATA                                 #
# ---------------------------------------------------------------------------- #

def process_game_info(df):

    df = (
        df
        .assign(
            gamepk = lambda x: x["gamepk"].astype(str),
            weather_temp = lambda x: x["weather_temp"].astype(str).str.extract(r"(\d+)")[0].astype(float),
            weather_wind_speed = lambda x: x["weather_wind"].str.split(",").str[0].str.extract(r"(\d+)").astype(float),
            weather_wind_direction = lambda x: x['weather_wind'].str.split(",").str[1]
        )
        .drop(['weather_wind'], axis=1)
    )

    return df


def process_schedule(df):
    df = (
        df
        .assign(
            gamepk = lambda x: x["gamepk"].astype(str),
            game_date = lambda x: pd.to_datetime(x["game_date"])
        )
    )

    # filter out <1% of games that have different game dates
    df = df.loc[df.groupby(['gamepk'])['game_datetime'].idxmin()]

    return df 





# ------------------------------- PLAY BY PLAY ------------------------------- #

def initial_pbp_processing(df, target_col):
    """Row-level pitch features (no ordering/grouping dependency)."""

    df = (
        df
        .assign(
            ** {target_col: lambda x: x["play_result"].isin(PBP.HITS).astype(int)},
            gamepk  = lambda x: x["gamepk"].astype(str),
            batter_id  = lambda x: x["batter_id"].astype(str),
            pitcher_id = lambda x: x["pitcher_id"].astype(str),
            speed_retention = lambda x: x['end_speed'] / x['start_speed'],
            perceived_velo = lambda x: x['start_speed'] * (60.5 / (60.5 - x['extension'])),
            is_swinging_strike = lambda x: x['pitch_call'].isin(['Swinging Strike', 'Swinging Strike (Blocked)']),
            is_called_strike = lambda x: x['pitch_call'] == 'Called Strike',
            # is chase — swung at a ball outside zone
            is_chase = lambda x: (x['is_swinging_strike']) & (x['zone'].isin([11, 12, 13, 14])),
            # is zone swing — swung at a strike
            is_zone_swing = lambda x: (x['pitch_call'].isin(['Swinging Strike', 'Foul', 'In Play'])) & (x['zone'].between(1, 9)),
            # plate location relative to batter's strike zone (normalized)
            plate_z_normalized = lambda x: (x['plate_z'] - x['strike_zone_bottom']) / (x['strike_zone_top'] - x['strike_zone_bottom']),
            # 0 = bottom of zone, 1 = top, negative = below, >1 = above
            # total movement magnitude
            movement_magnitude = lambda x: np.sqrt(x['pfx_x']**2 + x['pfx_z']**2)
        )
    )

    return df

def add_pbp_pitch_state(df):

    """Adds is_first_pitch, count_state. Sorts df — must run after initial_pbp_processing,
    before anything that depends on original row order."""  

    # adjust pitch count 
    df = df.sort_values(by=['gamepk', 'play_id', 'event_index'])

    for col in ['count_balls', 'count_strikes']:
        df[col] = df.groupby(['gamepk', 'play_id'])[col].shift(1)

    df.loc[df['pitch_number'] == 1, 'count_balls'] = 0
    df.loc[df['pitch_number'] == 1, 'count_strikes'] = 0

    df = (
        df
        .assign(
            # is first pitch of PA
            is_first_pitch = lambda x: (x['count_balls'] == 0) & (x['count_strikes'] == 0),
            # count leverage state — simplified
            count_state = lambda x: x['count_balls'].astype(str) + '-' + x['count_strikes'].astype(str)
        )
    )

    return df

def add_pbp_starting_pitcher(df):

    """Adds pitcher_role. Starter = first pitcher to throw for a team in the game
    (by play_id order) — doesn't handle openers."""

    starting_pitcher_ids = (
        df
        .sort_values(by=['gamepk','play_id'])
        .groupby(['gamepk','pitcher_team_id'])
        .first()
        .reset_index()
        [['gamepk', 'pitcher_team_id', 'pitcher_id']]
    )

    df = (
        df
        .merge(
            starting_pitcher_ids, 
            on=['gamepk', 'pitcher_team_id', 'pitcher_id'],
            how='left',
            indicator=True)
        .assign(
            pitcher_role = lambda x: np.where(x['_merge']=='both', 'sp', 'bullpen')
        )
        .drop(['_merge'], axis=1)
    )

    return df

def add_pbp_pa(df):
        
    # create plate appearance number 
    ba_pbp_num = (
        df[['gamepk', 'play_id', 'batter_id']]
        .drop_duplicates()
        .sort_values(by=['gamepk', 'play_id'])
        .assign(batter_pa_number = lambda x: x.groupby(['gamepk','batter_id']).cumcount() + 1)
    )

    df = (
        df
        .merge(
            ba_pbp_num,
            on=['gamepk', 'play_id', 'batter_id'],
            how='left'
        )
    )

    return df 

def add_pbp_game_date(df, schedule):
    df = (
        df
        .merge(
            schedule[['gamepk', 'game_date']],
            on=['gamepk'],
            how='left'
        )
        .assign(
            game_season = lambda x: x['game_date'].dt.year
        )
    )

    return df 


def build_pbp_features(pbp: pd.DataFrame, schedule, target_col: str = "is_hit") -> pd.DataFrame:

    pbp = initial_pbp_processing(pbp, target_col)
    pbp = add_pbp_pitch_state(pbp)   
    pbp = add_pbp_starting_pitcher(pbp)
    pbp = add_pbp_pa(pbp)
    pbp = add_pbp_game_date(pbp, schedule)

    return pbp

# ----------------------------- PITCHER BOXSCORE ----------------------------- #

def convert_ip_to_decimal(ip: pd.Series) -> pd.Series:
    """Baseball IP format (6.2 = 6 2/3 innings) to decimal."""
    full_innings = np.floor(ip)
    partial = (ip - full_innings).round(1)

    bad = ~partial.isin([0.0, 0.1, 0.2])
    if bad.any():
        raise ValueError(f"Unexpected IP partial value(s): {ip[bad].tolist()}")

    return full_innings + (partial / 0.3)


def process_pitcher_boxscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(ip=lambda x: convert_ip_to_decimal(x['ip']))


def create_batting_order(batter_boxscore):

    df = (
        batter_boxscore[~batter_boxscore['batting_order'].isnull()]
        [["gamepk", "personId", "batting_order"]]
        .drop_duplicates(subset=["gamepk", "personId"])
        .rename(columns={"personId": "batter_id"})
    )

    return df.assign(batting_order = lambda x: x['batting_order'].astype(int))

def create_pa_outcome(pbp, batter_boxscore, game_info, schedule):

    batting_order = create_batting_order(batter_boxscore)
    game_info = game_info[["gamepk", "game_season", "weather_condition", "weather_temp"]].drop_duplicates("gamepk")
    schedule = schedule[["gamepk", "game_date"]].drop_duplicates("gamepk")
    
    pa_outcome = pbp[['gamepk', 'batter_team_name', 'play_id', 'pitcher_id', 'pitcher_name', 'batter_id', 'batter_name', 'is_hit']].drop_duplicates().reset_index(drop=True)

    pa_outcome = pa_outcome.merge(
        schedule,
        on="gamepk", how="left",
    )
    # TODO: figure out why some pbp game_pks are missing from game_info
    pa_outcome = pa_outcome.merge(
        game_info,
        on="gamepk", how="left",
    )

    pa_outcome = pa_outcome.merge(batting_order, on=["gamepk", "batter_id"], how="inner")

    return pa_outcome