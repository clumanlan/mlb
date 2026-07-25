
import yaml
from datetime import datetime
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
from sklearn.calibration import calibration_curve

# need to shift play by play count data in the processing 
# get game_info's that are missing from the playbyplay df

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

# --------------------------------- GAME INFO -------------------------------- #
game_info = (
    game_info
    .assign(
        gamepk = lambda x: x["gamepk"].astype(str),
        weather_temp= lambda x: x["weather_temp"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
)

# --------------------------------- SCHEDULE --------------------------------- #
schedule = (
    schedule
    .assign(
        gamepk = lambda x: x["gamepk"].astype(str),
        game_date = lambda x: pd.to_datetime(x["game_date"])
    )
)

# filter out <1% of games that have different game dates
schedule = schedule.loc[schedule.groupby(['gamepk'])['game_datetime'].idxmin()]

# ------------------------------- PLAY BY PLAY ------------------------------- #

# dropped entirely - duplicates or not needed
DROP_COLS = [
    'batter_player_name',
    'pitcher_player_name',
]

# never features - just keys for joining/grouping
INDEX_COLS = [
    'gamepk',
    'play_id',
    'event_index',
    'pitch_number',
]

# entity keys - used for aggregations and lookups
ENTITY_COLS = [
    'batter_id',
    'batter_name',
    'batter_team_id',
    'batter_team_name',
    'pitcher_id',
    'pitcher_name',
    'pitcher_team_id',
    'pitcher_team_name',
]

# game state context - condition features or slice filters
CONTEXT_COLS = [
    'inning',
    'half_inning',
    'count_balls',
    'count_strikes',
    'count_outs',
]

# target / label adjacent - what happened
OUTCOME_COLS = [
    'play_result',
    'is_pitch',
    'is_in_play',
    'is_strike',
    'is_ball',
    'is_out',
    'pitch_call',
]

# raw pitch features - inputs to stuff/movement models
PITCH_PHYSICS_COLS = [
    'pitch_type',
    'start_speed',
    'end_speed',
    'extension',
    'plate_time',
    'spin_rate',
    'spin_direction',
    'pfx_x',
    'pfx_z',
    'break_vertical',
    'break_vertical_induced',
    'break_horizontal',
    'break_angle',
    'break_length',
    'break_y',
    'release_pos_x',
    'release_pos_y',
    'release_pos_z',
    'vx0',
    'vy0',
    'vz0',
    'ax',
    'ay',
    'az',
]

# location features - inputs to location/command models
PITCH_LOCATION_COLS = [
    'plate_x',
    'plate_z',
    'zone',
    'strike_zone_top',
    'strike_zone_bottom',
    'sz_x',
    'sz_y',
    'type_confidence',
]

# batted ball features - null for non-contact, inputs to xBA/xSLG
BATTED_BALL_COLS = [
    'launch_speed',
    'launch_angle',
    'total_distance',
    'trajectory',
    'hardness',
    'hit_location',
    'hit_coord_x',
    'hit_coord_y',
]

# play result bins 
HITS = {"Single", "Double", "Triple", "Home Run"}

OUTS = {
    "Strikeout", "Strikeout Double Play",
    "Groundout", "Forceout", "Grounded Into DP", "Double Play", 
    "Triple Play", "Fielders Choice Out", "Field Out",
    "Flyout", "Pop Out", "Sac Fly",
    "Lineout", "Bunt Groundout", "Bunt Pop Out", "Bunt Lineout",
    "Fielders Choice", 'Sac Bunt'
}

ON_BASE_NOT_HIT = {"Walk", "Intent Walk", "Hit By Pitch", "Catcher Interference"}

ERROR = 'Field Error'

# For pitcher PA denominator
PA_OUTCOMES = HITS | OUTS | ON_BASE_NOT_HIT

pbp = (
    pbp
    .assign(
        ** {TARGET: lambda x: x["play_result"].isin(HITS).astype(int)},
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

# adjust pitch count 
pbp = pbp.sort_values(by=['gamepk', 'play_id', 'event_index'])

for col in ['count_balls', 'count_strikes']:
    pbp[col] = pbp.groupby(['gamepk', 'play_id'])[col].shift(1)

pbp.loc[pbp['pitch_number'] == 1, 'count_balls'] = 0
pbp.loc[pbp['pitch_number'] == 1, 'count_strikes'] = 0

pbp = (
    pbp
    .assign(
        # is first pitch of PA
        is_first_pitch = lambda x: (x['count_balls'] == 0) & (x['count_strikes'] == 0),
        # count leverage state — simplified
        count_state = lambda x: x['count_balls'].astype(str) + '-' + x['count_strikes'].astype(str)
    )
)

# create starting pitcher 
starting_pitcher_pbp = (
    pbp
    .sort_values(by=['gamepk','play_id'])
    .groupby(['gamepk','pitcher_team_id'])
    .first()
    .reset_index()
    [['gamepk', 'pitcher_team_id', 'pitcher_id']]
)

pbp = (
    pbp
    .merge(
        starting_pitcher_pbp, 
        on=['gamepk', 'pitcher_team_id', 'pitcher_id'],
        how='left',
        indicator=True)
    .assign(
        pitcher_role = lambda x: np.where(x['_merge']=='both', 'sp', 'bullpen')
    )
    .drop(['_merge'], axis=1)
)

# create plate appearance number 
ba_pbp_num = (
    pbp[['gamepk', 'play_id', 'batter_id']]
    .drop_duplicates()
    .sort_values(by=['gamepk', 'play_id'])
    .assign(batter_pa_number = lambda x: x.groupby(['gamepk','batter_id']).cumcount() + 1)
)

pbp = (
    pbp
    .merge(
        ba_pbp_num,
        on=['gamepk', 'play_id', 'batter_id'],
        how='left'
    )
)

# create game date 
pbp = (
    pbp
    .merge(
        schedule[['gamepk', 'game_date']],
        on=['gamepk'],
        how='left'
    )
    .assign(
        game_season = lambda x: x['game_date'].dt.year
    )
)


# ----------------------------- PITCHER BOXSCORE ----------------------------- #

def convert_ip_to_decimal(ip: float) -> float:
    """Baseball IP format (6.2 = 6 2/3 innings) to decimal"""
    full_innings = int(ip)
    partial = round(ip % 1, 1)
    return full_innings + (partial / 0.3)

pitcher_boxscore = (
    pitcher_boxscore
    .assign(
        ip = lambda x: x['ip'].apply(convert_ip_to_decimal)
    )
)
# ------------------------------ BATTER BOXSCORE ----------------------------- #





# ---------------------------------------------------------------------------- #
#                                 CREATE TABLES                                #
# ---------------------------------------------------------------------------- #

# ------------------------------- BATTING ORDER ------------------------------ #
batting_order = (
    batter_boxscore[~batter_boxscore['batting_order'].isnull()]
    [["gamepk", "personId", "batting_order"]]
    .drop_duplicates(subset=["gamepk", "personId"])
    .rename(columns={"personId": "batter_id"})
    .assign(
        batting_order = lambda x: x['batting_order'].astype(int)
    )
)

# -------------------------------- PA OUTCOME -------------------------------- #
pa_outcome = pbp[['gamepk', 'batter_team_name', 'play_id', 'pitcher_id', 'pitcher_name', 'batter_id', 'batter_name', 'is_hit']].drop_duplicates().reset_index(drop=True)

pa_outcome = pa_outcome.merge(
    schedule[["gamepk", "game_date"]].drop_duplicates("gamepk"),
    on="gamepk", how="left",
)
# figure this out later why some game play by plays are missing from game info 
# Join game_season + weather from game_info
pa_outcome = pa_outcome.merge(
    game_info[["gamepk", "game_season", "weather_condition", "weather_temp"]].drop_duplicates("gamepk"),
    on="gamepk", how="left",
)

pa_outcome = pa_outcome.merge(batting_order, on=["gamepk", "batter_id"], how="left")

pa_outcome = pa_outcome[~pa_outcome['batting_order'].isnull()]


# ---------------------------------------------------------------------------- #
#                                BUILD FEATURES                                #
# ---------------------------------------------------------------------------- #

# ---------------------------- BATTER SEASON STATS --------------------------- #
season_batter_stats = (
    batter_boxscore
    .groupby(['personId', 'game_season'])
    [['h',  'k', 'bb', 'hr', 'ab', 'plate_appearances']].sum()
    .reset_index()
    .rename(columns={
            'h': 'season_total_h',
            'k': 'season_total_k',
            'bb': 'season_total_bb',
            'hr': 'season_total_hr',
            'ab': 'season_total_ab',
            'plate_appearances': 'season_total_plate_appearances'
        })
    .assign(
        season_ba = lambda x: np.round(x['season_total_h']/x['season_total_ab'], 2)
    )
)

last_season_batter_stats = (
    season_batter_stats
    .assign(
        game_season = lambda x: x['game_season']+1
    )
    .rename(columns={
        col: col.replace('season_', 'last_season_batter_')
        for col in season_batter_stats.columns
        if col.startswith('season_')
    })
)

pitcher_boxscore.columns.tolist()

# --------------------------- PITCHER BOXSCORE SEASON STATS --------------------------- #
season_pitcher_stats = (
    pitcher_boxscore
    .groupby(['personId', 'game_season'])
    [['h', 'r', 'er', 'bb', 'hr', 'k', 'p', 's', 'ip']].sum()
    .reset_index()
    .rename(columns={
        'h': 'season_total_h',
        'r': 'season_total_r',
        'k': 'season_total_k',
        'er': 'season_total_er',
        'bb': 'season_total_bb',
        'hr': 'season_total_hr',
        'p': 'season_total_p',
        's': 'season_total_s',
        'ip': 'season_total_ip'
    })
    .assign(
        season_whip = lambda x: np.round(
            (x['season_total_bb'] + x['season_total_h'])
            / x['season_total_ip'].replace(0, np.nan), 2),
        season_k_rate = lambda x: np.round(
            x['season_total_k']
            / x['season_total_ip'].replace(0, np.nan), 2),
        season_bb_rate = lambda x: np.round(
            x['season_total_bb']
            / x['season_total_ip'].replace(0, np.nan), 2),
        season_strike_rate = lambda x: np.round(
            x['season_total_s']
            / x['season_total_p'].replace(0, np.nan), 2),
        season_hr_rate = lambda x: np.round(
            x['season_total_hr']
            / x['season_total_ip'].replace(0, np.nan), 2),
    )
)
last_season_pitcher_stats = (
    season_pitcher_stats
    .assign(
        game_season = lambda x: x['game_season']+1
    )
    .rename(columns={
        col: col.replace('season_', 'last_season_pitcher_')
        for col in season_pitcher_stats.columns 
        if col.startswith('season_')
    })
)


# --------------------------- PITCHER PBP SEASON STATS --------------------------- #

# create a longer documentation later:
# - stuff: physical properties of a pitch
# - command: ability to find the zone
# - sequencing: is he unpredictable? 
# - durability: #'s of time through order 

# ----------------------------- STARTING PITCHER PBP SEASON STATS ----------------------------- #

pbp_sp_pitcher = pbp[pbp['pitcher_role']=='sp']

pbp_sp_pitcher['is_first_pitch_strike'] = (
    pbp_sp_pitcher['is_first_pitch'] & pbp_sp_pitcher['is_strike']
)

season_sp_pitcher_pbp_stats = (
    pbp_sp_pitcher
    .groupby(['pitcher_id', 'game_season'])
    .agg(
        # stuff
        season_pitcher_stuff_start_speed_mean = ('start_speed', 'mean'),
        season_pitcher_stuff_start_speed_max = ('start_speed', 'max'),
        season_pitcher_stuff_start_speed_std = ('start_speed', 'std'), # fatigue signal 
        season_pitcher_stuff_end_speed_mean = ('end_speed', 'mean'),
        season_pitcher_stuff_end_speed_max = ('end_speed', 'max'),
        season_pitcher_stuff_perceived_velo_mean = ('perceived_velo', 'mean'),
        season_pitcher_stuff_perceived_velo_max = ('perceived_velo', 'max'),
        season_pitcher_stuff_spin_rate_mean = ('spin_rate', 'mean'),
        season_pitcher_stuff_spin_rate_max = ('spin_rate', 'max'),
        season_pitcher_stuff_movement_magnitude_mean = ('movement_magnitude', 'mean'),
        season_pitcher_stuff_movement_magnitude_max = ('movement_magnitude', 'max'),
        season_pitcher_stuff_pfx_z_mean = ('pfx_z', 'mean'), # vertical spin induced movement
        season_pitcher_stuff_pfx_z_max = ('pfx_z', 'max'),
        season_pitcher_stuff_extension_mean = ('extension', 'mean'),
        season_pitcher_stuff_extension_max = ('extension', 'max'),
        season_pitcher_stuff_extension_std = ('extension', 'std'),
        season_pitcher_stuff_speed_retention_mean = ('speed_retention', 'mean'),  # end/start ratio

        # command 
        season_pitcher_command_in_play_rate = ('is_in_play', 'mean'),
        season_pitcher_command_swinging_strike_rate = ('is_swinging_strike', 'mean'),
        season_pitcher_command_plate_x_std = ('plate_x', 'std'), # how does horizontal landing vary 
        season_pitcher_command_plate_z_normalized_std = ('plate_z_normalized', 'std'), # how does vertical landing vary 
        season_pitcher_command_zone_rate = ('zone', lambda x: x.isin(range(1,10)).mean()),
        season_pitcher_command_ball_rate = ('is_ball', 'mean'),
        season_pitcher_command_strike_rate = ('is_strike', 'mean'),
        season_pitcher_command_called_strike_rate = ('is_called_strike', 'mean'),
        season_pitcher_command_chase_rate = ('is_chase', 'mean'),
        season_pitcher_command_zone_swing_rate = ('is_zone_swing', 'mean'),
        season_pitcher_command_first_pitch_strike_rate = ('is_first_pitch_strike', 'mean'),
    )
    .reset_index()
)


# last pitch stats 
last_pitch_pbp = pbp_sp_pitcher[pbp_sp_pitcher['pitch_number'] == pbp_sp_pitcher.groupby(['gamepk','play_id'])['pitch_number'].transform('max')].reset_index(drop=True)

season_last_pitch_pbp_sp_pitcher_stats = (
    last_pitch_pbp
    .groupby(['pitcher_id', 'game_season'])
    .agg(
        # PA length — what you have, good to keep
        season_pitcher_pa_pitch_count_mean = ('pitch_number', 'mean'),  # avg pitches per PA
        season_pitcher_pa_pitch_count_std = ('pitch_number', 'std'),

        # volume
        season_pitcher_pa_total = ('play_result', 'count'),

        # outcome rates 
        season_pitcher_pa_strikeout_rate = ('play_result', lambda x: x.isin({"Strikeout", "Strikeout Double Play"}).mean()),
        season_pitcher_pa_walk_rate = ('play_result', lambda x: x.isin({"Walk", "Intent Walk"}).mean()),
        season_pitcher_pa_hbp_rate = ('play_result', lambda x: x.eq("Hit By Pitch").mean()),
        season_pitcher_pa_hit_rate = ('play_result', lambda x: x.isin({"Single", "Double", "Triple", "Home Run"}).mean()),
        season_pitcher_pa_hr_rate = ('play_result', lambda x: x.eq("Home Run").mean()),
        season_pitcher_pa_single_rate = ('play_result', lambda x: x.eq("Single").mean()),
        season_pitcher_pa_xbh_rate = ('play_result', lambda x: x.isin({"Double", "Triple", "Home Run"}).mean()),

        # FIP components — compute FIP as derived column after
        season_pitcher_pa_fip_k = ('play_result', lambda x: x.isin({"Strikeout", "Strikeout Double Play"}).sum()),
        season_pitcher_pa_fip_bb = ('play_result', lambda x: x.isin({"Walk", "Hit By Pitch"}).sum()),
        season_pitcher_pa_fip_hr = ('play_result', lambda x: x.eq("Home Run").sum()),

        # count leverage — what count did PAs end on
        season_pitcher_pa_avg_final_balls = ('count_balls', 'mean'),
        season_pitcher_pa_avg_final_strikes = ('count_strikes', 'mean'),
        season_pitcher_pa_full_count_rate = ('count_balls', lambda x: ((x == 3) & (last_pitch_pbp.loc[x.index, 'count_strikes'] == 2)).mean()),
    )
    .reset_index()
)

# FIP as derived column after agg
C = 3.10  # league average FIP constant, adjusts yearly
season_last_pitch_pbp_sp_pitcher_stats['season_pitcher_pa_fip'] = (
    (13 * season_last_pitch_pbp_sp_pitcher_stats['season_pitcher_pa_fip_hr'] +
      3 * season_last_pitch_pbp_sp_pitcher_stats['season_pitcher_pa_fip_bb'] -
      2 * season_last_pitch_pbp_sp_pitcher_stats['season_pitcher_pa_fip_k']) /
     season_last_pitch_pbp_sp_pitcher_stats['season_pitcher_pa_total'] + C
)


season_last_inning_sp_pitcher_stats = (
    pbp_sp_pitcher.loc[pbp_sp_pitcher.groupby(['pitcher_id', 'gamepk'])['play_id'].idxmax()]
    .groupby(['pitcher_id', 'game_season'])
    .agg(
        # how deep did he go
        season_pitcher_avg_last_inning = ('inning', 'mean'), # what inning was he pulled
        season_pitcher_std_last_inning = ('inning', 'std'), # what inning was he pulled

        # velocity at exit
        season_pitcher_avg_last_inning_velo = ('start_speed', 'mean'),
        
        # command at exit
        season_pitcher_last_inning_ball_rate = ('is_ball', 'mean'),
        season_pitcher_last_inning_strike_rate = ('is_strike', 'mean'),

        # count state when pulled — was he in trouble?
        season_pitcher_last_inning_avg_balls = ('count_balls', 'mean'),
        season_pitcher_last_inning_avg_strikes = ('count_strikes', 'mean'),
        season_pitcher_last_inning_outs = ('count_outs', 'max'),  # how many outs were recorded
    )
    .reset_index()
)

# game pitch totals 
game_sp_pitch_counts = (
    pbp_sp_pitcher
    .groupby(['pitcher_id', 'gamepk', 'game_season'])['is_pitch']
    .sum()
    .reset_index()
    .rename(columns={'is_pitch':'total_pitches'})
)

season_sp_pitch_counts = (
    game_sp_pitch_counts
    .groupby(['pitcher_id', 'game_season'])
    .agg(
       season_pitcher_avg_pitch_count = ('total_pitches', 'mean'),
       season_pitcher_std_pitch_count = ('total_pitches', 'std'),
       season_pitcher_max_pitch_count = ('total_pitches', 'max')
    )
    .reset_index()
) 


# contact quality: 
sp_contact_only = pbp_sp_pitcher[pbp_sp_pitcher.is_in_play == True]

season_sp_contact_quality = (
    sp_contact_only
    .groupby(['pitcher_id', 'game_season'])
    .agg(
        season_pitcher_contact_hard_hit_rate = ('hardness', lambda x: x.eq('Hard').mean()),
        season_pitcher_contact_gb_rate = ('trajectory', lambda x: x.eq('Ground Ball').mean()),
        season_pitcher_contact_fb_rate = ('trajectory', lambda x: x.eq('Fly Ball').mean()),
        season_pitcher_contact_ld_rate = ('trajectory', lambda x: x.eq('Line Drive').mean()),
        season_pitcher_contact_avg_launch_speed = ('launch_speed', 'mean'),
        season_pitcher_contact_avg_launch_angle = ('launch_angle', 'mean'),
    )
)


# merge & process pbp sp stats into one df
season_sp_pbp_stats_combined = (
    season_sp_pitcher_pbp_stats
    .merge(season_last_pitch_pbp_sp_pitcher_stats, on=['pitcher_id','game_season'], how='left')
    .merge(season_last_inning_sp_pitcher_stats, on=['pitcher_id','game_season'], how='left')
    .merge(season_sp_pitch_counts,  on=['pitcher_id','game_season'], how='left')
    .merge(season_sp_contact_quality,  on=['pitcher_id','game_season'], how='left')
)


season_sp_pbp_stats_combined = (
    season_sp_pbp_stats_combined
    .assign(
        game_season = lambda x: x['game_season']+1
    )
    .rename(columns={
        col: col.replace('season_', 'last_season_pitcher_')
        for col in season_sp_pbp_stats_combined.columns 
        if col.startswith('season_')
    })
)



# so we can categorize by pitcher + pitch type  and come up with some kidn of summary stats 





# ---------------------------------------------------------------------------- #
#                                  TRAIN MODEL                                 #
# ---------------------------------------------------------------------------- #

# -------------------------------- CREATE DATA ------------------------------- #
model_df = pd.merge(pa_outcome, last_season_batter_stats, 
                    left_on=['game_season', 'batter_id'],
                    right_on=['game_season', 'personId'],
                    how='left')

model_df = pd.merge(model_df, last_season_pitcher_stats, 
                    left_on=['game_season', 'pitcher_id'],
                    right_on=['game_season', 'personId'],
                    how='left')

model_df = pd.merge(model_df, season_sp_pbp_stats_combined,
                    on=['game_season', 'pitcher_id'],
                    how='left')


# null check
new_cols = season_sp_pbp_stats_combined.columns.difference(['game_season', 'pitcher_id']).tolist()

null_summary = (
    model_df[new_cols]
    .isnull()
    .sum()
    .rename('null_count')
    .to_frame()
    .assign(null_rate = lambda x: x['null_count'] / len(model_df))
    .sort_values('null_rate', ascending=False)
)

print(null_summary.head(20))


model_df[model_df['last_season_pitcher_pitcher_std_last_inning'].isnull()]['game_season'].value_counts()


# for now pitchers that weren't starters in previous seasons get median filled
season_sp_pbp_stats_combined[season_sp_pbp_stats_combined['pitcher_id']=='681911']
pbp[pbp['pitcher_id']=='681911']['game_season'].value_counts()


# ------------------------------ DEFINE FEATURES ----------------------------- #
DATE_FEATS = ['game_season']


context_num_feats = ['weather_temp', 'batting_order']

last_season_pitcher_num_feats = [col for col in model_df.columns if col.startswith('last_season_pitcher')]
last_season_batter_num_feats = [col for col in model_df.columns if col.startswith('last_season_batter')]

# bin into larger num or cat feats 
NUM_FEATS = (
    context_num_feats + 
    last_season_pitcher_num_feats + 
    last_season_batter_num_feats
)


# -------------------------------- SPLIT DATA -------------------------------- #
# push this above later on

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()

X_train = train_df[NUM_FEATS + DATE_FEATS].copy()
y_train = train_df[TARGET].copy()
X_val   = val_df[NUM_FEATS + DATE_FEATS].copy()
y_val   = val_df[TARGET].copy()


# -------------------------------- IMPUTE FEATS ------------------------------- #

# start with simple impute median

train_num_median = X_train[NUM_FEATS].median()
X_train[NUM_FEATS] = X_train[NUM_FEATS].fillna(train_num_median)
X_val[NUM_FEATS] = X_val[NUM_FEATS].fillna(train_num_median)


# ------------------------------ SCALE NUM FEATS ----------------------------- #
scaler    = StandardScaler()
X_train[NUM_FEATS] = scaler.fit_transform(X_train[NUM_FEATS])
X_val[NUM_FEATS]  = scaler.transform(X_val[NUM_FEATS])



# ------------------------------ TRAIN MODELS -------------------------------- #
def get_calibration_df(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))

    rows = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        rows.append(_make_bucket_row(
            label=f"{lo:.3f}–{hi:.3f}",
            y_true=y_true[mask],
            y_prob=y_prob[mask],
        ))

    return pd.DataFrame(rows)

def plot_calibration_curve(y_true, results: dict, n_bins=10):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.15, 0.35], [0.15, 0.35], 'k--', label='Perfect calibration')

    for name, r in results.items():
        cal_df = get_calibration_df(y_true, r["proba"], n_bins=n_bins)
        ax.plot(cal_df["mean_pred"], cal_df["obs_hit_rate"], marker='o', label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed hit rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()
    plt.show()

def evaluate_hit_predictor(y_true, y_prob, baseline_prob=None, base_rate=0.22, n_bins=10, min_n=500):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    calibration_df = get_calibration_df(y_true, y_prob, n_bins=n_bins)

    print("=" * 75)
    print("CALIBRATION TABLE")
    print("=" * 75)
    print(calibration_df.to_string(index=False))

    # split into deciles, compare actual hit rate of top vs bottom 10%
    # large spread = model has real signal, small spread = model isn't separating well
    print("\n" + "=" * 75)
    print("DISCRIMINATION METRICS")
    print("=" * 75)
    p10, p90    = np.percentile(y_prob, [10, 90])
    bottom_mask = y_prob <= p10
    top_mask    = y_prob >= p90
    bottom_rate = y_true[bottom_mask].mean()
    top_rate    = y_true[top_mask].mean()
    print(f"  Decile cutoffs         : bottom {p10:.3f} / top {p90:.3f}")
    print(f"  Top-decile hit rate    : {top_rate:.3f}  (N={top_mask.sum()})")
    print(f"  Bottom-decile hit rate : {bottom_rate:.3f}  (N={bottom_mask.sum()})")
    print(f"  Spread                 : {top_rate - bottom_rate:.3f}")

    # brier score is MSE between predicted prob and actual outcome — lower is better
    # positive delta = baseline has higher error = model wins
    print("\n" + "=" * 75)
    print("BRIER SCORES")
    print("=" * 75)
    model_brier = brier_score_loss(y_true, y_prob)
    print(f"  Model    : {model_brier:.4f}")

    if baseline_prob is not None:
        baseline_prob  = np.asarray(baseline_prob)
        baseline_brier = brier_score_loss(y_true, baseline_prob)
        delta          = baseline_brier - model_brier
        direction      = "better" if delta > 0 else "worse"
        print(f"  Baseline : {baseline_brier:.4f}")
        print(f"  Delta    : {abs(delta):.4f} (model is {direction} than baseline)")


def _make_bucket_row(label, y_true, y_prob):
    return {
        "bucket":       label,
        "n":            len(y_true),
        # mean predicted prob — should track obs_hit_rate if well calibrated
        "mean_pred":    round(y_prob.mean(), 3),
        # actual hit rate for rows in this bin
        "obs_hit_rate": round(y_true.mean(), 3),
    }

models = {
    "Logistic regression": (LogisticRegression(max_iter=1000, random_state=42), X_train, X_val),
    "XGBoost": (xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss"), X_train, X_val),
}

results = {}
for name, (model, X_train, X_val) in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]
    results[name] = {
        "log_loss": log_loss(y_val, prob),
        "brier":  brier_score_loss(y_val, prob),
        "proba":    prob,
    }

    

evaluate_hit_predictor(y_true=y_val, y_prob=results['Logistic regression']['proba'])
evaluate_hit_predictor(y_true=y_val, y_prob=results['XGBoost']['proba'])



pd.Series(results['XGBoost']['proba']).describe()

# - CHECK BASELINE IF WE'RE FILTERING JUST FOR THOSE WITH BATTING ORDER 

# create plots:
# - top feature importance
# - look at top scores


%matplotlib inline
plot_calibration_curve(
    y_val,
    results
)


# FIGURE OUT TOP FEATURES + RETRAIN 



# THESE FEATURES SURVIVE INTO NEXT ROUND -----------------------------------



# add play by play pitcher season stats
# add rolling stats - in a controlled way