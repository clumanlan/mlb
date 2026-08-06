import pandas as pd 
from pathlib import Path

import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

from models.hit_predictor.processing.schema import PBP 

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

def _initial_pbp_processing(df, target_col):
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

def _add_pbp_pitch_state(df):

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

def _add_pbp_starting_pitcher(df):

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

def _add_pbp_pa(df):
        
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

def _add_pbp_game_date(df, schedule):
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

    pbp = _initial_pbp_processing(pbp, target_col)
    pbp = _add_pbp_pitch_state(pbp)   
    pbp = _add_pbp_starting_pitcher(pbp)
    pbp = _add_pbp_pa(pbp)
    pbp = _add_pbp_game_date(pbp, schedule)

    return pbp

def _convert_ip_to_decimal(ip: pd.Series) -> pd.Series:
    """Baseball IP format (6.2 = 6 2/3 innings) to decimal."""
    full_innings = np.floor(ip).astype('float64')
    partial = (ip - full_innings).round(1)

    bad = ~partial.isin([0.0, 0.1, 0.2])
    if bad.any():
        raise ValueError(f"Unexpected IP partial value(s): {ip[bad].tolist()}")

    return full_innings + (partial / 0.3)

def process_pitcher_boxscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(ip=lambda x: _convert_ip_to_decimal(x['ip']))

def _create_batting_order(batter_boxscore):

    df = (
        batter_boxscore[~batter_boxscore['batting_order'].isnull()]
        [["gamepk", "personId", "batting_order"]]
        .drop_duplicates(subset=["gamepk", "personId"])
        .rename(columns={"personId": "batter_id"})
    )

    return df.assign(batting_order = lambda x: x['batting_order'].astype(int))

def create_pa_outcome(pbp, batter_boxscore, game_info, schedule):

    batting_order = _create_batting_order(batter_boxscore)
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