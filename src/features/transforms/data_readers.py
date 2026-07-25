# data_readers.py
import io
import logging
from datetime import datetime, timedelta
import awswrangler as wr
import pandas as pd

logger = logging.getLogger()
BUCKET = 'mlbdk'
REGION = 'us-east-2'


def read_game_info(season: int) -> pd.DataFrame:
    return wr.s3.read_parquet(
        path=f"s3://{BUCKET}/processed_data/games/game_info/{season}/"
    )

def read_schedule(season: int) -> pd.DataFrame:
    return wr.s3.read_parquet(
        path=f"s3://{BUCKET}/processed_data/games/schedule/{season}/"
    )

def read_playbyplay(season: int) -> pd.DataFrame:
    return wr.s3.read_parquet(
        path=f"s3://{BUCKET}/processed_data/prepared/playbyplay/{season}/"
    )

def read_batter_boxscore(season: int) -> pd.DataFrame:
    return wr.s3.read_parquet(
        path=f"s3://{BUCKET}/processed_data/prepared/batter_boxscore/{season}/"
    )

def read_pitcher_boxscore(season: int) -> pd.DataFrame:
    return wr.s3.read_parquet(
        path=f"s3://{BUCKET}/processed_data/prepared/pitcher_boxscore/{season}/"
    )
