import awswrangler as wr
import pandas as pd
import boto3
import numpy as np
from datetime import date
import pyarrow as pa
from datetime import datetime 

class GetData:
    
    def __init__(self) -> None:
        self.bucket_name = 'mlbdk'
        s3 = boto3.resource('s3')
        self.mlb_bucket = s3.Bucket(self.bucket_name)

    def get_game_info(self, season=None, date=None):
        """
        Get game info data
        Args:
            season: int, get specific season
            date: str (YYYY-MM-DD), get specific date
            If neither provided, returns current season
        """
        if date:
            date_obj = pd.to_datetime(date)
            year = date_obj.year
            formatted_date = date_obj.strftime('%Y-%m-%d')
            path = f's3://{self.bucket_name}/processed_data/games/game_info/{year}/{formatted_date}.parquet'
        elif season:
            path = f's3://{self.bucket_name}/processed_data/games/game_info/{season}/'
        else:
            path = f's3://{self.bucket_name}/processed_data/games/game_info/'
        
        game_info = wr.s3.read_parquet(path=path)

        return game_info

    def get_batter_boxscore(self, season=None, date=None):
        """
        Get batter boxscore data
        Args:
            season: int, get specific season
            date: str (YYYY-MM-DD), get specific date
            If neither provided, returns current season
        """
        if date:
            date_obj = pd.to_datetime(date)
            year = date_obj.year
            formatted_date = date_obj.strftime('%Y-%m-%d')
            path = f's3://{self.bucket_name}/processed_data/prepared/batter_boxscore/{year}/{formatted_date}.parquet'

        elif season:
            year = season
            path = f's3://{self.bucket_name}/processed_data/prepared/batter_boxscore/{year}/'

        else:
            path = f's3://{self.bucket_name}/processed_data/prepared/batter_boxscore/'

        batter_boxscore = wr.s3.read_parquet(path=path)

        return batter_boxscore

    def get_pitcher_boxscore(self, season=None, date=None):
        """
        Get pitcher boxscore data
        Args:
            season: int, get specific season
            date: str (YYYY-MM-DD), get specific date
            If neither provided, returns current season
        """
        if date:
            date_obj = pd.to_datetime(date)
            year = date_obj.year
            formatted_date = date_obj.strftime('%Y-%m-%d')
            path = f's3://{self.bucket_name}/processed_data/prepared/pitcher_boxscore/{year}/{formatted_date}.parquet'

        elif season:
            path = f's3://{self.bucket_name}/processed_data/prepared/pitcher_boxscore/{season}/'

        else:
            path = f's3://{self.bucket_name}/processed_data/prepared/pitcher_boxscore/'

        pitcher_boxscore = wr.s3.read_parquet(path=path)

        return pitcher_boxscore

    def get_playbyplay(self, season=None, date=None):
        """
        Get play-by-play data
        Args:
            season: int, get specific season
            date: str (YYYY-MM-DD), get specific date
            If neither provided, returns current season
        """
        if date:
            date_obj = pd.to_datetime(date)
            year = date_obj.year
            formatted_date = date_obj.strftime('%Y-%m-%d')
            path = f's3://{self.bucket_name}/processed_data/prepared/playbyplay/{year}/{formatted_date}.parquet'

        elif season:
            path = f's3://{self.bucket_name}/processed_data/prepared/playbyplay/{season}/'

        else:
            path = f's3://{self.bucket_name}/processed_data/prepared/playbyplay/'
            
        playbyplay_chunks = wr.s3.read_parquet(path=path, chunked=True)
        playbyplay_list = []

        for chunk in playbyplay_chunks:
            if 'spin_direction' in chunk.columns:
                chunk['spin_direction'] = chunk['spin_direction'].astype('float64')
            playbyplay_list.append(chunk)

        playbyplay = pd.concat(playbyplay_list, ignore_index=True)

        return playbyplay
    
    def get_schedule(self):

        schedule = wr.s3.read_parquet(path=f's3://{self.bucket_name}/processed_data/games/schedule/')
        numeric_columns = schedule.select_dtypes(include=['number']).columns
        schedule[numeric_columns] = schedule[numeric_columns].astype(np.float64)
        
        return schedule

    def get_player_info(self):
        
        player_info = wr.s3.read_parquet(path=f's3://{self.bucket_name}/raw_data/reference/player_info/')
        numeric_columns = player_info.select_dtypes(include=['number']).columns
        player_info[numeric_columns] = player_info[numeric_columns].astype(np.float64)
        
        return player_info
