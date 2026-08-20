import os
import pandas as pd
import statsapi
from math import ceil
from time import sleep
from datetime import datetime, timedelta
from typing import Optional, List, Union, Tuple
import boto3
from botocore.exceptions import ClientError
import tempfile
import logging
import requests

cloudwatch = boto3.client('cloudwatch', region_name='us-east-2')
logger = logging.getLogger(__name__)

def _upload_to_s3(df, s3_key, s3_prefix=''):

    aws_bucket = 'mlbdk'
    aws_region = 'us-east-2'

    if s3_prefix:
        s3_key = f"{s3_prefix.rstrip('/')}/{s3_key}"

    s3_client = boto3.client('s3', region_name=aws_region)
    try:
        # Use temporary file for parquet upload
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            df.to_parquet(temp_file.name, index=False)
            s3_client.upload_file(temp_file.name, aws_bucket, s3_key)
            # Clean up temp file
            os.unlink(temp_file.name)
    
    except ClientError as e:
        error_msg = f"Error uploading to S3: {e}"
        raise

def fetch_schedule_df(game_date: str,
                      sleep_time: Optional[float] = 1.1,
                      save_to_s3: bool = True,
                      s3_prefix: str = '') -> Tuple[pd.DataFrame, list]:
    """
    Fetch MLB schedule data for a single date
    
    Args:
        date (str): Date in 'YYYY-MM-DD' format
        sleep_time (float): Seconds to wait (mainly for consistency with other functions)
        save_to_s3 (bool): Whether to save to S3
    
    Returns:
        tuple: (DataFrame, list of errors)
    """
    
    def process_schedule_dataframe(schedule_list):
        """Clean and process the combined schedule DataFrame"""
        if not schedule_list:
            return pd.DataFrame()
        
        final_df = pd.concat(schedule_list, ignore_index=True)
        
        # Early return if empty
        if final_df.empty:
            return final_df
        
        # Integer/Float fields
        numeric_cols = [
            'game_num',        # 1.0
            'away_score',      # 7.0
            'home_score',      # 3.0
            'venue_id'         # 15 (convert to int if it's actually numeric)
        ]
        
        for col in numeric_cols:
            if col in final_df.columns:
                if col == 'venue_id':
                    # venue_id might be better as int if it's numeric
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0).astype('Int64')
                else:
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        # String fields (keep as strings but ensure consistency)
        str_cols = [
            'game_id',                    # 2025/06/01/detmlb-kcamlb-1
            'game_datetime',              # 2025-06-01T19:10:00Z
            'game_date',                  # 2025-06-01
            'game_type',                  # R
            'status',                     # Final
            'away_name',                  # Detroit Tigers
            'home_name',                  # Kansas City Royals
            'away_id',                    # 116
            'home_id',                    # 118
            'doubleheader',               # N
            'home_probable_pitcher',      # Kris Bubic
            'away_probable_pitcher',      # Keider Montero
            'home_pitcher_note',          # 
            'away_pitcher_note',          # 
            'current_inning',             # Final
            'inning_state',               # 
            'venue_name',                 # Kauffman Stadium
            'series_status',              # 
            'winning_team',               # Detroit Tigers
            'losing_team',                # Kansas City Royals
            'winning_pitcher',            # Keider Montero
            'losing_pitcher',             # Kris Bubic
            'save_pitcher',               # Jason Foley
            'summary',                    # Game summary text
            'data_fetched_at'           # 2025-01-15T10:30:00
        ]
        
        for col in str_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str).replace('nan', '')
        
        # Handle the national_broadcasts object column
        if 'national_broadcasts' in final_df.columns:
            # Convert to string representation if it's a complex object
            final_df['national_broadcasts'] = final_df['national_broadcasts'].astype(str)
        
        # Add metadata
        final_df['data_fetched_at'] = datetime.now().isoformat()
        
        # Sort by game date and game_id for consistency
        if 'game_date' in final_df.columns:
            final_df = final_df.sort_values(['game_date', 'game_id']).reset_index(drop=True)
        
        return final_df
    
    complete_schedule_list = []
    error_list = []
    
    try:
        logger.info(f"Fetching schedule for {game_date}")
        schedule_data = statsapi.schedule(date=game_date)
        
        if schedule_data:
            schedule_df = pd.DataFrame(schedule_data)
            complete_schedule_list.append(schedule_df)
        else:
            logger.info(f"No games found for {game_date}")
            
    except Exception as e:
        error_list.append(game_date)
        error_msg = f"Failed to get schedule for {game_date}: {e}"
        logger.error(error_msg)

    # Optional sleep (mainly for consistency with other functions)
    if sleep_time:
        sleep(sleep_time)
    
    # Process and save final data
    if complete_schedule_list:
        final_df = process_schedule_dataframe(complete_schedule_list)
        
        # Save final file
        if save_to_s3:
            year_folder = datetime.strptime(game_date, "%Y-%m-%d").year

            s3_key = f"raw_data/games/schedule/{year_folder}/{game_date}.parquet"
            _upload_to_s3(final_df, s3_key, s3_prefix=s3_prefix)
        
        return final_df, error_list
    else:
        logger.warning("No schedule data was successfully fetched.")
        return pd.DataFrame(), error_list
    
def fetch_game_info(gamepks: list[int],
                    game_date: str,
                    sleep_time: Optional[float] = 1.1,
                    save_to_s3: bool = True,
                    year_folder: int = datetime.now().year,
                    s3_prefix: str = ''):

    def process_game_data(game_dict):

        rel_game_cols = ['game_type','game_doubleHeader','game_id','game_gamedayType','game_tiebreaker',
            'game_gameNumber','game_calendarEventID','game_season','game_seasonDisplay']
        rel_status_cols = ['status_statusCode', 'status_startTimeTBD']

        game = pd.json_normalize(game_dict['gameData']['game']).add_prefix('game_')
        game = game[rel_game_cols]

        status = pd.json_normalize(game_dict['gameData']['status']).add_prefix('status_')
        status = status[rel_status_cols]

        weather = pd.json_normalize(game_dict['gameData']['weather']).add_prefix('weather_')

        try:
            probable_pitchers_away = pd.json_normalize(game_dict['gameData']['probablePitchers']['away']).add_prefix('probable_pitcher_away_')
            probable_pitchers_away = probable_pitchers_away[['probable_pitcher_away_id', 'probable_pitcher_away_fullName']]
        except KeyError:
            probable_pitchers_away = pd.DataFrame({
                'probable_pitcher_away_id': [None], 
                'probable_pitcher_away_fullName': [None]
            })

        try:   
            probable_pitchers_home = pd.json_normalize(game_dict['gameData']['probablePitchers']['home']).add_prefix('probable_pitcher_home_')
            probable_pitchers_home = probable_pitchers_home[['probable_pitcher_home_id', 'probable_pitcher_home_fullName']]
        except KeyError:
            probable_pitchers_home = pd.DataFrame({
                'probable_pitcher_home_id': [None], 
                'probable_pitcher_home_fullName': [None]
            })
            
        game_df = pd.concat([game, status, weather, probable_pitchers_away, probable_pitchers_home], axis=1)
        game_df['game_duration_minutes'] = game_dict['gameData']['gameInfo']['gameDurationMinutes']
        game_df['gamePk'] = game_dict['gamePk']

        return game_df 
    
    def cast_game_info_types(game_df_combined):
        """Cast game info columns to proper types"""
        if game_df_combined.empty:
            return game_df_combined
        
        # Integer fields
        int_cols = [
            'game_gameNumber',           # 1
            'game_season',               # 2025
            'probable_pitcher_away_id',  # 672456
            'probable_pitcher_home_id',  # 663460
            'game_duration_minutes',     # 142
            'gamePk'                     # 777677
        ]
        
        for col in int_cols:
            if col in game_df_combined.columns:
                game_df_combined[col] = pd.to_numeric(game_df_combined[col], errors='coerce').fillna(0).astype('Int64')
        
        # String fields (keep as strings but ensure consistency)
        str_cols = [
            'game_type',                     # R
            'game_doubleHeader',             # N
            'game_id',                       # 2025/06/01/detmlb-kcamlb-1
            'game_gamedayType',              # P
            'game_tiebreaker',               # N
            'game_calendarEventID',          # 114-777677
            'game_seasonDisplay',            # 2025
            'status_statusCode',             # F
            'weather_condition',             # Partly Cloudy
            'weather_temp',                  # 84
            'weather_wind',                  # 3 mph, R To L
            'probable_pitcher_away_fullName', # Keider Montero
            'probable_pitcher_home_fullName'  # Kris Bubic
        ]
        
        for col in str_cols:
            if col in game_df_combined.columns:
                game_df_combined[col] = game_df_combined[col].astype(str)
        
        # Boolean fields
        bool_cols = ['status_startTimeTBD']  # False
        for col in bool_cols:
            if col in game_df_combined.columns:
                game_df_combined[col] = game_df_combined[col].astype(bool)
        
        return game_df_combined
    game_df_list = []
    gamepk_errors = []
    
    for gamepk in gamepks:
        try:
            game_dict = statsapi.get('game',{'gamePk':gamepk})
            game_df = process_game_data(game_dict)
            game_df_list.append(game_df)
            sleep(sleep_time)

        except (requests.RequestException, KeyError, ValueError) as e:
            gamepk_errors.append(gamepk)
            logger.error(f"Error fetching game {gamepk}: {e}")
            continue

    game_df_combined = pd.concat(game_df_list, ignore_index=True)
    game_df_combined = cast_game_info_types(game_df_combined)

    if save_to_s3:
        year_folder = datetime.strptime(game_date, "%Y-%m-%d").year

        s3_key = f"raw_data/games/game_info/{year_folder}/{game_date}.parquet"
        _upload_to_s3(game_df_combined, s3_key, s3_prefix=s3_prefix)

    return game_df_combined, gamepk_errors

def fetch_batter_pitcher_boxscore(gamepks: list[int],
                                game_date: str,
                                sleep_time: Optional[float] = 1.1,
                                save_to_s3=True,
                                s3_prefix: str = ''):
    
    def process_players_boxscore(boxscore_dict):

        results = {'batters':[], 'pitchers':[]}

        for player_type in ['Batters','Pitchers']:
            for side in ['home', 'away']:
                players = pd.DataFrame(boxscore_dict[f'{side}{player_type}'][1:])

                players['team_id'] = boxscore_dict['teamInfo'][f'{side}']['id']
                players['team_name'] = boxscore_dict['teamInfo'][f'{side}']['teamName']

                results[player_type.lower()].append(players)

            
        batters_df = pd.concat(results['batters'], ignore_index=True)
        pitchers_df = pd.concat(results['pitchers'], ignore_index=True)

        return batters_df, pitchers_df

    def cast_batter_boxscore_types(batter_df_complete):
        """Cast batter boxscore columns to proper types"""
        if batter_df_complete.empty:
            return batter_df_complete
        
        # Integer fields
        int_cols = [
            'ab',              # 4 (at bats)
            'r',               # 0 (runs)
            'h',               # 0 (hits)
            'doubles',         # 0 (doubles)
            'triples',         # 0 (triples)
            'hr',              # 0 (home runs)
            'rbi',             # 0 (RBIs)
            'bb',              # 0 (walks)
            'k',               # 1 (strikeouts)
            'lob',             # 0 (left on base)
            'personId',        # 663697 (player ID)
            'battingOrder',    # 100 (batting order)
            'team_id',         # 118 (team ID)
            'gamePk'           # 777677 (game ID)
        ]
        
        for col in int_cols:
            if col in batter_df_complete.columns:
                batter_df_complete[col] = pd.to_numeric(batter_df_complete[col], errors='coerce').fillna(0).astype('Int64')
        
        # Float fields (batting stats)
        float_cols = [
            'avg',             # .235 (batting average)
            'ops',             # .628 (on-base plus slugging)
            'obp',             # .328 (on-base percentage)
            'slg'              # .300 (slugging percentage)
        ]
        
        for col in float_cols:
            if col in batter_df_complete.columns:
                batter_df_complete[col] = pd.to_numeric(batter_df_complete[col], errors='coerce').fillna(0.0)
        
        # String fields
        str_cols = [
            'name',            # "1 India DH"
            'field',           # (empty in sample)
            'substitution',    # "False" 
            'note',            # "India"
            'nameposition',    # "DH"
            'team_name'        # "Royals"
        ]
        
        for col in str_cols:
            if col in batter_df_complete.columns:
                batter_df_complete[col] = batter_df_complete[col].astype(str)
        
        return batter_df_complete
    
    def cast_pitcher_boxscore_types(pitcher_df_complete):
        """Cast pitcher boxscore columns to proper types"""
        if pitcher_df_complete.empty:
            return pitcher_df_complete
        
        # Float fields (pitching stats)
        float_cols = [
            'ip',              # 7.0 (innings pitched)
            'h',               # 4 (hits allowed)
            'r',               # 1 (runs allowed)
            'er',              # 1 (earned runs)
            'bb',              # 2 (walks)
            'k',               # 9 (strikeouts)
            'hr',              # 0 (home runs allowed)
            'pc',              # 99 (pitch count - could be int, but sometimes has decimals)
            'era'              # 1.43 (earned run average)
        ]
        
        for col in float_cols:
            if col in pitcher_df_complete.columns:
                pitcher_df_complete[col] = pd.to_numeric(pitcher_df_complete[col], errors='coerce').fillna(0.0)
        
        # Integer fields
        int_cols = [
            'personId',        # 663460 (player ID)
            'team_id',         # 118 (team ID)
            'gamePk'           # 777677 (game ID)
        ]
        
        for col in int_cols:
            if col in pitcher_df_complete.columns:
                pitcher_df_complete[col] = pd.to_numeric(pitcher_df_complete[col], errors='coerce').fillna(0).astype('Int64')
        
        # String fields
        str_cols = [
            'name',            # "Bubic (L, 5-3)"
            'field',           # (empty in sample)
            'ps',              # 69 (could be strikes thrown - might want as int)
            'nameposition',    # "Bubic" 
            'note',            # "(L, 5-3)"
            'team_name'        # "Royals"
        ]
        
        for col in str_cols:
            if col in pitcher_df_complete.columns:
                pitcher_df_complete[col] = pitcher_df_complete[col].astype(str)
        
        return pitcher_df_complete

    batter_df_list = []
    pitcher_df_list = []
    gamepk_errors = []

    for gamepk in gamepks:
        try:
            boxscore_dict =  statsapi.boxscore_data(gamepk, timecode=None)
            batters_df, pitchers_df = process_players_boxscore(boxscore_dict)   
            
            batters_df['gamePk'] = gamepk 
            pitchers_df['gamePk'] = gamepk 

            batter_df_list.append(batters_df)
            pitcher_df_list.append(pitchers_df)

            sleep(sleep_time)

        except (requests.RequestException, KeyError, ValueError) as e:
            gamepk_errors.append(gamepk)
            logger.error(f"Error fetching game {gamepk}: {e}")
            continue

    batter_df_complete = pd.concat(batter_df_list, ignore_index=True)
    pitcher_df_complete = pd.concat(pitcher_df_list, ignore_index=True)

    batter_df_complete = cast_batter_boxscore_types(batter_df_complete)
    pitcher_df_complete = cast_pitcher_boxscore_types(pitcher_df_complete)

    if save_to_s3:
        year_folder = datetime.strptime(game_date, "%Y-%m-%d").year

        s3_batter_key = f"raw_data/games/batter_boxscore/{year_folder}/{game_date}.parquet"
        s3_pitcher_key = f"raw_data/games/pitcher_boxscore/{year_folder}/{game_date}.parquet"

        _upload_to_s3(batter_df_complete, s3_batter_key, s3_prefix=s3_prefix)
        _upload_to_s3(pitcher_df_complete, s3_pitcher_key, s3_prefix=s3_prefix)

    return batter_df_complete, pitcher_df_complete, gamepk_errors

def fetch_playbyplay_data(gamepks: List,
                            game_date: str = None,
                            sleep_time: Optional[float] = 1.1,
                            save_to_s3: bool = True,
                            s3_prefix: str = ''):
    """
    Fetch Statcast data for a list of game PKs
    
    Args:
        gamepks: List of game primary keys
        sleep_time (float): Seconds to wait after API call
        save_to_s3 (bool): Whether to save final results to S3
    
    Returns:
        tuple: (DataFrame, list of errors)
    """
    
    def parse_plays_to_dataframe(all_plays):
        rows = []
        
        for play in all_plays:
            # Get the main play info
            result = play.get('result', {})
            about = play.get('about', {})
            count = play.get('count', {})
            matchup = play.get('matchup', {})
            play_events = play.get('playEvents', [])
            
            # Extract each pitch/event
            for event in play_events:
                if event.get('type') != 'pitch' or not event.get('isPitch'):
                    continue  # Skip non-pitch events
                
                # Extract game date from startTime
                start_time = about.get('startTime')
                if start_time:
                    # Parse ISO timestamp and extract date
                    game_date = start_time.split('T')[0] 
                else:
                    game_date = None
                    
                row = {
                    # Play-level info
                    'play_id': about.get('atBatIndex'),
                    'inning': about.get('inning'),
                    'half_inning': about.get('halfInning'),
                    'batter_id': matchup.get('batter', {}).get('id'),
                    'batter_name': matchup.get('batter', {}).get('fullName'),
                    'pitcher_id': matchup.get('pitcher', {}).get('id'),
                    'pitcher_name': matchup.get('pitcher', {}).get('fullName'),
                    'play_result': result.get('event'),
                    'play_description': result.get('description'),
                    'game_date': game_date,
                    
                    # Event-level info
                    'event_index': event.get('index'),
                    'event_type': event.get('type'),
                    'is_pitch': event.get('isPitch'),
                    'pitch_number': event.get('pitchNumber'),
                    
                    # Pitch details
                    'pitch_call': event.get('details', {}).get('call', {}).get('description'),
                    'pitch_call_code': event.get('details', {}).get('call', {}).get('code'),
                    'pitch_type': event.get('details', {}).get('type', {}).get('description'),
                    'pitch_type_code': event.get('details', {}).get('type', {}).get('code'),
                    'is_in_play': event.get('details', {}).get('isInPlay'),
                    'is_strike': event.get('details', {}).get('isStrike'),
                    'is_ball': event.get('details', {}).get('isBall'),
                    'is_out': event.get('details', {}).get('isOut'),
                    'count_balls': event.get('count', {}).get('balls'),
                    'count_strikes': event.get('count', {}).get('strikes'),
                    'count_outs': event.get('count', {}).get('outs'),
                }
                
                # Add pitch data if available
                if event.get('pitchData'):
                    pitch_data = event['pitchData']
                    coordinates = pitch_data.get('coordinates', {})
                    breaks = pitch_data.get('breaks', {})
                    
                    row.update({
                        # Basic pitch metrics
                        'start_speed': pitch_data.get('startSpeed'),
                        'end_speed': pitch_data.get('endSpeed'),
                        'zone': pitch_data.get('zone'),
                        'type_confidence': pitch_data.get('typeConfidence'),
                        'plate_time': pitch_data.get('plateTime'),
                        'extension': pitch_data.get('extension'),
                        'strike_zone_top': pitch_data.get('strikeZoneTop'),
                        'strike_zone_bottom': pitch_data.get('strikeZoneBottom'),
                        
                        # Coordinates
                        'plate_x': coordinates.get('pX'),
                        'plate_z': coordinates.get('pZ'),
                        'release_pos_x': coordinates.get('x0'),
                        'release_pos_y': coordinates.get('y0'),
                        'release_pos_z': coordinates.get('z0'),
                        'pfx_x': coordinates.get('pfxX'),
                        'pfx_z': coordinates.get('pfxZ'),
                        'vx0': coordinates.get('vX0'),
                        'vy0': coordinates.get('vY0'),
                        'vz0': coordinates.get('vZ0'),
                        'ax': coordinates.get('aX'),
                        'ay': coordinates.get('aY'),
                        'az': coordinates.get('aZ'),
                        'sz_x': coordinates.get('x'),
                        'sz_y': coordinates.get('y'),
                        
                        # Breaks
                        'spin_rate': breaks.get('spinRate'),
                        'spin_direction': breaks.get('spinDirection'),
                        'break_angle': breaks.get('breakAngle'),
                        'break_length': breaks.get('breakLength'),
                        'break_y': breaks.get('breakY'),
                        'break_vertical': breaks.get('breakVertical'),
                        'break_vertical_induced': breaks.get('breakVerticalInduced'),
                        'break_horizontal': breaks.get('breakHorizontal'),
                    })
                
                # Add hit data if available
                if event.get('hitData'):
                    hit_data = event['hitData']
                    hit_coords = hit_data.get('coordinates', {})
                    row.update({
                        'launch_speed': hit_data.get('launchSpeed'),
                        'launch_angle': hit_data.get('launchAngle'),
                        'total_distance': hit_data.get('totalDistance'),
                        'trajectory': hit_data.get('trajectory'),
                        'hardness': hit_data.get('hardness'),
                        'hit_location': hit_data.get('location'),
                        'hit_coord_x': hit_coords.get('coordX'),
                        'hit_coord_y': hit_coords.get('coordY')
                    })
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def cast_playbyplay_types(complete_df):
        """Cast play-by-play columns to proper types"""
        if complete_df.empty:
            return complete_df
        
        # Integer fields
        int_cols = [
            'play_id',              # 25 (at bat index)
            'inning',               # 1 (inning number)
            'batter_id',            # 670231 (batter player ID)
            'pitcher_id',           # 676684 (pitcher player ID)
            'event_index',          # 6 (event index)
            'pitch_number',         # 5 (pitch number in at-bat)
            'count_balls',          # 1 (ball count)
            'count_strikes',        # 2 (strike count)
            'count_outs',           # 0 (out count)
            'zone',                 # 9 (strike zone location)
            'spin_rate',            # 2282 (revolutions per minute - integer)
            'total_distance',       # 17 (feet)
            'hit_location',         # 1 (field location number)
            'gamepk'               # 777677 (game ID)
        ]
        
        for col in int_cols:
            if col in complete_df.columns:
                complete_df[col] = pd.to_numeric(complete_df[col], errors='coerce').fillna(0).astype('Int64')
        
        # Float fields (pitch physics and hit data)
        float_cols = [
            'start_speed',           # 97.9 (pitch velocity)
            'end_speed',             # 90.1 (velocity at plate)
            'type_confidence',       # 0.95 (pitch type confidence)
            'plate_time',            # 0.383 (time to plate)
            'extension',             # 6.379 (release extension)
            'strike_zone_top',       # 3.319 (top of strike zone)
            'strike_zone_bottom',    # 1.513 (bottom of strike zone)
            'plate_x',               # -0.075 (horizontal location at plate)
            'plate_z',               # 2.576 (vertical location at plate)
            'release_pos_x',         # -1.733 (release position x)
            'release_pos_y',         # 50.002 (release position y)
            'release_pos_z',         # 5.530 (release position z)
            'pfx_x',                 # -3.079 (horizontal movement)
            'pfx_z',                 # 8.228 (vertical movement)
            'vx0', 'vy0', 'vz0',     # Initial velocity components
            'ax', 'ay', 'az',        # Acceleration components
            'sz_x', 'sz_y',          # Strike zone coordinates
            'spin_direction',        # 169.2 (spin direction in degrees)
            'break_angle',           # 1.2 (break angle)
            'break_length',          # 2.8 (break length)
            'break_y',               # -14.4 (break y)
            'break_vertical',        # 14.0 (vertical break)
            'break_vertical_induced', # 4.6 (induced vertical break)
            'break_horizontal',      # 4.6 (horizontal break)
            'launch_speed',          # 84.6 (exit velocity)
            'launch_angle',          # -8.0 (launch angle)
            'hit_coord_x',           # 126.48 (hit coordinate x)
            'hit_coord_y'            # 178.54 (hit coordinate y)
        ]
        
        for col in float_cols:
            if col in complete_df.columns:
                complete_df[col] = pd.to_numeric(complete_df[col], errors='coerce')
        
        # Boolean fields
        bool_cols = [
            'is_pitch',              # True
            'is_in_play',            # True
            'is_strike',             # False
            'is_ball',               # False
            'is_out'                 # True
        ]
        
        for col in bool_cols:
            if col in complete_df.columns:
                complete_df[col] = complete_df[col].astype(bool)
        
        # String fields
        str_cols = [
            'half_inning',           # "bottom"
            'batter_name',           # "John Rave"
            'pitcher_name',          # "Will Vest"
            'play_result',           # "Groundout"
            'play_description',      # "John Rave grounds out, pitcher Will Vest to fi..."
            'event_type',            # "pitch"
            'pitch_call',            # "In play, out(s)"
            'pitch_call_code',       # "X"
            'pitch_type',            # "Four-Seam Fastball"
            'pitch_type_code',       # "FF"
            'trajectory',            # "ground_ball"
            'hardness',              # "medium"
            'game_date'              # "2024-09-21" (added for grouping)
        ]
        
        for col in str_cols:
            if col in complete_df.columns:
                complete_df[col] = complete_df[col].astype(str)
        
        return complete_df
    
    gamepk_error_list = []
    playbyplay_list = []
    
    for gamepk in gamepks:
        try: 
            url = f"https://statsapi.mlb.com/api/v1/game/{gamepk}/playByPlay"
            response = requests.get(url)
            response = response.json()
            
            playbyplay_df = parse_plays_to_dataframe(response['allPlays'])
            playbyplay_df['gamepk'] = gamepk
            
            playbyplay_list.append(playbyplay_df)
            sleep(sleep_time)
            
        except Exception as e:
            gamepk_error_list.append(gamepk)
            error_msg = f"Failed to get play-by-play data for {gamepk}: {e}"
            logger.error(error_msg)

    if playbyplay_list:
        complete_df = pd.concat(playbyplay_list, ignore_index=True)
        complete_df = cast_playbyplay_types(complete_df)
        
        if save_to_s3:
            s3_date = game_date if game_date else complete_df['game_date'].iloc[0]
            year_folder = datetime.strptime(s3_date, "%Y-%m-%d").year
            s3_key = f"raw_data/playbyplay/{year_folder}/{s3_date}.parquet"
            _upload_to_s3(complete_df, s3_key, s3_prefix=s3_prefix)
        
        return complete_df, gamepk_error_list
    else:
        logger.warning("No play-by-play data was successfully fetched.")
        return pd.DataFrame(), gamepk_error_list

def send_metrics_to_cloudwatch(results, date):
    """Send custom metrics to CloudWatch for dashboard visualization"""
    metrics = []
    
    # Schedule metrics
    if 'schedule' in results:
        metrics.extend([
            {
                'MetricName': 'ScheduleGamesFound',
                'Value': results['schedule']['games_found'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            },
            {
                'MetricName': 'ScheduleErrors',
                'Value': results['schedule']['errors'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            }
        ])
    
    # Game info metrics
    if 'game_info' in results:
        metrics.extend([
            {
                'MetricName': 'GameInfoProcessed',
                'Value': results['game_info']['games_processed'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            },
            {
                'MetricName': 'GameInfoErrors',
                'Value': results['game_info']['errors'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            }
        ])
    
    # Boxscore metrics
    if 'boxscore' in results:
        metrics.extend([
            {
                'MetricName': 'BattersBoxscoreProcessed',
                'Value': results['boxscore']['batters_boxscore_processed'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            },
            {
                'MetricName': 'PitchersBoxscoreProcessed',
                'Value': results['boxscore']['pitchers_boxscore_processed'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            },
            {
                'MetricName': 'BoxscoreErrors',
                'Value': results['boxscore']['errors'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            }
        ])
    
    # Play-by-play metrics
    if 'playbyplay' in results:
        metrics.extend([
            {
                'MetricName': 'PlayByPlayProcessed',
                'Value': results['playbyplay']['games_processed'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            },
            {
                'MetricName': 'PlayByPlayErrors',
                'Value': results['playbyplay']['errors'],
                'Unit': 'Count',
                'Dimensions': [{'Name': 'ProcessDate', 'Value': date}]
            }
        ])
    
    # Send metrics in batches (CloudWatch allows max 20 per call)
    for i in range(0, len(metrics), 20):
        batch = metrics[i:i+20]
        try:
            cloudwatch.put_metric_data(
                Namespace='mlb/DataProcessing',
                MetricData=batch
            )
        except ClientError as e:
            error_msg = f"Failed to send metrics batch: {e}"
            logger.error(error_msg)



