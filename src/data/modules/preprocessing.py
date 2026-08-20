import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
# ---- Output column schemas (single source of truth for downstream and Cursor) ----
# Captured from: PYTHONPATH=. .mlb_venv/bin/python scripts/print_dataframe_columns.py
# Reduce to minimal set later when model and required features are fixed.
SCHEDULE_COLUMNS = [
    'gamepk', 'game_datetime', 'game_date', 'game_type', 'status', 'away_name', 'home_name',
    'away_id', 'home_id', 'game_num', 'home_probable_pitcher', 'away_probable_pitcher',
    'away_score', 'home_score', 'venue_id', 'venue_name', 'winning_team', 'losing_team',
    'winning_pitcher', 'losing_pitcher',
]

RELEVANT_GAME_TYPES = ['R', 'F', 'D', 'L', 'W']  # excludes S, E, A (spring training, exhibition, all-star game)


PITCHER_BOXSCORE_COLUMNS = [
    'personId', 'gamepk', 'team_id', 'ip', 'h', 'r', 'er', 'bb', 'k', 'hr', 'p', 's', 'outcome', 'fantasy_points', 
]

BATTER_BOXSCORE_COLUMNS = [
    'gamepk', 'batting_order', 'personId', 'team_id', 'ab', 'h', 'r', 'doubles', 'triples', 'hr',
    'rbi', 'sb', 'bb', 'k', 'lob', 'plate_appearances', 'singles', 'total_bases_from_h',
    'total_bases', 'fantasy_points',
]

PLAYER_INFO_COLUMNS = [
    'personId', 'player_name', 'weight', 'height_in_inches', 'birthDate',
    'strikeZoneTop', 'strikeZoneBottom', 'batSide', 'pitchHand',
]

GAME_INFO_COLUMNS = [
    'gamepk', 'game_season', 'weather_condition', 'weather_temp', 'weather_wind',
    'probable_pitcher_away_id', 'probable_pitcher_away_fullName',
    'probable_pitcher_home_id', 'probable_pitcher_home_fullName', 'game_duration_minutes',
]

PREPARE_BATTER_BOXSCORE_COLUMNS = BATTER_BOXSCORE_COLUMNS + [
    'player_name', 'game_type', 'game_date', 'game_datetime', 'game_season',
]


PREPARE_PITCHER_BOXSCORE_COLUMNS = PITCHER_BOXSCORE_COLUMNS + [
    'player_name', 'game_type', 'game_date', 'game_datetime', 'game_season'
]

# prepare_playbyplay: explicit output columns (raw play-by-play cols + joined names/teams). From scripts/print_dataframe_columns.py.
PLAYBYPLAY_COLUMNS = [
    'play_id', 'inning', 'half_inning', 'batter_id', 'batter_name', 'pitcher_id', 'pitcher_name',
    'play_result', 'play_description', 'event_index', 'event_type', 'is_pitch', 'pitch_number',
    'pitch_call', 'pitch_call_code', 'pitch_type', 'pitch_type_code', 'is_in_play', 'is_strike',
    'is_ball', 'is_out', 'count_balls', 'count_strikes', 'count_outs', 'start_speed', 'end_speed',
    'zone', 'type_confidence', 'plate_time', 'extension', 'strike_zone_top', 'strike_zone_bottom',
    'plate_x', 'plate_z', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_x', 'sz_y', 'spin_rate', 'spin_direction',
    'break_angle', 'break_length', 'break_y', 'break_vertical', 'break_vertical_induced',
    'break_horizontal', 'launch_speed', 'launch_angle', 'total_distance', 'trajectory',
    'hardness', 'hit_location', 'hit_coord_x', 'hit_coord_y', 'gamepk',
    'batter_player_name', 'pitcher_player_name', 'batter_team_id', 'batter_team_name',
    'pitcher_team_id', 'pitcher_team_name',
]

# ---- Column type schemas (single source of truth for casting) ----
# All entity keys (personId, gamepk, team_id) are str to ensure join consistency.
# Numeric stats are float32 to save memory; IDs/counts that are never fractional are int32.
# All timestamps are datetime64[ns] UTC-naive (MLB API returns Eastern, we normalize at source).

SCHEDULE_SCHEMA = {
    'gamepk': 'str',
    'game_datetime': 'datetime64[ns]',
    'game_date': 'datetime64[ns]',
    'game_type': 'str',
    'status': 'str',
    'away_name': 'str',
    'home_name': 'str',
    'away_id': 'str',       # keep as str - used in joins with batter_team_id
    'home_id': 'str',       # same
    'game_num': 'int32',
    'home_probable_pitcher': 'str',
    'away_probable_pitcher': 'str',
    'away_score': 'float32',   # float to handle nulls (postponed games etc.)
    'home_score': 'float32',
    'venue_id': 'str',
    'venue_name': 'str',
    'winning_team': 'str',
    'losing_team': 'str',
    'winning_pitcher': 'str',
    'losing_pitcher': 'str',
}

PITCHER_BOXSCORE_SCHEMA = {
    'personId': 'str',
    'gamepk': 'str',
    'ip': 'float32',
    'h': 'float32',
    'r': 'float32',
    'er': 'float32',
    'bb': 'float32',
    'k': 'float32',
    'hr': 'float32',
    'p': 'float32',
    's': 'float32',
    'outcome': 'str',          # W/L/S or NaN
    'fantasy_points': 'float32',
    'team_id': 'str',
}

BATTER_BOXSCORE_SCHEMA = {
    'gamepk': 'str',
    'batting_order': 'str',    # "1"-"9" or None
    'personId': 'str',
    'team_id': 'str',
    'ab': 'int32',
    'h': 'int32',
    'r': 'int32',
    'doubles': 'int32',
    'triples': 'int32',
    'hr': 'int32',
    'rbi': 'int32',
    'sb': 'int32',
    'bb': 'int32',
    'k': 'int32',
    'lob': 'int32',
    'plate_appearances': 'int32',
    'singles': 'int32',
    'total_bases_from_h': 'int32',
    'total_bases': 'int32',
    'fantasy_points': 'float32',
}

PLAYER_INFO_SCHEMA = {
    'personId': 'str',
    'player_name': 'str',
    'weight': 'float32',
    'height_in_inches': 'int32',
    'birthDate': 'datetime64[ns]',
    'strikeZoneTop': 'float32',
    'strikeZoneBottom': 'float32',
    'batSide': 'str',
    'pitchHand': 'str',
}

GAME_INFO_SCHEMA = {
    'gamepk': 'str',
    'game_season': 'Int32',
    'weather_condition': 'str',
    'weather_temp': 'str',     # "72 F" etc - parse later if needed as feature
    'weather_wind': 'str',
    'probable_pitcher_away_id': 'str',
    'probable_pitcher_away_fullName': 'str',
    'probable_pitcher_home_id': 'str',
    'probable_pitcher_home_fullName': 'str',
    'game_duration_minutes': 'float32',  # float to handle nulls
}

PREPARE_BATTER_BOXSCORE_SCHEMA = {
    # from BATTER_BOXSCORE_SCHEMA
    'gamepk': 'str',
    'batting_order': 'str',
    'personId': 'str',
    'team_id': 'str',
    'ab': 'int32',
    'h': 'int32',
    'r': 'int32',
    'doubles': 'int32',
    'triples': 'int32',
    'hr': 'int32',
    'rbi': 'int32',
    'sb': 'int32',
    'bb': 'int32',
    'k': 'int32',
    'lob': 'int32',
    'plate_appearances': 'int32',
    'singles': 'int32',
    'total_bases_from_h': 'int32',
    'total_bases': 'int32',
    'fantasy_points': 'float32',
    # added by prepare_*
    'player_name': 'str',
    'game_type': 'str',
    'game_date': 'datetime64[ns]',
    'game_datetime': 'datetime64[ns]',
    'game_season': 'Int32',   # nullable — some games missing from processed schedule (non-Final status)
}

PREPARE_PITCHER_BOXSCORE_SCHEMA = {
    'personId': 'str',
    'gamepk': 'str',
    'team_id': 'str',
    'ip': 'float32',
    'h': 'float32',
    'r': 'float32',
    'er': 'float32',
    'bb': 'float32',
    'k': 'float32',
    'hr': 'float32',
    'p': 'float32',
    's': 'float32',
    'outcome': 'str',
    'fantasy_points': 'float32',
    # added by prepare_*
    'player_name': 'str',
    'game_type': 'str',
    'game_date': 'datetime64[ns]',
    'game_datetime': 'datetime64[ns]',
    'game_season': 'Int32',
}

# Always present - if these are missing something went wrong
PLAYBYPLAY_REQUIRED_COLUMNS = [
    'play_id', 'inning', 'half_inning', 'batter_id', 'batter_name', 
    'pitcher_id', 'pitcher_name', 'play_result', 'play_description', 
    'event_index', 'event_type', 'is_pitch', 'pitch_number', 'pitch_call',
    'pitch_call_code', 'pitch_type', 'pitch_type_code', 'is_in_play', 
    'is_strike', 'is_ball', 'is_out', 'count_balls', 'count_strikes', 
    'count_outs', 'gamepk', 'batter_player_name',
    'pitcher_player_name', 'batter_team_id', 'batter_team_name',
    'pitcher_team_id', 'pitcher_team_name',
]

# Statcast era only - legitimately absent for older games
PLAYBYPLAY_STATCAST_COLUMNS = [
    'start_speed', 'end_speed', 'zone', 'type_confidence', 'plate_time',
    'extension', 'strike_zone_top', 'strike_zone_bottom', 'plate_x', 'plate_z',
    'release_pos_x', 'release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_x', 'sz_y', 'spin_rate',
    'spin_direction', 'break_angle', 'break_length', 'break_y', 
    'break_vertical', 'break_vertical_induced', 'break_horizontal',
    'launch_speed', 'launch_angle', 'total_distance', 'trajectory',
    'hardness', 'hit_location', 'hit_coord_x', 'hit_coord_y',
]

# Explicit types for statcast columns — all float64 except categoricals and location int
PLAYBYPLAY_STATCAST_SCHEMA = {
    'start_speed':              'float64',
    'end_speed':                'float64',
    'zone':                     'float64',
    'type_confidence':          'float64',
    'plate_time':               'float64',
    'extension':                'float64',
    'strike_zone_top':          'float64',
    'strike_zone_bottom':       'float64',
    'plate_x':                  'float64',
    'plate_z':                  'float64',
    'release_pos_x':            'float64',
    'release_pos_y':            'float64',
    'release_pos_z':            'float64',
    'pfx_x':                    'float64',
    'pfx_z':                    'float64',
    'vx0':                      'float64',
    'vy0':                      'float64',
    'vz0':                      'float64',
    'ax':                       'float64',
    'ay':                       'float64',
    'az':                       'float64',
    'sz_x':                     'float64',
    'sz_y':                     'float64',
    'spin_rate':                'float64',
    'spin_direction':           'float64',
    'break_angle':              'float64',
    'break_length':             'float64',
    'break_y':                  'float64',
    'break_vertical':           'float64',
    'break_vertical_induced':   'float64',
    'break_horizontal':         'float64',
    'launch_speed':             'float64',
    'launch_angle':             'float64',
    'total_distance':           'float64',
    'trajectory':               'str',
    'hardness':                 'str',
    'hit_location':             'float64',  # float to handle nulls (most pitches have no hit)
    'hit_coord_x':              'float64',
    'hit_coord_y':              'float64',
}

# Combined - preserves the original PLAYBYPLAY_COLUMNS contract
PLAYBYPLAY_COLUMNS = PLAYBYPLAY_REQUIRED_COLUMNS + PLAYBYPLAY_STATCAST_COLUMNS

# FIX 3: Define PLAYBYPLAY_REQUIRED_SCHEMA (was referenced but never defined)
# Only covers PLAYBYPLAY_REQUIRED_COLUMNS — statcast cols are NaN-filled and not enforced
# since they're legitimately absent for pre-Statcast games.
PLAYBYPLAY_REQUIRED_SCHEMA = {
    'play_id':              'Int64',
    'inning':               'Int64',
    'half_inning':          'str',
    'batter_id':            'str',
    'batter_name':          'str',
    'pitcher_id':           'str',
    'pitcher_name':         'str',
    'play_result':          'str',
    'play_description':     'str',
    'event_index':          'Int64',
    'event_type':           'str',
    'is_pitch':             'bool',
    'pitch_number':         'Int64',
    'pitch_call':           'str',
    'pitch_call_code':      'str',
    'pitch_type':           'str',
    'pitch_type_code':      'str',
    'is_in_play':           'bool',
    'is_strike':            'bool',
    'is_ball':              'bool',
    'is_out':               'bool',
    'count_balls':          'Int64',
    'count_strikes':        'Int64',
    'count_outs':           'Int64',
    'gamepk':               'str',
    'batter_player_name':   'str',
    'pitcher_player_name':  'str',
    'batter_team_id':       'str',
    'batter_team_name':     'str',
    'pitcher_team_id':      'str',
    'pitcher_team_name':    'str',
}


def enforce_schema(df: pd.DataFrame, schema: dict, context: str = "") -> pd.DataFrame:
    """
    Cast all columns to declared types. Raises clearly on failure.
    String columns: NaN preserved as NaN (not cast to "nan").
    """
    df = df.copy()

    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"enforce_schema [{context}]: missing columns {missing}")

    for col, dtype in schema.items():
        try:
            if dtype == 'str':
                # avoid casting NaN to the string "nan"
                mask = df[col].isna()
                df[col] = df[col].astype(str)
                df[col] = df[col].mask(mask)

            elif dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            raise TypeError(f"enforce_schema [{context}]: cannot cast '{col}' to {dtype}: {e}")

    return df[list(schema.keys())]  # enforces column order too

def process_pitcher_boxscore(df):
    """
    Normalize pitcher boxscore: filter invalid rows, coerce types, add outcome and fantasy_points.

    Parameters
    ----------
    df : pd.DataFrame
        Raw pitcher boxscore. Must contain: gamePk, personId, ip, h, r, er, bb, k, hr, p, s,
        namefield, name, note.

    Returns
    -------
    pd.DataFrame
        One row per (personId, gamepk) with columns given by PITCHER_BOXSCORE_COLUMNS.
        gamePk is renamed to gamepk (str); numeric cols are float; outcome is W/L/S extract.
    """
    df = df[df['ip'] != 'IP']
    df = df.rename({'gamePk':'gamepk'}, axis=1)

    rel_str_convert_cols = ['personId', 'gamepk']
    df[rel_str_convert_cols] = df[rel_str_convert_cols].astype(int).astype(str)

    rel_float_cols = ['ip', 'h', 'r', 'er', 'bb', 'k', 'hr', 'p', 's']
    df[rel_float_cols] = df[rel_float_cols].astype(float)

    df = (
        df
        .assign(
            pitcher_name = lambda x: x['name'],
            outcome = lambda x: x['note'].str.extract(r'\((W|L|S)'),
            fantasy_points = lambda x:                 
                x['ip'] * 2.25 +       # Innings pitched: +2.25 pts per inning
                x['k'] * 2.00 +        # Strikeouts: +2.00 pts
                x['er'] * -2.00 +      # Earned runs allowed: -2.00 pts
                x['h'] * -0.60 +       # Hits against: -0.60 pts
                x['bb'] * -0.60        # Walks against: -0.60 pts
            )
        .drop(['namefield', 'name', 'note', 'pitcher_name'], axis=1) # if we add outcome later we'd need to dedupe
    )

    df = df.drop_duplicates().reset_index(drop=True)

    df['composite_score'] = (
        df[rel_float_cols].sum(axis=1) + 
        df['outcome'].notna().astype(int)
    ) # this creates a score that has the max actions

    df = df.sort_values(by='composite_score', ascending=False).drop_duplicates(['personId', 'gamepk']).reset_index(drop=True)
    df = df.drop(['composite_score'], axis=1)
    df = df[PITCHER_BOXSCORE_COLUMNS]

    return enforce_schema(df, schema=PITCHER_BOXSCORE_SCHEMA, context='process_pitcher_boxscore')


def process_batter_boxscore(df):
    """
    Normalize batter boxscore: filter out header rows, coerce types, add derived stats and team_id.

    Team lookup was generated by checking teamnames that were missing teamids and asking
    chatgpt for the mapping (mainly teams that switched names in later seasons).

    Parameters
    ----------
    df : pd.DataFrame
        Raw batter boxscore. Must contain: gamePk, personId, namefield, team_name, battingOrder,
        ab, h, r, doubles, triples, hr, rbi, sb, bb, k, lob, plus name, substitution, note.

    Returns
    -------
    pd.DataFrame
        One row per (personId, gamepk) with columns given by BATTER_BOXSCORE_COLUMNS.
        gamePk is renamed to gamepk (str); team_id from team_lookup; plate_appearances, singles, etc. added.
    """
    team_lookup = {
        # American League East
        'Orioles': 110,
        'Red Sox': 111,
        'Yankees': 147,
        'Rays': 139,
        'Blue Jays': 141,
        
        # American League Central
        'White Sox': 145,
        'Guardians': 114,
        'Tigers': 116,
        'Royals': 118,
        'Twins': 142,
        
        # American League West
        'Astros': 117,
        'Angels': 108,
        'Athletics': 133,
        'Mariners': 136,
        'Rangers': 140,
        
        # National League East
        'Braves': 144,
        'Marlins': 146,
        'Mets': 121,
        'Phillies': 143,
        'Nationals': 120,
        
        # National League Central
        'Cubs': 112,
        'Reds': 113,
        'Brewers': 158,
        'Pirates': 134,
        'Cardinals': 138,
        
        # National League West
        'Diamondbacks': 109,
        'Rockies': 115,
        'Dodgers': 119,
        'Padres': 135,
        'Giants': 137,
        
        # Historical/Former team names
        'Indians': 114,  # now Guardians
        'Devil Rays': 139,  # now Rays
        'Expos': 120,  # now Nationals
        'D-backs': 109
    }

    df = df[~df['namefield'].str.contains('Batters')]

    str_to_float_cols = ['ab', 'h', 'r', 'doubles','triples', 'hr', 'rbi', 'sb',
        'bb', 'k', 'lob']
    df[str_to_float_cols] = df[str_to_float_cols].astype(int)


    df = (
        df
        .assign(
            gamepk = lambda x: x['gamePk'].apply(lambda y: str(int(float(y)))),  # float → int → str
            batting_order_temp = lambda x: pd.to_numeric(x['battingOrder'], errors='coerce'),
            batting_order = lambda x: np.where(
                (x['batting_order_temp'] % 100 == 0) & x['batting_order_temp'].notna(),
                x['battingOrder'].astype(str).str[0],
                None
            ),
            team_name = lambda x: x['team_name'].str.replace(' Batters', '', regex=False),
            personId = lambda x: x['personId'].astype(int).astype(str),
            team_id = lambda x: x['team_name'].map(team_lookup).astype(str),
            plate_appearances = lambda x: x['ab'] + x['bb'],
            singles = lambda x: x['h'] - x['doubles'] - x['triples'] - x['hr'],
            total_bases_from_h = lambda x: x['singles'] + x['doubles']*2 + x['triples']*3 + x['hr']*4,
            total_bases = lambda x: x['singles'] + x['doubles']*2 + x['triples']*3 + x['hr']*4 + x['bb'],
            fantasy_points = lambda x: (
                x['singles'] * 3 +
                x['doubles'] * 5 +
                x['triples'] * 8 +
                x['hr'] * 10 +
                x['rbi'] * 2 +
                x['r'] * 2 +
                x['bb'] * 2 +
                x['sb'] * 5
            )
        )
        .drop(['namefield', 'name', 'batting_order_temp', 'gamePk', 'substitution', 'note', 'team_name', 'battingOrder'], axis=1)
    )

    df = df[(df['plate_appearances'] > 0) & (df['gamepk'] != '716404')]
    df = df.sort_values('ab', ascending=False).drop_duplicates(['personId', 'gamepk'])
    df = df[BATTER_BOXSCORE_COLUMNS]

    return enforce_schema(df, schema=BATTER_BOXSCORE_SCHEMA, context='process_batter_boxscore')


def process_schedule(df):
    """
    Filter to final games, normalize datetimes and game id, and return a fixed column set.

    Parameters
    ----------
    df : pd.DataFrame
        Raw schedule. Must contain at least: game_id, game_datetime, game_date,
        game_type, status, away_name, home_name, away_id, home_id, game_num,
        home_probable_pitcher, away_probable_pitcher, away_score, home_score,
        venue_id, venue_name, winning_team, losing_team, winning_pitcher, losing_pitcher.

    Returns
    -------
    pd.DataFrame
        One row per game with columns given by SCHEDULE_COLUMNS (see module-level constant).
        game_id is renamed to gamepk; game_datetime and game_date are datetime64.
    """
    df = df[df['status'] == 'Final'].copy()
    df['game_datetime'] = pd.to_datetime(df['game_datetime'])
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    df = (
        df
        .sort_values('game_date')
        .drop_duplicates(subset='game_id', keep='last'))

    df = df.rename({'game_id': 'gamepk'}, axis=1)

    return enforce_schema(df, schema=SCHEDULE_SCHEMA, context='process_schedule')


def process_player_info(df):
    """
    Dedupe by person_id, parse height and birthDate, rename person_id to personId, return fixed columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw player info. Must contain: person_id, player_name, weight, height, birthDate,
        strikeZoneTop, strikeZoneBottom, batSide, pitchHand.

    Returns
    -------
    pd.DataFrame
        One row per personId with columns given by PLAYER_INFO_COLUMNS.
        birthDate is datetime64; height_in_inches is int.
    """
    df = df.drop_duplicates(subset='person_id')  # some strike zones are slightly different
    df = df[df['height'].notna() & (df['height'] != '')]  # drop players with no height on record

    df = (
        df
        .assign(
            birthDate=lambda x: pd.to_datetime(x['birthDate']),
            height_feet=lambda x: x['height'].str.extract(r"(\d+)' (\d+)\"")[0].astype(int),
            height_inches=lambda x: x['height'].str.extract(r"(\d+)' (\d+)\"")[1].astype(int),
            height_in_inches=lambda x: x['height_feet'] * 12 + x['height_inches']
        )
        .rename({'person_id': 'personId'}, axis=1)
        .drop(['height_feet', 'height_inches', 'height'], axis=1)
    )
    df['personId'] = df['personId'].astype(str)
    df = df[PLAYER_INFO_COLUMNS]

    return enforce_schema(df, schema=PLAYER_INFO_SCHEMA, context='process_player_info')


def process_game_info(df):
    """
    Return a subset of game info columns (weather, probable pitchers, duration).

    Parameters
    ----------
    df : pd.DataFrame
        Raw game info. Must contain columns listed in GAME_INFO_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Columns given by GAME_INFO_COLUMNS (see module-level constant).
    """
    df = df.rename({'gamePk': 'gamepk'}, axis=1)
    return enforce_schema(df[GAME_INFO_COLUMNS], schema=GAME_INFO_SCHEMA, context='process_game_info')


def prepare_batter_boxscore(batter_boxscore, player_info, schedule):
    """
    Join batter boxscore with player_info (player_name) and schedule (game date/time/type), add game_season.

    Parameters
    ----------
    batter_boxscore : pd.DataFrame
        Output of process_batter_boxscore; must have columns BATTER_BOXSCORE_COLUMNS.
    player_info : pd.DataFrame
        Output of process_player_info; must have personId, player_name.
    schedule : pd.DataFrame
        Output of process_schedule; must have gamepk, game_type, game_date, game_datetime.

    Returns
    -------
    pd.DataFrame
        Columns given by PREPARE_BATTER_BOXSCORE_COLUMNS (see module-level constant).
    """
    batter_boxscore = pd.merge(
        batter_boxscore, 
        player_info, 
        on='personId', 
        how='left')

    schedule_join = schedule[['gamepk','game_type','game_date','game_datetime']].drop_duplicates()

    batter_boxscore = pd.merge(
        batter_boxscore, 
        schedule_join, 
        on=['gamepk'], 
        how='inner'
    ).sort_values(by=['game_datetime']).reset_index(drop=True)

    # Drop rows where schedule join failed — gamepk not in processed schedule means
    # the game was postponed or suspended and not marked Final
    unmatched = batter_boxscore['game_date'].isna().sum()
    if unmatched:
        logger.warning(f"prepare_batter_boxscore: dropping {unmatched} rows with no matching schedule entry (postponed/suspended games)")
    batter_boxscore = batter_boxscore[batter_boxscore['game_date'].notna()]

    batter_boxscore['game_season'] = batter_boxscore['game_date'].dt.year
    batter_boxscore = batter_boxscore[PREPARE_BATTER_BOXSCORE_COLUMNS]

    RELEVANT_GAME_TYPES = ['R', 'F', 'D', 'L', 'W']  # excludes S (spring training), E (exhibition), A (all-star)
    batter_boxscore = batter_boxscore[batter_boxscore['game_type'].isin(RELEVANT_GAME_TYPES)]

    return enforce_schema(
        batter_boxscore, 
        PREPARE_BATTER_BOXSCORE_SCHEMA, 
        context="prepare_batter_boxscore"
    )

def prepare_pitcher_boxscore(pitcher_boxscore, player_info, schedule):
    
    pitcher_boxscore = pd.merge(pitcher_boxscore, player_info, on='personId', how='left')

    unmatched = pitcher_boxscore['player_name'].isnull().sum()
    if unmatched:
        logger.warning(f"prepare_pitcher_boxscore: {unmatched} rows with no player_info match")

    schedule_join = schedule[['gamepk', 'game_type', 'game_date', 'game_datetime']].drop_duplicates()
    pitcher_boxscore = pd.merge(
        pitcher_boxscore,
        schedule_join,
        on='gamepk',
        how='inner'
    ).sort_values(by=['game_datetime']).reset_index(drop=True)

    pitcher_boxscore = pitcher_boxscore[pitcher_boxscore['game_type'].isin(RELEVANT_GAME_TYPES)]
    pitcher_boxscore['game_season'] = pitcher_boxscore['game_date'].dt.year

    pitcher_boxscore = pitcher_boxscore[PREPARE_PITCHER_BOXSCORE_COLUMNS]

    return enforce_schema(pitcher_boxscore, PREPARE_PITCHER_BOXSCORE_SCHEMA, context="prepare_pitcher_boxscore")

def prepare_playbyplay(playbyplay, player_info, schedule, strict=False):
    """
    Add batter/pitcher names and team ids to play-by-play; normalize ids to str.

    Parameters
    ----------
    playbyplay : pd.DataFrame
        Raw play-by-play. Must contain: batter_id, pitcher_id, gamepk, half_inning.
        May use 'batter'/'pitcher' instead of batter_id/pitcher_id (caller can normalize).
        Other columns (e.g. play_id, inning, play_result, pitch_number) are pass-through.
    player_info : pd.DataFrame
        Output of process_player_info; must have personId, player_name.
    schedule : pd.DataFrame
        Output of process_schedule; must have gamepk, away_name, home_name, away_id, home_id.

    Returns
    -------
    pd.DataFrame
        Columns given by PLAYBYPLAY_COLUMNS (see module-level constant). Any column missing
        from raw is present as NaN (reindex).
    """
    playbyplay[['batter_id', 'pitcher_id', 'gamepk']] = playbyplay[['batter_id', 'pitcher_id', 'gamepk']].astype(str)

    playbyplay = pd.merge(
        playbyplay,
        player_info[['player_name', 'personId']].rename({'player_name': 'batter_player_name'}, axis=1),
        left_on='batter_id',
        right_on='personId',
        how='left').drop(['personId'], axis=1)
    playbyplay['batter_player_name'] = playbyplay['batter_player_name'].fillna(playbyplay['batter_name'])

    playbyplay = pd.merge(
        playbyplay,
        player_info[['player_name', 'personId']].rename({'player_name': 'pitcher_player_name'}, axis=1),
        left_on='pitcher_id',
        right_on='personId',
        how='left').drop(['personId'], axis=1)
    playbyplay['pitcher_player_name'] = playbyplay['pitcher_player_name'].fillna(playbyplay['pitcher_name'])
    
    schedule = schedule[['gamepk', 'away_name', 'home_name', 'away_id', 'home_id', 'game_type']]

    playbyplay = pd.merge(playbyplay, schedule, how='left', on='gamepk')

    RELEVANT_GAME_TYPES = ['R', 'F', 'D', 'L', 'W']  # excludes S (spring training), E (exhibition), A (all-star)
    playbyplay = playbyplay[playbyplay['game_type'].isin(RELEVANT_GAME_TYPES)]

    playbyplay = (
        playbyplay
        .assign(
            batter_team_id=lambda x: np.where(x['half_inning']=='top', x['away_id'], x['home_id']),
            batter_team_name=lambda x: np.where(x['half_inning']=='top', x['away_name'], x['home_name']),
            pitcher_team_id=lambda x: np.where(x['half_inning']=='top', x['home_id'], x['away_id']),
            pitcher_team_name=lambda x: np.where(x['half_inning']=='top', x['home_name'], x['away_name'])
        )
        .drop(['home_id', 'away_id', 'home_name', 'away_name'], axis=1)
    )
    
    # enforce required columns - these must exist, raise if not
    missing_required = [c for c in PLAYBYPLAY_REQUIRED_COLUMNS if c not in playbyplay.columns]
    if missing_required:
        raise ValueError(f"prepare_playbyplay: missing required columns {missing_required}")

    # statcast columns - add as NaN if absent, but log which ones so you know
    missing_statcast = [c for c in PLAYBYPLAY_STATCAST_COLUMNS if c not in playbyplay.columns]
    if missing_statcast:
        print(f"prepare_playbyplay: statcast columns absent (expected for older data): {missing_statcast}")
        for col in missing_statcast:
            playbyplay[col] = np.nan

    required_df = enforce_schema(
        playbyplay,
        PLAYBYPLAY_REQUIRED_SCHEMA,
        context="prepare_playbyplay"
    )
    statcast_df = enforce_schema(
        playbyplay[PLAYBYPLAY_STATCAST_COLUMNS],
        PLAYBYPLAY_STATCAST_SCHEMA,
        context="prepare_playbyplay_statcast"
    )
    return pd.concat([required_df, statcast_df], axis=1)[PLAYBYPLAY_COLUMNS]

