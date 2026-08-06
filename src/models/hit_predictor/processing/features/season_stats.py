import pandas as pd
import numpy as np

def _shift_to_last_season(df):
    """Shift game_season forward one year and rename season_* -> last_season_*.

    Used to turn a season-aggregate feature table into a lookup that can be
    joined onto next season's games as "what did this player do last year."
    """

    return (
        df
        .assign(
            game_season = lambda x: x['game_season']+1
        )
        .rename(columns={
            col: col.replace('season_', 'last_season_')
            for col in df.columns
            if col.startswith('season_')
        })
    )

def _prefix_stat_pitcher_cols(df: pd.DataFrame, prefix: str, key_cols: list[str] = ['pitcher_id', 'game_season']) -> pd.DataFrame:
    stat_cols = [c for c in df.columns if c not in key_cols]
    return df.rename(columns={c: f'{prefix}{c}' for c in stat_cols})

def _create_boxscore_batter_stats(batter_boxscore):
        
    df = (
        batter_boxscore
        .groupby(['personId', 'game_season'])
        [['h',  'k', 'bb', 'hr', 'ab', 'plate_appearances']].sum()
        .reset_index()
        .rename(columns={
                'h': 'season_batter_total_h',
                'k': 'season_batter_total_k',
                'bb': 'season_batter_total_bb',
                'hr': 'season_batter_total_hr',
                'ab': 'season_batter_total_ab',
                'plate_appearances': 'season_batter_total_plate_appearances'
            })
        .assign(
            season_ba = lambda x: np.round(x['season_batter_total_h']/x['season_batter_total_ab'], 2)
        )
    )

    return _shift_to_last_season(df)

def build_batter_stats(batter_boxscore):

    df = _create_boxscore_batter_stats(batter_boxscore)

    return _shift_to_last_season(df)

def build_pitcher_stats(pitcher_boxscore):
        
    df = (
        pitcher_boxscore
        .groupby(['personId', 'game_season'])
        [['h', 'r', 'er', 'bb', 'hr', 'k', 'p', 's', 'ip']].sum()
        .reset_index()
        .assign(
                whip = lambda x: np.round((x['bb'] + x['h']) / x['ip'].replace(0, np.nan), 2),
                k_rate = lambda x: np.round(x['k'] / x['ip'].replace(0, np.nan), 2),
                bb_rate = lambda x: np.round(x['bb'] / x['ip'].replace(0, np.nan), 2),
                strike_rate = lambda x: np.round(x['s'] / x['p'].replace(0, np.nan), 2),
                hr_rate = lambda x: np.round(x['hr'] / x['ip'].replace(0, np.nan), 2),
                )
            )

    df = _prefix_stat_pitcher_cols(df, 'pitcher_season_')

    return _shift_to_last_season(df)



# --------------------------- PITCHER PBP SEASON STATS --------------------------- #

# create a longer documentation later:
# - stuff: physical properties of a pitch
# - command: ability to find the zone
# - sequencing: is he unpredictable? 
# - durability: #'s of time through order 


def _create_pitcher_stuff_command_stats(pbp_role):

    pbp_role['is_first_pitch_strike'] = (
        pbp_role['is_first_pitch'] & pbp_role['is_strike']
    )

    df = (
        pbp_role
        .groupby(['pitcher_id', 'game_season'])
        .agg(
            # stuff
            stuff_start_speed_mean = ('start_speed', 'mean'),
            stuff_start_speed_max = ('start_speed', 'max'),
            stuff_start_speed_std = ('start_speed', 'std'), # fatigue signal 
            stuff_end_speed_mean = ('end_speed', 'mean'),
            stuff_end_speed_max = ('end_speed', 'max'),
            stuff_perceived_velo_mean = ('perceived_velo', 'mean'),
            stuff_perceived_velo_max = ('perceived_velo', 'max'),
            stuff_spin_rate_mean = ('spin_rate', 'mean'),
            stuff_spin_rate_max = ('spin_rate', 'max'),
            stuff_movement_magnitude_mean = ('movement_magnitude', 'mean'),
            stuff_movement_magnitude_max = ('movement_magnitude', 'max'),
            stuff_pfx_z_mean = ('pfx_z', 'mean'), # vertical spin induced movement
            stuff_pfx_z_max = ('pfx_z', 'max'),
            stuff_extension_mean = ('extension', 'mean'),
            stuff_extension_max = ('extension', 'max'),
            stuff_extension_std = ('extension', 'std'),
            stuff_speed_retention_mean = ('speed_retention', 'mean'),  # end/start ratio

            # command 
            command_in_play_rate = ('is_in_play', 'mean'),
            command_swinging_strike_rate = ('is_swinging_strike', 'mean'),
            command_plate_x_std = ('plate_x', 'std'), # how does horizontal landing vary 
            command_plate_z_normalized_std = ('plate_z_normalized', 'std'), # how does vertical landing vary 
            command_zone_rate = ('zone', lambda x: x.isin(range(1,10)).mean()),
            command_ball_rate = ('is_ball', 'mean'),
            command_strike_rate = ('is_strike', 'mean'),
            command_called_strike_rate = ('is_called_strike', 'mean'),
            command_chase_rate = ('is_chase', 'mean'),
            command_zone_swing_rate = ('is_zone_swing', 'mean'),
            command_first_pitch_strike_rate = ('is_first_pitch_strike', 'mean'),
        )
        .reset_index()
    )


    return _prefix_stat_pitcher_cols(df, prefix='pitcher_season_')

def _create_pitcher_pa_outcome_stats(pbp_role: pd.DataFrame) -> pd.DataFrame:

    last_pitch_pbp = (
        pbp_role[pbp_role['pitch_number'] == pbp_role.groupby(['gamepk', 'play_id'])['pitch_number'].transform('max')]
        .reset_index(drop=True)
    )

    df = (
        last_pitch_pbp
        .groupby(['pitcher_id', 'game_season'])
        .agg(
            pitch_count_mean=('pitch_number', 'mean'),
            pitch_count_std=('pitch_number', 'std'),
            total=('play_result', 'count'),
            strikeout_rate=('play_result', lambda x: x.isin({"Strikeout", "Strikeout Double Play"}).mean()),
            walk_rate=('play_result', lambda x: x.isin({"Walk", "Intent Walk"}).mean()),
            hbp_rate=('play_result', lambda x: x.eq("Hit By Pitch").mean()),
            hit_rate=('play_result', lambda x: x.isin({"Single", "Double", "Triple", "Home Run"}).mean()),
            hr_rate=('play_result', lambda x: x.eq("Home Run").mean()),
            single_rate=('play_result', lambda x: x.eq("Single").mean()),
            xbh_rate=('play_result', lambda x: x.isin({"Double", "Triple", "Home Run"}).mean()),
            fip_k=('play_result', lambda x: x.isin({"Strikeout", "Strikeout Double Play"}).sum()),
            fip_bb=('play_result', lambda x: x.isin({"Walk", "Hit By Pitch"}).sum()),
            fip_hr=('play_result', lambda x: x.eq("Home Run").sum()),
            avg_final_balls=('count_balls', 'mean'),
            avg_final_strikes=('count_strikes', 'mean'),
            full_count_rate=('count_balls', lambda x: ((x == 3) & (last_pitch_pbp.loc[x.index, 'count_strikes'] == 2)).mean()),
        )
        .reset_index()
    )

    C = 3.10  # league average FIP constant, adjusts yearly
    df['pa_fip'] = (13 * df['pa_fip_hr'] + 3 * df['pa_fip_bb'] - 2 * df['pa_fip_k']) / df['pa_total'] + C

    return _prefix_stat_pitcher_cols(df, prefix='pitcher_season_pa_')

def _create_pitcher_last_inning_stats(pbp_role: pd.DataFrame) -> pd.DataFrame:

    df = (
        pbp_role.loc[pbp_role.groupby(['pitcher_id', 'gamepk'])['play_id'].idxmax()]
        .groupby(['pitcher_id', 'game_season'])
        .agg(
            avg_last_inning=('inning', 'mean'),
            std_last_inning=('inning', 'std'),
            avg_last_inning_velo=('start_speed', 'mean'),
            last_inning_ball_rate=('is_ball', 'mean'),
            last_inning_strike_rate=('is_strike', 'mean'),
            last_inning_avg_balls=('count_balls', 'mean'),
            last_inning_avg_strikes=('count_strikes', 'mean'),
            last_inning_outs=('count_outs', 'max'),
        )
        .reset_index()
    )

    return _prefix_stat_pitcher_cols(df, prefix='pitcher_season_')

def _create_pitcher_pitch_count_stats(pbp_role):

    game_pitch_totals = (
        pbp_role
        .groupby(['pitcher_id', 'gamepk', 'game_season'])['is_pitch']
        .sum()
        .reset_index()
        .rename(columns={'is_pitch':'total_pitches'})
    )

    df = (
        game_pitch_totals
        .groupby(['pitcher_id', 'game_season'])
        .agg(
        avg_pitch_count = ('total_pitches', 'mean'),
        std_pitch_count = ('total_pitches', 'std'),
        max_pitch_count = ('total_pitches', 'max')
        )
        .reset_index()
    ) 

    return _prefix_stat_pitcher_cols(df, prefix='pitcher_season_game_')

def _create_pitcher_contact_quality_stats(pbp_role):

    contact_only = pbp_role[pbp_role.is_in_play == True]

    df = (
        contact_only
        .groupby(['pitcher_id', 'game_season'])
        .agg(
            hard_hit_rate = ('hardness', lambda x: x.eq('Hard').mean()),
            gb_rate = ('trajectory', lambda x: x.eq('Ground Ball').mean()),
            fb_rate = ('trajectory', lambda x: x.eq('Fly Ball').mean()),
            ld_rate = ('trajectory', lambda x: x.eq('Line Drive').mean()),
            avg_launch_speed = ('launch_speed', 'mean'),
            avg_launch_angle = ('launch_angle', 'mean'),
        )
    )

    return _prefix_stat_pitcher_cols(df, prefix='pitcher_season_contact_')

def build_pbp_pitcher_feats(pbp: pd.DataFrame, pitcher_role: str | None = None) -> pd.DataFrame:

    """Build pitcher pbp features.

    Args:
        pbp: Pitch-by-pitch dataframe with pitcher_role column.
        pitcher_role: 'sp', 'bullpen', or None for all appearances.
    """

    if pitcher_role is not None:
        pbp = pbp[pbp['pitcher_role']==pitcher_role]

    pbp = pbp.assign(is_first_pitch_strike = lambda x: x['is_first_pitch'] & x['is_strike'])

    pitcher_stuff_command_stats = _create_pitcher_stuff_command_stats(pbp)
    pitcher_pa_outcome_stats = _create_pitcher_pa_outcome_stats(pbp)
    pitcher_last_inning_stats = _create_pitcher_last_inning_stats(pbp)
    pitcher_pitch_count_stats = _create_pitcher_pitch_count_stats(pbp)
    pitcher_contact_quality_stats = _create_pitcher_contact_quality_stats(pbp)

    df = (
        pitcher_stuff_command_stats
        .merge(pitcher_pa_outcome_stats, on=['pitcher_id','game_season'], how='left')
        .merge(pitcher_last_inning_stats, on=['pitcher_id','game_season'], how='left')
        .merge(pitcher_pitch_count_stats,  on=['pitcher_id','game_season'], how='left')
        .merge(pitcher_contact_quality_stats,  on=['pitcher_id','game_season'], how='left')
    )

    return _shift_to_last_season(df)





