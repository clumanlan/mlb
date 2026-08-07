import pandas as pd
import numpy as np

# Stat definitions, formulas, and "why this stat" rationale: see FEATURE_GLOSSARY.md
# (src/models/hit_predictor/FEATURE_GLOSSARY.md) — section 5 has a feature-name ->
# function lookup table. Keep definitions there, not duplicated in this file.

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
            if 'season_' in col
        })
    )

def _prefix_stat_cols(df: pd.DataFrame, prefix: str, key_cols: list[str]) -> pd.DataFrame:
    stat_cols = [c for c in df.columns if c not in key_cols]
    return df.rename(columns={c: f'{prefix}{c}' for c in stat_cols})

def _create_boxscore_batter_stats(batter_boxscore):

    df = (
        batter_boxscore
        .groupby(['personId', 'game_season'])
        [['h',  'k', 'bb', 'hr', 'ab', 'plate_appearances', 'total_bases_from_h']].sum()
        .reset_index()
    )

    df = _prefix_stat_cols(df, prefix='batter_season_', key_cols=['personId', 'game_season'])

    df = (
        df
        .assign(
            batter_season_ba = lambda x: np.round(x['batter_season_h'] / x['batter_season_ab'].replace(0, np.nan), 2),
            batter_season_slg = lambda x: np.round(x['batter_season_total_bases_from_h'] / x['batter_season_ab'].replace(0, np.nan), 2),
        )
        .assign(
            batter_season_iso = lambda x: np.round(x['batter_season_slg'] - x['batter_season_ba'], 2),
            batter_season_babip = lambda x: np.round(
                (x['batter_season_h'] - x['batter_season_hr'])
                / (x['batter_season_ab'] - x['batter_season_k'] - x['batter_season_hr']).replace(0, np.nan),
                2
            ),
        )
    )

    return df

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

    df = _prefix_stat_cols(df, 'pitcher_season_', key_cols=['pitcher_id', 'game_season'])

    return _shift_to_last_season(df)



# --------------------------- PITCHER PBP SEASON STATS --------------------------- #
# See FEATURE_GLOSSARY.md for stat definitions/rationale. Categories below:
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


    return _prefix_stat_cols(df, prefix='pitcher_season_', key_cols=['pitcher_id', 'game_season'])

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
    df['fip'] = (13 * df['fip_hr'] + 3 * df['fip_bb'] - 2 * df['fip_k']) / df['total'] + C

    return _prefix_stat_cols(df, prefix='pitcher_season_pa_', key_cols=['pitcher_id', 'game_season'])

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

    return _prefix_stat_cols(df, prefix='pitcher_season_', key_cols=['pitcher_id', 'game_season'])

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

    return _prefix_stat_cols(df, prefix='pitcher_season_game_', key_cols=['pitcher_id', 'game_season'])

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
        .reset_index()
    )

    return _prefix_stat_cols(df, prefix='pitcher_season_contact_', key_cols=['pitcher_id', 'game_season'])

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


# --------------------------- BATTER PBP SEASON STATS --------------------------- #
# See FEATURE_GLOSSARY.md section 5 for the feature -> function lookup table.

def _create_batter_pa_outcome_stats(pbp: pd.DataFrame) -> pd.DataFrame:

    last_pitch_pbp = (
        pbp[pbp['pitch_number'] == pbp.groupby(['gamepk', 'play_id'])['pitch_number'].transform('max')]
        .reset_index(drop=True)
    )

    df = (
        last_pitch_pbp
        .groupby(['batter_id', 'game_season'])
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
            avg_final_balls=('count_balls', 'mean'),
            avg_final_strikes=('count_strikes', 'mean'),
            full_count_rate=('count_balls', lambda x: ((x == 3) & (last_pitch_pbp.loc[x.index, 'count_strikes'] == 2)).mean()),
        )
        .reset_index()
    )

    return _prefix_stat_cols(df, prefix='batter_season_pa_', key_cols=['batter_id', 'game_season'])

def _create_batter_plate_discipline_stats(pbp: pd.DataFrame) -> pd.DataFrame:

    all_pitches_stats = (
        pbp
        .groupby(['batter_id', 'game_season'])
        .agg(
            o_swing_rate = ('is_chase', 'mean'),
            z_swing_rate = ('is_zone_swing', 'mean'),
            swinging_strike_rate = ('is_swinging_strike', 'mean'),
            zone_rate = ('zone', lambda x: x.isin(range(1, 10)).mean()),
        )
        .reset_index()
    )

    swings_only = pbp[pbp['is_swing']]

    contact_stats = (
        swings_only
        .groupby(['batter_id', 'game_season'])
        .agg(
            contact_rate = ('is_swinging_strike', lambda x: (~x).mean()),
        )
        .reset_index()
    )

    df = all_pitches_stats.merge(contact_stats, on=['batter_id', 'game_season'], how='left')

    return _prefix_stat_cols(df, prefix='batter_season_', key_cols=['batter_id', 'game_season'])

def _create_batter_in_play_contact_stats(pbp: pd.DataFrame) -> pd.DataFrame:

    contact_only = pbp[pbp.is_in_play == True]

    df = (
        contact_only
        .groupby(['batter_id', 'game_season'])
        .agg(
            hard_hit_rate = ('launch_speed', lambda x: (x.dropna() >= 95).mean()),
            sweet_spot_rate = ('launch_angle', lambda x: x.dropna().between(8, 32).mean()),
            gb_rate = ('trajectory', lambda x: x.eq('Ground Ball').mean()),
            fb_rate = ('trajectory', lambda x: x.eq('Fly Ball').mean()),
            ld_rate = ('trajectory', lambda x: x.eq('Line Drive').mean()),
            avg_launch_speed = ('launch_speed', 'mean'),
            avg_launch_angle = ('launch_angle', 'mean'),
        )
        .reset_index()
    )

    return _prefix_stat_cols(df, prefix='batter_season_contact_', key_cols=['batter_id', 'game_season'])

def _create_batter_foul_contact_stats(pbp: pd.DataFrame) -> pd.DataFrame:

    swings_only = pbp[pbp['is_swing']]

    foul_stats = (
        swings_only
        .groupby(['batter_id', 'game_season'])
        .agg(foul_rate = ('pitch_call', lambda x: x.eq('Foul').mean()))
        .reset_index()
    )

    contact_only = pbp[(pbp['pitch_call'] == 'Foul') | (pbp['is_in_play'] == True)]

    contact_foul_stats = (
        contact_only
        .groupby(['batter_id', 'game_season'])
        .agg(contact_foul_rate = ('pitch_call', lambda x: x.eq('Foul').mean()))
        .reset_index()
    )

    df = foul_stats.merge(contact_foul_stats, on=['batter_id', 'game_season'], how='left')

    return _prefix_stat_cols(df, prefix='batter_season_', key_cols=['batter_id', 'game_season'])

def _create_batter_two_strike_foul_stats(pbp: pd.DataFrame) -> pd.DataFrame:

    two_strike_swings = pbp[(pbp['count_strikes'] == 2) & (pbp['is_swing'])]

    df = (
        two_strike_swings
        .groupby(['batter_id', 'game_season'])
        .agg(two_strike_foul_rate = ('pitch_call', lambda x: x.eq('Foul').mean()))
        .reset_index()
    )

    return _prefix_stat_cols(df, prefix='batter_season_', key_cols=['batter_id', 'game_season'])

def build_pbp_batter_feats(pbp: pd.DataFrame) -> pd.DataFrame:

    """Build batter pbp features.

    Args:
        pbp: Pitch-by-pitch dataframe. No role-filter parameter — unlike
            build_pbp_pitcher_feats, there's no batter-side equivalent of
            pitcher_role; batter pbp features run over all pitches unconditionally.
    """

    batter_pa_outcome_stats = _create_batter_pa_outcome_stats(pbp)
    batter_plate_discipline_stats = _create_batter_plate_discipline_stats(pbp)
    batter_in_play_contact_stats = _create_batter_in_play_contact_stats(pbp)
    batter_foul_contact_stats = _create_batter_foul_contact_stats(pbp)
    batter_two_strike_foul_stats = _create_batter_two_strike_foul_stats(pbp)

    df = (
        batter_pa_outcome_stats
        .merge(batter_plate_discipline_stats, on=['batter_id', 'game_season'], how='left')
        .merge(batter_in_play_contact_stats, on=['batter_id', 'game_season'], how='left')
        .merge(batter_foul_contact_stats, on=['batter_id', 'game_season'], how='left')
        .merge(batter_two_strike_foul_stats, on=['batter_id', 'game_season'], how='left')
    )

    return _shift_to_last_season(df)



