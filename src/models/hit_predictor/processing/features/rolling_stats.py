import numpy as np
import pandas as pd

from .season_stats import _prefix_stat_cols, _pitcher_role_lookup
from models.hit_predictor.processing.pipeline import _create_batting_order

# Stat definitions, formulas, and "why this stat" rationale: see FEATURE_GLOSSARY.md
# and season_stats.py's header comment — same stat categories/formulas as season_stats.py,
# computed at per-game grain and rolled forward game-by-game instead of once per season.
#
# Core correctness rule throughout this file: roll counts, not rates. A rate is only ever
# divided once, at the very end (_finalize_rates), from already-rolled numerator/denominator
# sums — never by averaging per-game rates, which is wrong whenever games have different
# sample sizes.


def _rolling_sum(
    df: pd.DataFrame, entity_col: str, cols: list[str], window: str | int, sort_col: str = 'game_date'
) -> pd.DataFrame:
    """Roll `cols` per `entity_col`, excluding the current game's own row.

    window='season': expanding sum within (entity_col, game_season) — resets at season
        boundaries.
    window=<int>: trailing N-game sum per entity_col — carries across season boundaries,
        since a fixed-length recent-form window has no reason to reset on Opening Day.

    sort_col: column to order rows by before rolling — defaults to game_date, which
        has no time component and can't reliably order two rows sharing the same date
        (e.g. both games of a doubleheader). Pass 'game_datetime' when that matters
        (see test_rolling_sum_sort_col_orders_same_date_rows_by_finer_grained_column).
    """

    df = df.sort_values([entity_col, sort_col]).reset_index(drop=True)

    if window == 'season':
        rolled = df.groupby([entity_col, 'game_season'])[cols].transform(
            lambda s: s.cumsum().shift(1)
        )
    else:
        rolled = df.groupby(entity_col)[cols].transform(
            lambda s: s.rolling(window, min_periods=1).sum().shift(1)
        )

    return df.assign(**{c: rolled[c] for c in cols})


def _rolling_max(
    df: pd.DataFrame, entity_col: str, cols: list[str], window: str | int, sort_col: str = 'game_date'
) -> pd.DataFrame:
    """Same shape as _rolling_sum but takes a rolling max — exact for max-type
    stats, since max-of-per-game-maxes equals the true max across the window.
    See _rolling_sum's sort_col docstring."""

    df = df.sort_values([entity_col, sort_col]).reset_index(drop=True)

    if window == 'season':
        rolled = df.groupby([entity_col, 'game_season'])[cols].transform(
            lambda s: s.cummax().shift(1)
        )
    else:
        rolled = df.groupby(entity_col)[cols].transform(
            lambda s: s.rolling(window, min_periods=1).max().shift(1)
        )

    return df.assign(**{c: rolled[c] for c in cols})


def _rolling_pooled_std(
    df: pd.DataFrame,
    entity_col: str,
    n_col: str,
    sum_col: str,
    sumsq_col: str,
    window: str | int,
    out_col: str,
) -> pd.DataFrame:
    """Derive a rolling sample std from per-game (n, sum, sum_of_squares) —
    an exact rolling std can't be recovered from per-game std/mean alone, so
    the per-game layer must carry these three summary values instead.

    var = (sum_of_squares - sum^2/n) / (n-1); NaN when fewer than 2 prior pitches.
    """

    rolled = _rolling_sum(df, entity_col, [n_col, sum_col, sumsq_col], window)

    n = rolled[n_col]
    total = rolled[sum_col]
    total_sq = rolled[sumsq_col]

    variance = (total_sq - (total ** 2) / n) / (n - 1)
    variance = variance.where(n > 1)

    return rolled.assign(**{out_col: np.sqrt(variance)})


def _finalize_rates(df: pd.DataFrame, rate_defs: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Compute {output_col: numerator/denominator} once, from already-rolled
    sums — never from averaging per-game rates. Divide-by-zero guarded to NaN."""

    result = df.copy()
    for out_col, (num_col, denom_col) in rate_defs.items():
        result[out_col] = result[num_col] / result[denom_col].replace(0, np.nan)
    return result


def _validate_window(window: str | int) -> None:
    if window == 'season':
        return
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be 'season' or a positive int, got {window!r}")


def _rolling_prefix(entity: str, window: str | int) -> str:
    if window == 'season':
        return f'{entity}_roll_season_'
    return f'{entity}_roll_last{window}g_'


BOX_KEY_COLS = ['personId', 'gamepk', 'game_date', 'game_season']


def _box_key_cols(entity_col: str) -> list[str]:
    return [entity_col, 'gamepk', 'game_date', 'game_season']


def build_batter_rolling_stats(batter_boxscore: pd.DataFrame, window: str | int) -> pd.DataFrame:
    _validate_window(window)

    stat_cols = ['h', 'k', 'bb', 'hr', 'ab', 'plate_appearances', 'total_bases_from_h']
    df = batter_boxscore[BOX_KEY_COLS + stat_cols]

    df = _rolling_sum(df, entity_col='personId', cols=stat_cols, window=window)

    df = (
        df
        .assign(
            ba = lambda x: x['h'] / x['ab'].replace(0, np.nan),
            slg = lambda x: x['total_bases_from_h'] / x['ab'].replace(0, np.nan),
        )
        .assign(
            iso = lambda x: x['slg'] - x['ba'],
            babip = lambda x: (x['h'] - x['hr']) / (x['ab'] - x['k'] - x['hr']).replace(0, np.nan),
        )
    )

    return _prefix_stat_cols(df, prefix=_rolling_prefix('batter', window), key_cols=BOX_KEY_COLS)


def build_pitcher_rolling_stats(pitcher_boxscore: pd.DataFrame, window: str | int, entity_col: str = 'personId') -> pd.DataFrame:
    _validate_window(window)

    stat_cols = ['h', 'r', 'er', 'bb', 'hr', 'k', 'p', 's', 'ip']
    key_cols = _box_key_cols(entity_col)
    df = pitcher_boxscore[key_cols + stat_cols].copy()
    # games_n: rolling count of games pitched so far — a real, meaningful 0
    # for a season/career opener (not NaN), same convention as
    # game_context.py's build_pitcher_start_ip_this_season's starts_n.
    # Workload sample-size signal in its own right, and the shrinkage-weight
    # input for k_predictor's rolling-stats-shrunk-to-last-season blend.
    df['games_n'] = 1

    df = _rolling_sum(df, entity_col=entity_col, cols=stat_cols + ['games_n'], window=window)
    df['games_n'] = df['games_n'].fillna(0)

    df = df.assign(
        whip = lambda x: (x['bb'] + x['h']) / x['ip'].replace(0, np.nan),
        k_rate = lambda x: x['k'] / x['ip'].replace(0, np.nan),
        bb_rate = lambda x: x['bb'] / x['ip'].replace(0, np.nan),
        strike_rate = lambda x: x['s'] / x['p'].replace(0, np.nan),
        hr_rate = lambda x: x['hr'] / x['ip'].replace(0, np.nan),
    )

    return _prefix_stat_cols(df, prefix=_rolling_prefix('pitcher', window), key_cols=key_cols)


def build_pitcher_rolling_stats_all_roles(pitcher_boxscore: pd.DataFrame, pbp: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """Rolling equivalent of season_stats.build_pitcher_stats_all_roles: sp
    rows rolled per individual pitcher as before, bullpen rows pooled by
    team (a specific reliever's identity isn't knowable pre-game). Both
    tagged and stacked under a common pitcher_key_id column.
    """

    # Only pitcher_role is needed — pitcher_boxscore already has its own
    # team_id column, so pulling in pitcher_team_id too would just collide
    # with it (team_id_x/team_id_y) for no benefit.
    role_lookup = _pitcher_role_lookup(pbp)[['gamepk', 'pitcher_id', 'pitcher_role']].rename(
        columns={'pitcher_id': 'personId'}
    )
    # personId/gamepk come straight from parquet and may not match pbp's
    # str-cast id columns in dtype — cast both explicitly before merging,
    # same defensive pattern as pipeline.py's _add_pbp_handedness.
    tagged = pitcher_boxscore.assign(
        personId=lambda x: x['personId'].astype(str), gamepk=lambda x: x['gamepk'].astype(str)
    ).merge(
        role_lookup.assign(
            personId=lambda x: x['personId'].astype(str), gamepk=lambda x: x['gamepk'].astype(str)
        ),
        on=['gamepk', 'personId'],
        how='left',
    )

    sp_box = tagged[tagged['pitcher_role'] == 'sp']
    bullpen_box = tagged[tagged['pitcher_role'] == 'bullpen']

    sp = (
        build_pitcher_rolling_stats(sp_box, window=window, entity_col='personId')
        .rename(columns={'personId': 'pitcher_key_id'})
        .assign(pitcher_role='sp')
    )
    # pitcher_boxscore has one row per INDIVIDUAL pitcher per game — a real bullpen
    # outing routinely uses 2+ relievers in the same game. _rolling_sum (inside
    # build_pitcher_rolling_stats) operates via .transform(), which preserves one
    # output row per INPUT row rather than collapsing to one per (team, game) — so
    # multiple relievers in the same game must be summed into one team-game total
    # BEFORE rolling, or every downstream join fans out and shift(1) corrupts across
    # teammates sharing the same game_date instead of skipping the whole game. This
    # mirrors what the pbp-derived per-game layer (_pitcher_pbp_per_game) already
    # does before its own rolling step.
    bullpen_per_game = (
        bullpen_box
        .groupby(['team_id', 'gamepk', 'game_date', 'game_season'])
        [['h', 'r', 'er', 'bb', 'hr', 'k', 'p', 's', 'ip']].sum()
        .reset_index()
    )
    bullpen = (
        build_pitcher_rolling_stats(bullpen_per_game, window=window, entity_col='team_id')
        .rename(columns={'team_id': 'pitcher_key_id'})
        .assign(pitcher_role='bullpen')
    )

    return pd.concat([sp, bullpen], ignore_index=True)


# --------------------------- PITCHER PBP ROLLING STATS --------------------------- #
# Same 5 categories as season_stats.py's build_pbp_pitcher_feats (stuff, command,
# PA-outcome, last-inning, pitch-count, contact quality), collapsed to per-game grain
# first, then rolled. Every mean/rate is decomposed into raw sum+n (or true-count+total)
# at the per-game layer and only divided once, after rolling — see module docstring.

PBP_PITCHER_KEY_COLS = ['pitcher_id', 'gamepk', 'game_date', 'game_season']


def _pbp_pitcher_key_cols(entity_col: str) -> list[str]:
    return [entity_col, 'gamepk', 'game_date', 'game_season']


def _pitcher_stuff_command_per_game(pbp: pd.DataFrame, entity_col: str = 'pitcher_id') -> pd.DataFrame:
    group_cols = _pbp_pitcher_key_cols(entity_col)

    df = (
        pbp
        .groupby(group_cols)
        .agg(
            stuff_start_speed_sum=('start_speed', 'sum'), stuff_start_speed_n=('start_speed', 'count'),
            stuff_start_speed_sumsq=('start_speed', lambda s: (s ** 2).sum()),
            stuff_start_speed_max=('start_speed', 'max'),

            stuff_end_speed_sum=('end_speed', 'sum'), stuff_end_speed_n=('end_speed', 'count'),
            stuff_end_speed_max=('end_speed', 'max'),

            stuff_perceived_velo_sum=('perceived_velo', 'sum'), stuff_perceived_velo_n=('perceived_velo', 'count'),
            stuff_perceived_velo_max=('perceived_velo', 'max'),

            stuff_spin_rate_sum=('spin_rate', 'sum'), stuff_spin_rate_n=('spin_rate', 'count'),
            stuff_spin_rate_max=('spin_rate', 'max'),

            stuff_movement_magnitude_sum=('movement_magnitude', 'sum'),
            stuff_movement_magnitude_n=('movement_magnitude', 'count'),
            stuff_movement_magnitude_max=('movement_magnitude', 'max'),

            stuff_pfx_z_sum=('pfx_z', 'sum'), stuff_pfx_z_n=('pfx_z', 'count'),
            stuff_pfx_z_max=('pfx_z', 'max'),

            stuff_extension_sum=('extension', 'sum'), stuff_extension_n=('extension', 'count'),
            stuff_extension_sumsq=('extension', lambda s: (s ** 2).sum()),
            stuff_extension_max=('extension', 'max'),

            stuff_speed_retention_sum=('speed_retention', 'sum'), stuff_speed_retention_n=('speed_retention', 'count'),

            command_plate_x_sum=('plate_x', 'sum'), command_plate_x_n=('plate_x', 'count'),
            command_plate_x_sumsq=('plate_x', lambda s: (s ** 2).sum()),

            command_plate_z_normalized_sum=('plate_z_normalized', 'sum'),
            command_plate_z_normalized_n=('plate_z_normalized', 'count'),
            command_plate_z_normalized_sumsq=('plate_z_normalized', lambda s: (s ** 2).sum()),

            command_in_play_n=('is_in_play', 'sum'),
            command_swinging_strike_n=('is_swinging_strike', 'sum'),
            command_zone_n=('zone', lambda x: x.isin(range(1, 10)).sum()),
            command_ball_n=('is_ball', 'sum'),
            command_strike_n=('is_strike', 'sum'),
            command_called_strike_n=('is_called_strike', 'sum'),
            command_chase_n=('is_chase', 'sum'),
            command_zone_swing_n=('is_zone_swing', 'sum'),
            command_first_pitch_strike_n=('is_first_pitch_strike', 'sum'),
        )
        .reset_index()
    )

    n_pitches = pbp.groupby(group_cols).size().rename('n_pitches').reset_index()

    return df.merge(n_pitches, on=group_cols, how='left')


def _pitcher_pa_outcome_per_game(pbp: pd.DataFrame, entity_col: str = 'pitcher_id') -> pd.DataFrame:
    group_cols = _pbp_pitcher_key_cols(entity_col)

    last_pitch_pbp = (
        pbp[pbp['pitch_number'] == pbp.groupby(['gamepk', 'play_id'])['pitch_number'].transform('max')]
        .reset_index(drop=True)
    )

    return (
        last_pitch_pbp
        .groupby(group_cols)
        .agg(
            pa_total=('play_result', 'count'),
            pa_pitch_count_sum=('pitch_number', 'sum'),
            pa_pitch_count_sumsq=('pitch_number', lambda s: (s ** 2).sum()),
            pa_strikeout_n=('play_result', lambda x: x.isin({"Strikeout", "Strikeout Double Play"}).sum()),
            pa_walk_n=('play_result', lambda x: x.isin({"Walk", "Intent Walk"}).sum()),
            pa_hbp_n=('play_result', lambda x: x.eq("Hit By Pitch").sum()),
            pa_hit_n=('play_result', lambda x: x.isin({"Single", "Double", "Triple", "Home Run"}).sum()),
            pa_hr_n=('play_result', lambda x: x.eq("Home Run").sum()),
            pa_single_n=('play_result', lambda x: x.eq("Single").sum()),
            pa_xbh_n=('play_result', lambda x: x.isin({"Double", "Triple", "Home Run"}).sum()),
            pa_fip_bb_n=('play_result', lambda x: x.isin({"Walk", "Hit By Pitch"}).sum()),
            pa_final_balls_sum=('count_balls', 'sum'), pa_final_balls_n=('count_balls', 'count'),
            pa_final_strikes_sum=('count_strikes', 'sum'), pa_final_strikes_n=('count_strikes', 'count'),
            pa_full_count_n=('count_balls', lambda x: (
                (x == 3) & (last_pitch_pbp.loc[x.index, 'count_strikes'] == 2)
            ).sum()),
        )
        .reset_index()
    )


def _pitcher_last_inning_per_game(pbp: pd.DataFrame, entity_col: str = 'pitcher_id') -> pd.DataFrame:
    group_cols = _pbp_pitcher_key_cols(entity_col)

    last_pitch_pbp = (
        pbp[pbp['pitch_number'] == pbp.groupby(['gamepk', 'play_id'])['pitch_number'].transform('max')]
        .reset_index(drop=True)
    )
    # At entity_col='pitcher_team_id' this naturally means "the last pitch
    # thrown by any reliever on that team that game" — the right team-level
    # meaning, for free, from generalizing the same idxmax selection.
    last_pa_per_game = last_pitch_pbp.loc[
        last_pitch_pbp.groupby([entity_col, 'gamepk'])['play_id'].idxmax()
    ]

    cols = group_cols + [
        'inning', 'start_speed', 'is_ball', 'is_strike', 'count_balls', 'count_strikes', 'count_outs',
    ]
    return (
        last_pa_per_game[cols]
        .rename(columns={
            'inning': 'last_inning_inning',
            'start_speed': 'last_inning_velo',
            'is_ball': 'last_inning_ball',
            'is_strike': 'last_inning_strike',
            'count_balls': 'last_inning_balls',
            'count_strikes': 'last_inning_strikes',
            'count_outs': 'last_inning_outs',
        })
        .reset_index(drop=True)
    )


def _pitcher_contact_quality_per_game(pbp: pd.DataFrame, entity_col: str = 'pitcher_id') -> pd.DataFrame:
    group_cols = _pbp_pitcher_key_cols(entity_col)

    contact_only = pbp[pbp['is_in_play'] == True]

    return (
        contact_only
        .groupby(group_cols)
        .agg(
            contact_n=('trajectory', 'count'),
            contact_hard_hit_n=('hardness', lambda x: x.eq('Hard').sum()),
            contact_gb_n=('trajectory', lambda x: x.eq('Ground Ball').sum()),
            contact_fb_n=('trajectory', lambda x: x.eq('Fly Ball').sum()),
            contact_ld_n=('trajectory', lambda x: x.eq('Line Drive').sum()),
            contact_launch_speed_sum=('launch_speed', 'sum'), contact_launch_speed_n=('launch_speed', 'count'),
            contact_launch_angle_sum=('launch_angle', 'sum'), contact_launch_angle_n=('launch_angle', 'count'),
        )
        .reset_index()
    )


PITCHER_PBP_CONTACT_COUNT_COLS = [
    'contact_n', 'contact_hard_hit_n', 'contact_gb_n', 'contact_fb_n', 'contact_ld_n',
    'contact_launch_speed_sum', 'contact_launch_speed_n', 'contact_launch_angle_sum', 'contact_launch_angle_n',
]


def _pitcher_pbp_per_game(pbp: pd.DataFrame, pitcher_role: str | None = None, entity_col: str = 'pitcher_id') -> pd.DataFrame:
    if pitcher_role is not None:
        pbp = pbp[pbp['pitcher_role'] == pitcher_role]

    pbp = pbp.assign(is_first_pitch_strike=lambda x: x['is_first_pitch'] & x['is_strike'])

    key_cols = _pbp_pitcher_key_cols(entity_col)
    df = _pitcher_stuff_command_per_game(pbp, entity_col=entity_col)
    df = df.merge(_pitcher_pa_outcome_per_game(pbp, entity_col=entity_col), on=key_cols, how='left')
    df = df.merge(_pitcher_last_inning_per_game(pbp, entity_col=entity_col), on=key_cols, how='left')
    df = df.merge(_pitcher_contact_quality_per_game(pbp, entity_col=entity_col), on=key_cols, how='left')

    df[PITCHER_PBP_CONTACT_COUNT_COLS] = df[PITCHER_PBP_CONTACT_COUNT_COLS].astype(float).fillna(0.0)

    df['games_n'] = 1
    df['last_inning_inning_sumsq'] = df['last_inning_inning'] ** 2
    df['pitch_count_sum_src'] = df['n_pitches']
    df['pitch_count_sumsq_src'] = df['n_pitches'] ** 2
    df['pitch_count_max_src'] = df['n_pitches']

    return df


def build_pbp_pitcher_rolling_feats(
    pbp: pd.DataFrame, window: str | int, pitcher_role: str | None = None, entity_col: str = 'pitcher_id'
) -> pd.DataFrame:
    """Rolling equivalent of season_stats.py's build_pbp_pitcher_feats. Same category
    boundaries and formulas, computed at per-game grain and rolled forward.

    entity_col: 'pitcher_id' (default) for per-pitcher rolling stats, or
    'pitcher_team_id' to pool a team's appearances (e.g. bullpen) into one
    rolled series instead of one per individual pitcher.
    """

    _validate_window(window)

    key_cols = _pbp_pitcher_key_cols(entity_col)
    per_game = _pitcher_pbp_per_game(pbp, pitcher_role=pitcher_role, entity_col=entity_col)

    sum_cols = [
        'stuff_end_speed_sum', 'stuff_end_speed_n',
        'stuff_perceived_velo_sum', 'stuff_perceived_velo_n',
        'stuff_spin_rate_sum', 'stuff_spin_rate_n',
        'stuff_movement_magnitude_sum', 'stuff_movement_magnitude_n',
        'stuff_pfx_z_sum', 'stuff_pfx_z_n',
        'stuff_speed_retention_sum', 'stuff_speed_retention_n',
        'command_in_play_n', 'command_swinging_strike_n', 'command_zone_n', 'command_ball_n',
        'command_strike_n', 'command_called_strike_n', 'command_chase_n', 'command_zone_swing_n',
        'command_first_pitch_strike_n', 'n_pitches',
        'pa_total', 'pa_pitch_count_sum', 'pa_strikeout_n', 'pa_walk_n', 'pa_hbp_n', 'pa_hit_n', 'pa_hr_n', 'pa_single_n',
        'pa_xbh_n', 'pa_fip_bb_n', 'pa_final_balls_sum', 'pa_final_balls_n',
        'pa_final_strikes_sum', 'pa_final_strikes_n', 'pa_full_count_n',
        'games_n', 'last_inning_inning', 'last_inning_velo',
        'last_inning_ball', 'last_inning_strike', 'last_inning_balls', 'last_inning_strikes',
        'pitch_count_sum_src',
    ] + PITCHER_PBP_CONTACT_COUNT_COLS

    max_cols = [
        'stuff_start_speed_max', 'stuff_end_speed_max', 'stuff_perceived_velo_max',
        'stuff_spin_rate_max', 'stuff_movement_magnitude_max', 'stuff_pfx_z_max', 'stuff_extension_max',
        'last_inning_outs', 'pitch_count_max_src',
    ]

    rolled = _rolling_sum(per_game, entity_col=entity_col, cols=sum_cols, window=window)
    rolled = _rolling_max(rolled, entity_col=entity_col, cols=max_cols, window=window)

    # std stats: each fed the pristine per_game (not `rolled`) so their internal rolling
    # of n/sum/sumsq never re-rolls a column the two calls above already rolled. n_col/sum_col
    # are re-merged in (rolled version replacing the stale raw one still sitting in `target`
    # from per_game) since std-only-path columns like stuff_start_speed_sum/n never went
    # through the plain _rolling_sum(sum_cols) pass above but their _mean still needs them rolled.
    def _merge_std(target, n_col, sum_col, sumsq_col, out_col):
        std_df = _rolling_pooled_std(per_game, entity_col, n_col, sum_col, sumsq_col, window, out_col)
        target = target.drop(columns=[n_col, sum_col])
        return target.merge(std_df[key_cols + [n_col, sum_col, out_col]], on=key_cols, how='left')

    rolled = _merge_std(rolled, 'stuff_start_speed_n', 'stuff_start_speed_sum', 'stuff_start_speed_sumsq', 'stuff_start_speed_std')
    rolled = _merge_std(rolled, 'stuff_extension_n', 'stuff_extension_sum', 'stuff_extension_sumsq', 'stuff_extension_std')
    rolled = _merge_std(rolled, 'command_plate_x_n', 'command_plate_x_sum', 'command_plate_x_sumsq', 'command_plate_x_std')
    rolled = _merge_std(rolled, 'command_plate_z_normalized_n', 'command_plate_z_normalized_sum', 'command_plate_z_normalized_sumsq', 'command_plate_z_normalized_std')
    rolled = _merge_std(rolled, 'games_n', 'last_inning_inning', 'last_inning_inning_sumsq', 'last_inning_std')
    rolled = _merge_std(rolled, 'games_n', 'pitch_count_sum_src', 'pitch_count_sumsq_src', 'pitch_count_std')
    rolled = _merge_std(rolled, 'pa_total', 'pa_pitch_count_sum', 'pa_pitch_count_sumsq', 'pa_pitch_count_std')

    rolled = rolled.assign(
        stuff_start_speed_mean = lambda x: x['stuff_start_speed_sum'] / x['stuff_start_speed_n'].replace(0, np.nan),
        stuff_end_speed_mean = lambda x: x['stuff_end_speed_sum'] / x['stuff_end_speed_n'].replace(0, np.nan),
        stuff_perceived_velo_mean = lambda x: x['stuff_perceived_velo_sum'] / x['stuff_perceived_velo_n'].replace(0, np.nan),
        stuff_spin_rate_mean = lambda x: x['stuff_spin_rate_sum'] / x['stuff_spin_rate_n'].replace(0, np.nan),
        stuff_movement_magnitude_mean = lambda x: x['stuff_movement_magnitude_sum'] / x['stuff_movement_magnitude_n'].replace(0, np.nan),
        stuff_pfx_z_mean = lambda x: x['stuff_pfx_z_sum'] / x['stuff_pfx_z_n'].replace(0, np.nan),
        stuff_extension_mean = lambda x: x['stuff_extension_sum'] / x['stuff_extension_n'].replace(0, np.nan),
        stuff_speed_retention_mean = lambda x: x['stuff_speed_retention_sum'] / x['stuff_speed_retention_n'].replace(0, np.nan),

        command_in_play_rate = lambda x: x['command_in_play_n'] / x['n_pitches'].replace(0, np.nan),
        command_swinging_strike_rate = lambda x: x['command_swinging_strike_n'] / x['n_pitches'].replace(0, np.nan),
        command_zone_rate = lambda x: x['command_zone_n'] / x['n_pitches'].replace(0, np.nan),
        command_ball_rate = lambda x: x['command_ball_n'] / x['n_pitches'].replace(0, np.nan),
        command_strike_rate = lambda x: x['command_strike_n'] / x['n_pitches'].replace(0, np.nan),
        command_called_strike_rate = lambda x: x['command_called_strike_n'] / x['n_pitches'].replace(0, np.nan),
        command_chase_rate = lambda x: x['command_chase_n'] / x['n_pitches'].replace(0, np.nan),
        command_zone_swing_rate = lambda x: x['command_zone_swing_n'] / x['n_pitches'].replace(0, np.nan),
        command_first_pitch_strike_rate = lambda x: x['command_first_pitch_strike_n'] / x['n_pitches'].replace(0, np.nan),

        pa_pitch_count_mean = lambda x: x['pa_pitch_count_sum'] / x['pa_total'].replace(0, np.nan),
        pa_strikeout_rate = lambda x: x['pa_strikeout_n'] / x['pa_total'].replace(0, np.nan),
        pa_walk_rate = lambda x: x['pa_walk_n'] / x['pa_total'].replace(0, np.nan),
        pa_hbp_rate = lambda x: x['pa_hbp_n'] / x['pa_total'].replace(0, np.nan),
        pa_hit_rate = lambda x: x['pa_hit_n'] / x['pa_total'].replace(0, np.nan),
        pa_hr_rate = lambda x: x['pa_hr_n'] / x['pa_total'].replace(0, np.nan),
        pa_single_rate = lambda x: x['pa_single_n'] / x['pa_total'].replace(0, np.nan),
        pa_xbh_rate = lambda x: x['pa_xbh_n'] / x['pa_total'].replace(0, np.nan),
        pa_avg_final_balls = lambda x: x['pa_final_balls_sum'] / x['pa_final_balls_n'].replace(0, np.nan),
        pa_avg_final_strikes = lambda x: x['pa_final_strikes_sum'] / x['pa_final_strikes_n'].replace(0, np.nan),
        pa_full_count_rate = lambda x: x['pa_full_count_n'] / x['pa_total'].replace(0, np.nan),

        last_inning_avg = lambda x: x['last_inning_inning'] / x['games_n'].replace(0, np.nan),
        last_inning_avg_velo = lambda x: x['last_inning_velo'] / x['games_n'].replace(0, np.nan),
        last_inning_ball_rate = lambda x: x['last_inning_ball'] / x['games_n'].replace(0, np.nan),
        last_inning_strike_rate = lambda x: x['last_inning_strike'] / x['games_n'].replace(0, np.nan),
        last_inning_avg_balls = lambda x: x['last_inning_balls'] / x['games_n'].replace(0, np.nan),
        last_inning_avg_strikes = lambda x: x['last_inning_strikes'] / x['games_n'].replace(0, np.nan),

        pitch_count_avg = lambda x: x['pitch_count_sum_src'] / x['games_n'].replace(0, np.nan),
        pitch_count_max = lambda x: x['pitch_count_max_src'],

        contact_hard_hit_rate = lambda x: x['contact_hard_hit_n'] / x['contact_n'].replace(0, np.nan),
        contact_gb_rate = lambda x: x['contact_gb_n'] / x['contact_n'].replace(0, np.nan),
        contact_fb_rate = lambda x: x['contact_fb_n'] / x['contact_n'].replace(0, np.nan),
        contact_ld_rate = lambda x: x['contact_ld_n'] / x['contact_n'].replace(0, np.nan),
        contact_avg_launch_speed = lambda x: x['contact_launch_speed_sum'] / x['contact_launch_speed_n'].replace(0, np.nan),
        contact_avg_launch_angle = lambda x: x['contact_launch_angle_sum'] / x['contact_launch_angle_n'].replace(0, np.nan),
    )

    fip_constant = 3.10
    rolled['pa_fip'] = (
        (13 * rolled['pa_hr_n'] + 3 * rolled['pa_fip_bb_n'] - 2 * rolled['pa_strikeout_n'])
        / rolled['pa_total'].replace(0, np.nan)
        + fip_constant
    )

    final_cols = [
        'stuff_start_speed_mean', 'stuff_start_speed_max', 'stuff_start_speed_std',
        'stuff_end_speed_mean', 'stuff_end_speed_max',
        'stuff_perceived_velo_mean', 'stuff_perceived_velo_max',
        'stuff_spin_rate_mean', 'stuff_spin_rate_max',
        'stuff_movement_magnitude_mean', 'stuff_movement_magnitude_max',
        'stuff_pfx_z_mean', 'stuff_pfx_z_max',
        'stuff_extension_mean', 'stuff_extension_max', 'stuff_extension_std',
        'stuff_speed_retention_mean',
        'command_in_play_rate', 'command_swinging_strike_rate',
        'command_plate_x_std', 'command_plate_z_normalized_std',
        'command_zone_rate', 'command_ball_rate', 'command_strike_rate', 'command_called_strike_rate',
        'command_chase_rate', 'command_zone_swing_rate', 'command_first_pitch_strike_rate',
        'pa_pitch_count_mean', 'pa_pitch_count_std',
        'pa_strikeout_rate', 'pa_walk_rate', 'pa_hbp_rate', 'pa_hit_rate', 'pa_hr_rate',
        'pa_single_rate', 'pa_xbh_rate', 'pa_avg_final_balls', 'pa_avg_final_strikes',
        'pa_full_count_rate', 'pa_fip',
        'last_inning_avg', 'last_inning_std', 'last_inning_avg_velo',
        'last_inning_ball_rate', 'last_inning_strike_rate',
        'last_inning_avg_balls', 'last_inning_avg_strikes', 'last_inning_outs',
        'pitch_count_avg', 'pitch_count_std', 'pitch_count_max',
        'contact_hard_hit_rate', 'contact_gb_rate', 'contact_fb_rate', 'contact_ld_rate',
        'contact_avg_launch_speed', 'contact_avg_launch_angle',
        # Sample-size denominators behind the rates above — kept as their own
        # features (not just consumed internally) so a model can learn to
        # trust a rate less when it's built from a thin window, an implicit
        # substitute for hand-designed shrinkage. See FEATURE_GLOSSARY.md.
        'n_pitches', 'pa_total', 'contact_n', 'games_n',
    ]

    rolled = rolled[key_cols + final_cols]

    return _prefix_stat_cols(rolled, prefix=_rolling_prefix('pitcher', window), key_cols=key_cols)


def build_pbp_pitcher_rolling_feats_all_roles(pbp: pd.DataFrame, window: str | int) -> pd.DataFrame:

    """Rolling equivalent of season_stats.py's build_pbp_pitcher_feats_all_roles:
    builds both role variants, tags each row with which role it represents, and
    stacks them so the result can be joined on ['pitcher_key_id', 'gamepk',
    'pitcher_role'] instead of ['pitcher_id', 'gamepk'] alone. Each role's rolling
    window only ever rolls forward that role's own prior games — a swingman's SP
    rolling stat and bullpen rolling stat never blend into each other.

    The bullpen half is pooled by team (entity_col='pitcher_team_id') rather
    than kept per individual pitcher_id — see build_pbp_pitcher_feats_all_roles
    for the same rationale (a specific reliever's identity isn't knowable
    pre-game). Both halves are renamed to a common pitcher_key_id column.
    """

    sp = (
        build_pbp_pitcher_rolling_feats(pbp, window=window, pitcher_role='sp')
        .rename(columns={'pitcher_id': 'pitcher_key_id'})
        .assign(pitcher_role='sp')
    )
    bullpen = (
        build_pbp_pitcher_rolling_feats(pbp, window=window, pitcher_role='bullpen', entity_col='pitcher_team_id')
        .rename(columns={'pitcher_team_id': 'pitcher_key_id'})
        .assign(pitcher_role='bullpen')
    )

    return pd.concat([sp, bullpen], ignore_index=True)


# --------------------------- BATTER PBP ROLLING STATS --------------------------- #
# Same 4 categories as season_stats.py's build_pbp_batter_feats (PA-outcome, plate
# discipline, in-play contact, foul contact, two-strike foul) — no max/std categories
# on the batter side, unlike pitcher's stuff/command.

PBP_BATTER_KEY_COLS = ['batter_id', 'gamepk', 'game_date', 'game_season']


def _batter_pa_outcome_per_game(pbp: pd.DataFrame) -> pd.DataFrame:
    group_cols = PBP_BATTER_KEY_COLS

    last_pitch_pbp = (
        pbp[pbp['pitch_number'] == pbp.groupby(['gamepk', 'play_id'])['pitch_number'].transform('max')]
        .reset_index(drop=True)
    )

    return (
        last_pitch_pbp
        .groupby(group_cols)
        .agg(
            pa_total=('play_result', 'count'),
            pa_pitch_count_sum=('pitch_number', 'sum'),
            pa_pitch_count_sumsq=('pitch_number', lambda s: (s ** 2).sum()),
            pa_strikeout_n=('play_result', lambda x: x.isin({"Strikeout", "Strikeout Double Play"}).sum()),
            pa_walk_n=('play_result', lambda x: x.isin({"Walk", "Intent Walk"}).sum()),
            pa_hbp_n=('play_result', lambda x: x.eq("Hit By Pitch").sum()),
            pa_hit_n=('play_result', lambda x: x.isin({"Single", "Double", "Triple", "Home Run"}).sum()),
            pa_hr_n=('play_result', lambda x: x.eq("Home Run").sum()),
            pa_single_n=('play_result', lambda x: x.eq("Single").sum()),
            pa_xbh_n=('play_result', lambda x: x.isin({"Double", "Triple", "Home Run"}).sum()),
            pa_final_balls_sum=('count_balls', 'sum'), pa_final_balls_n=('count_balls', 'count'),
            pa_final_strikes_sum=('count_strikes', 'sum'), pa_final_strikes_n=('count_strikes', 'count'),
            pa_full_count_n=('count_balls', lambda x: (
                (x == 3) & (last_pitch_pbp.loc[x.index, 'count_strikes'] == 2)
            ).sum()),
        )
        .reset_index()
    )


def _batter_plate_discipline_per_game(pbp: pd.DataFrame) -> pd.DataFrame:
    group_cols = PBP_BATTER_KEY_COLS

    all_pitches = (
        pbp
        .groupby(group_cols)
        .agg(
            chase_n=('is_chase', 'sum'),
            zone_swing_n=('is_zone_swing', 'sum'),
            swinging_strike_n=('is_swinging_strike', 'sum'),
            zone_n=('zone', lambda x: x.isin(range(1, 10)).sum()),
        )
        .reset_index()
    )
    n_pitches = pbp.groupby(group_cols).size().rename('n_pitches').reset_index()
    all_pitches = all_pitches.merge(n_pitches, on=group_cols, how='left')

    swings_only = pbp[pbp['is_swing']]
    swing_stats = (
        swings_only
        .groupby(group_cols)
        .agg(
            swing_n=('is_swinging_strike', 'count'),
            contact_n=('is_swinging_strike', lambda x: (~x).sum()),
        )
        .reset_index()
    )

    return all_pitches.merge(swing_stats, on=group_cols, how='left')


def _batter_in_play_contact_per_game(pbp: pd.DataFrame) -> pd.DataFrame:
    group_cols = PBP_BATTER_KEY_COLS

    contact_only = pbp[pbp['is_in_play'] == True]

    return (
        contact_only
        .groupby(group_cols)
        .agg(
            hard_hit_n=('launch_speed', lambda x: (x.dropna() >= 95).sum()),
            launch_speed_valid_n=('launch_speed', 'count'),
            sweet_spot_n=('launch_angle', lambda x: x.dropna().between(8, 32).sum()),
            launch_angle_valid_n=('launch_angle', 'count'),
            gb_n=('trajectory', lambda x: x.eq('Ground Ball').sum()),
            fb_n=('trajectory', lambda x: x.eq('Fly Ball').sum()),
            ld_n=('trajectory', lambda x: x.eq('Line Drive').sum()),
            contact_trajectory_n=('trajectory', 'count'),
            launch_speed_sum=('launch_speed', 'sum'),
            launch_angle_sum=('launch_angle', 'sum'),
        )
        .reset_index()
    )


def _batter_foul_contact_per_game(pbp: pd.DataFrame) -> pd.DataFrame:
    group_cols = PBP_BATTER_KEY_COLS

    swings_only = pbp[pbp['is_swing']]
    foul_stats = (
        swings_only
        .groupby(group_cols)
        .agg(
            foul_n=('pitch_call', lambda x: x.eq('Foul').sum()),
            foul_swing_n=('pitch_call', 'count'),
        )
        .reset_index()
    )

    contact_only = pbp[(pbp['pitch_call'] == 'Foul') | (pbp['is_in_play'] == True)]
    contact_foul_stats = (
        contact_only
        .groupby(group_cols)
        .agg(
            contact_foul_n=('pitch_call', lambda x: x.eq('Foul').sum()),
            foul_or_inplay_n=('pitch_call', 'count'),
        )
        .reset_index()
    )

    return foul_stats.merge(contact_foul_stats, on=group_cols, how='left')


def _batter_two_strike_foul_per_game(pbp: pd.DataFrame) -> pd.DataFrame:
    group_cols = PBP_BATTER_KEY_COLS

    two_strike_swings = pbp[(pbp['count_strikes'] == 2) & (pbp['is_swing'])]

    return (
        two_strike_swings
        .groupby(group_cols)
        .agg(
            two_strike_foul_n=('pitch_call', lambda x: x.eq('Foul').sum()),
            two_strike_swing_n=('pitch_call', 'count'),
        )
        .reset_index()
    )


BATTER_PBP_FILL_ZERO_COLS = [
    'swing_n', 'contact_n',
    'hard_hit_n', 'launch_speed_valid_n', 'sweet_spot_n', 'launch_angle_valid_n',
    'gb_n', 'fb_n', 'ld_n', 'contact_trajectory_n', 'launch_speed_sum', 'launch_angle_sum',
    'foul_n', 'foul_swing_n', 'contact_foul_n', 'foul_or_inplay_n',
    'two_strike_foul_n', 'two_strike_swing_n',
]


def _batter_pbp_per_game(pbp: pd.DataFrame) -> pd.DataFrame:
    key_cols = PBP_BATTER_KEY_COLS

    df = _batter_pa_outcome_per_game(pbp)
    df = df.merge(_batter_plate_discipline_per_game(pbp), on=key_cols, how='left')
    df = df.merge(_batter_in_play_contact_per_game(pbp), on=key_cols, how='left')
    df = df.merge(_batter_foul_contact_per_game(pbp), on=key_cols, how='left')
    df = df.merge(_batter_two_strike_foul_per_game(pbp), on=key_cols, how='left')

    df[BATTER_PBP_FILL_ZERO_COLS] = df[BATTER_PBP_FILL_ZERO_COLS].astype(float).fillna(0.0)

    return df


def build_pbp_batter_rolling_feats(pbp: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """Rolling equivalent of season_stats.py's build_pbp_batter_feats. Same category
    boundaries and formulas, computed at per-game grain and rolled forward."""

    _validate_window(window)

    key_cols = PBP_BATTER_KEY_COLS
    entity_col = 'batter_id'
    per_game = _batter_pbp_per_game(pbp)

    sum_cols = [
        'pa_total', 'pa_pitch_count_sum', 'pa_strikeout_n', 'pa_walk_n', 'pa_hbp_n', 'pa_hit_n',
        'pa_hr_n', 'pa_single_n', 'pa_xbh_n', 'pa_final_balls_sum', 'pa_final_balls_n',
        'pa_final_strikes_sum', 'pa_final_strikes_n', 'pa_full_count_n',
        'chase_n', 'zone_swing_n', 'swinging_strike_n', 'zone_n', 'n_pitches',
        'swing_n', 'contact_n',
        'hard_hit_n', 'launch_speed_valid_n', 'sweet_spot_n', 'launch_angle_valid_n',
        'gb_n', 'fb_n', 'ld_n', 'contact_trajectory_n', 'launch_speed_sum', 'launch_angle_sum',
        'foul_n', 'foul_swing_n', 'contact_foul_n', 'foul_or_inplay_n',
        'two_strike_foul_n', 'two_strike_swing_n',
    ]

    rolled = _rolling_sum(per_game, entity_col=entity_col, cols=sum_cols, window=window)

    std_df = _rolling_pooled_std(
        per_game, entity_col, 'pa_total', 'pa_pitch_count_sum', 'pa_pitch_count_sumsq',
        window, 'pa_pitch_count_std',
    )
    rolled = rolled.merge(std_df[key_cols + ['pa_pitch_count_std']], on=key_cols, how='left')

    rolled = rolled.assign(
        pa_pitch_count_mean = lambda x: x['pa_pitch_count_sum'] / x['pa_total'].replace(0, np.nan),
        pa_strikeout_rate = lambda x: x['pa_strikeout_n'] / x['pa_total'].replace(0, np.nan),
        pa_walk_rate = lambda x: x['pa_walk_n'] / x['pa_total'].replace(0, np.nan),
        pa_hbp_rate = lambda x: x['pa_hbp_n'] / x['pa_total'].replace(0, np.nan),
        pa_hit_rate = lambda x: x['pa_hit_n'] / x['pa_total'].replace(0, np.nan),
        pa_hr_rate = lambda x: x['pa_hr_n'] / x['pa_total'].replace(0, np.nan),
        pa_single_rate = lambda x: x['pa_single_n'] / x['pa_total'].replace(0, np.nan),
        pa_xbh_rate = lambda x: x['pa_xbh_n'] / x['pa_total'].replace(0, np.nan),
        pa_avg_final_balls = lambda x: x['pa_final_balls_sum'] / x['pa_final_balls_n'].replace(0, np.nan),
        pa_avg_final_strikes = lambda x: x['pa_final_strikes_sum'] / x['pa_final_strikes_n'].replace(0, np.nan),
        pa_full_count_rate = lambda x: x['pa_full_count_n'] / x['pa_total'].replace(0, np.nan),

        o_swing_rate = lambda x: x['chase_n'] / x['n_pitches'].replace(0, np.nan),
        z_swing_rate = lambda x: x['zone_swing_n'] / x['n_pitches'].replace(0, np.nan),
        swinging_strike_rate = lambda x: x['swinging_strike_n'] / x['n_pitches'].replace(0, np.nan),
        zone_rate = lambda x: x['zone_n'] / x['n_pitches'].replace(0, np.nan),
        contact_rate = lambda x: x['contact_n'] / x['swing_n'].replace(0, np.nan),

        contact_hard_hit_rate = lambda x: x['hard_hit_n'] / x['launch_speed_valid_n'].replace(0, np.nan),
        contact_sweet_spot_rate = lambda x: x['sweet_spot_n'] / x['launch_angle_valid_n'].replace(0, np.nan),
        contact_gb_rate = lambda x: x['gb_n'] / x['contact_trajectory_n'].replace(0, np.nan),
        contact_fb_rate = lambda x: x['fb_n'] / x['contact_trajectory_n'].replace(0, np.nan),
        contact_ld_rate = lambda x: x['ld_n'] / x['contact_trajectory_n'].replace(0, np.nan),
        contact_avg_launch_speed = lambda x: x['launch_speed_sum'] / x['launch_speed_valid_n'].replace(0, np.nan),
        contact_avg_launch_angle = lambda x: x['launch_angle_sum'] / x['launch_angle_valid_n'].replace(0, np.nan),

        foul_rate = lambda x: x['foul_n'] / x['foul_swing_n'].replace(0, np.nan),
        contact_foul_rate = lambda x: x['contact_foul_n'] / x['foul_or_inplay_n'].replace(0, np.nan),

        two_strike_foul_rate = lambda x: x['two_strike_foul_n'] / x['two_strike_swing_n'].replace(0, np.nan),
    )

    final_cols = [
        'pa_pitch_count_mean', 'pa_pitch_count_std',
        'pa_strikeout_rate', 'pa_walk_rate', 'pa_hbp_rate', 'pa_hit_rate', 'pa_hr_rate',
        'pa_single_rate', 'pa_xbh_rate', 'pa_avg_final_balls', 'pa_avg_final_strikes',
        'pa_full_count_rate',
        'o_swing_rate', 'z_swing_rate', 'swinging_strike_rate', 'zone_rate', 'contact_rate',
        'contact_hard_hit_rate', 'contact_sweet_spot_rate', 'contact_gb_rate', 'contact_fb_rate',
        'contact_ld_rate', 'contact_avg_launch_speed', 'contact_avg_launch_angle',
        'foul_rate', 'contact_foul_rate',
        'two_strike_foul_rate',
        # Sample-size denominators behind the rates above — kept as their own
        # features (not just consumed internally) so a model can learn to
        # trust a rate less when it's built from a thin window, an implicit
        # substitute for hand-designed shrinkage. See FEATURE_GLOSSARY.md.
        'n_pitches', 'pa_total', 'swing_n', 'contact_trajectory_n',
        'foul_swing_n', 'foul_or_inplay_n', 'two_strike_swing_n',
    ]

    rolled = rolled[key_cols + final_cols]

    return _prefix_stat_cols(rolled, prefix=_rolling_prefix('batter', window), key_cols=key_cols)


# ------------------- TEAM-LEVEL BATTER STRIKEOUT ROLLING FEATS ------------------- #
# k_predictor v2's opposing-lineup rolling K rate. Same pooling shape as
# build_pitcher_rolling_stats_all_roles' bullpen case: a lineup has 9 distinct
# batter identities per game, so per-batter-per-game rows must be collapsed into
# one team-game row BEFORE rolling (_rolling_sum's .transform() preserves one
# output row per input row, not per (team, game)). Pooled to STARTING LINEUP
# batters only (those with a batting_order that game) rather than everyone who
# appeared — a bench/pinch-hit PA isn't part of "the lineup this pitcher expects
# to face." The resulting table is role-agnostic (one row per (team_id, gamepk)
# regardless of who they played), so it doubles as both "the opposing team's
# rolling K rate" (merge on batter_team_id) and "the pitcher's own team's rolling
# K rate" (merge on pitcher_team_id) without needing a second function.

def build_team_batter_strikeout_rolling_feats(
    pbp: pd.DataFrame, batter_boxscore: pd.DataFrame, window: str | int,
) -> pd.DataFrame:
    _validate_window(window)

    starters = _create_batting_order(batter_boxscore)[['gamepk', 'batter_id']]
    per_batter_game = _batter_pa_outcome_per_game(pbp)[PBP_BATTER_KEY_COLS + ['pa_total', 'pa_strikeout_n']]
    team_lookup = pbp[['batter_id', 'gamepk', 'batter_team_id']].drop_duplicates()

    starters_pa = per_batter_game.merge(starters, on=['gamepk', 'batter_id'], how='inner')
    starters_pa = starters_pa.merge(team_lookup, on=['batter_id', 'gamepk'], how='left')

    team_per_game = (
        starters_pa
        .groupby(['batter_team_id', 'gamepk', 'game_date', 'game_season'])[['pa_total', 'pa_strikeout_n']]
        .sum()
        .reset_index()
    )

    key_cols = ['batter_team_id', 'gamepk', 'game_date', 'game_season']
    rolled = _rolling_sum(team_per_game, entity_col='batter_team_id', cols=['pa_total', 'pa_strikeout_n'], window=window)
    rolled = rolled.assign(
        pa_strikeout_rate=lambda x: x['pa_strikeout_n'] / x['pa_total'].replace(0, np.nan)
    )

    final_cols = ['pa_total', 'pa_strikeout_n', 'pa_strikeout_rate']
    rolled = rolled[key_cols + final_cols]

    return _prefix_stat_cols(rolled, prefix=_rolling_prefix('team', window), key_cols=key_cols)
