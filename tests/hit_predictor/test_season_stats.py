import numpy as np
import pandas as pd

import inspect

from models.hit_predictor.processing.features.season_stats import (
    _create_batter_foul_contact_stats,
    _create_batter_in_play_contact_stats,
    _create_batter_pa_outcome_stats,
    _create_batter_plate_discipline_stats,
    _create_batter_two_strike_foul_stats,
    _create_pitcher_contact_quality_stats,
    _shift_to_last_season,
    build_batter_stats,
    build_pbp_batter_feats,
)


def test_contact_quality_stats_returns_pitcher_id_and_game_season_as_columns():
    """_create_pitcher_contact_quality_stats groups by pitcher_id/game_season but
    never resets the index, so downstream merges in build_pbp_pitcher_feats fail
    with a KeyError since those keys only exist as index levels, not columns."""

    pbp = pd.DataFrame({
        'pitcher_id': ['1', '1', '2'],
        'game_season': [2023, 2023, 2023],
        'is_in_play': [True, True, True],
        'hardness': ['Hard', 'Medium', 'Hard'],
        'trajectory': ['Line Drive', 'Fly Ball', 'Ground Ball'],
        'launch_speed': [102.0, 90.0, 98.0],
        'launch_angle': [15.0, 25.0, -5.0],
    })

    result = _create_pitcher_contact_quality_stats(pbp)

    assert 'pitcher_id' in result.columns
    assert 'game_season' in result.columns


def _make_batter_boxscore_row(**overrides):
    row = {
        'personId': '1',
        'game_season': 2023,
        'ab': 10,
        'h': 3,
        'k': 2,
        'bb': 1,
        'hr': 1,
        'plate_appearances': 11,
        'total_bases_from_h': 7,
    }
    row.update(overrides)
    return row


def test_build_batter_stats_shifts_game_season_by_exactly_one():
    """build_batter_stats previously double-shifted: _create_boxscore_batter_stats
    shifted internally, then build_batter_stats shifted again, landing game_season
    two years ahead instead of one."""

    batter_boxscore = pd.DataFrame([_make_batter_boxscore_row(game_season=2023)])

    result = build_batter_stats(batter_boxscore)

    assert result['game_season'].tolist() == [2024]


def test_build_batter_stats_rate_stats_are_nan_not_crash_on_zero_ab():
    """A batter with AB=0 (e.g. pinch-runner-only appearance, or a walk-only game)
    must not raise a ZeroDivisionError-equivalent; AVG/SLG/ISO/BABIP should be NaN."""

    batter_boxscore = pd.DataFrame([_make_batter_boxscore_row(
        personId='2', ab=0, h=0, k=0, bb=1, hr=0, plate_appearances=1, total_bases_from_h=0,
    )])

    result = build_batter_stats(batter_boxscore)

    row = result.iloc[0]
    assert np.isnan(row['batter_last_season_ba'])
    assert np.isnan(row['batter_last_season_slg'])
    assert np.isnan(row['batter_last_season_iso'])
    assert np.isnan(row['batter_last_season_babip'])


def test_build_batter_stats_babip_is_nan_on_zero_babip_denominator():
    """BABIP = (H-HR)/(AB-K-HR). A batter who struck out on every AB has a zero
    denominator even though AB itself is nonzero, so this must be guarded
    independently of the AB==0 case above."""

    batter_boxscore = pd.DataFrame([_make_batter_boxscore_row(
        personId='3', ab=2, h=0, k=2, bb=0, hr=0, plate_appearances=2, total_bases_from_h=0,
    )])

    result = build_batter_stats(batter_boxscore)

    row = result.iloc[0]
    assert row['batter_last_season_ba'] == 0.0
    assert np.isnan(row['batter_last_season_babip'])


def test_build_batter_stats_computes_rate_stats_correctly():
    batter_boxscore = pd.DataFrame([_make_batter_boxscore_row()])

    result = build_batter_stats(batter_boxscore)

    row = result.iloc[0]
    assert row['batter_last_season_ba'] == 0.3
    assert row['batter_last_season_slg'] == 0.7
    assert row['batter_last_season_iso'] == 0.4
    assert row['batter_last_season_babip'] == 0.29


def test_create_batter_pa_outcome_stats_returns_batter_id_and_game_season_as_columns():
    pbp = pd.DataFrame([
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 1,
         'play_result': None, 'count_balls': 0, 'count_strikes': 0},
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 2,
         'play_result': 'Single', 'count_balls': 1, 'count_strikes': 1},
    ])

    result = _create_batter_pa_outcome_stats(pbp)

    assert 'batter_id' in result.columns
    assert 'game_season' in result.columns


def test_create_batter_pa_outcome_stats_excludes_fip_columns():
    """FIP is a pitching run-prevention stat and is meaningless attributed to a
    batter — the batter PA outcome function must not carry over the pitcher
    version's fip_k/fip_bb/fip_hr/fip columns."""

    pbp = pd.DataFrame([
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 1,
         'play_result': 'Home Run', 'count_balls': 1, 'count_strikes': 1},
    ])

    result = _create_batter_pa_outcome_stats(pbp)

    fip_cols = [c for c in result.columns if 'fip' in c]
    assert fip_cols == []


def test_create_batter_pa_outcome_stats_computes_rates_from_last_pitch_of_each_pa():
    """Only the final pitch of each PA (by pitch_number within gamepk/play_id)
    should count toward the outcome rates — earlier pitches in the same PA must
    be excluded from the denominator."""

    pbp = pd.DataFrame([
        # PA 1: 2 pitches, final pitch is a Single
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 1,
         'play_result': None, 'count_balls': 0, 'count_strikes': 0},
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 2,
         'play_result': 'Single', 'count_balls': 1, 'count_strikes': 1},
        # PA 2: 1 pitch, Strikeout
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 2, 'pitch_number': 1,
         'play_result': 'Strikeout', 'count_balls': 0, 'count_strikes': 2},
    ])

    result = _create_batter_pa_outcome_stats(pbp)

    row = result.iloc[0]
    assert row['batter_season_pa_total'] == 2
    assert row['batter_season_pa_hit_rate'] == 0.5
    assert row['batter_season_pa_single_rate'] == 0.5
    assert row['batter_season_pa_strikeout_rate'] == 0.5
    assert row['batter_season_pa_walk_rate'] == 0.0


def _make_plate_discipline_pbp():
    return pd.DataFrame([
        # ball outside zone, taken (not a chase)
        {'batter_id': '1', 'game_season': 2023, 'zone': 12,
         'is_chase': False, 'is_zone_swing': False, 'is_swinging_strike': False, 'is_swing': False},
        # called strike inside zone, taken (not a zone swing)
        {'batter_id': '1', 'game_season': 2023, 'zone': 5,
         'is_chase': False, 'is_zone_swing': False, 'is_swinging_strike': False, 'is_swing': False},
        # swinging strike outside zone: a chase AND a whiff
        {'batter_id': '1', 'game_season': 2023, 'zone': 13,
         'is_chase': True, 'is_zone_swing': False, 'is_swinging_strike': True, 'is_swing': True},
        # foul inside zone: a zone swing WITH contact (not a whiff)
        {'batter_id': '1', 'game_season': 2023, 'zone': 4,
         'is_chase': False, 'is_zone_swing': True, 'is_swinging_strike': False, 'is_swing': True},
    ])


def test_create_batter_plate_discipline_stats_returns_batter_id_and_game_season_as_columns():
    result = _create_batter_plate_discipline_stats(_make_plate_discipline_pbp())

    assert 'batter_id' in result.columns
    assert 'game_season' in result.columns


def test_create_batter_plate_discipline_stats_excludes_stuff_columns():
    """stuff_* (pitch physics: velocity, spin, movement) describes the pitch
    thrown, not the batter — only the command_*-equivalent plate-discipline
    metrics should be adapted for batters."""

    result = _create_batter_plate_discipline_stats(_make_plate_discipline_pbp())

    stuff_cols = [c for c in result.columns if 'stuff' in c]
    assert stuff_cols == []


def test_create_batter_plate_discipline_stats_rates_use_all_pitches_denominator():
    """O-Swing%, Z-Swing%, SwStr%, and Zone% are all defined over every pitch
    seen, not just swings."""

    result = _create_batter_plate_discipline_stats(_make_plate_discipline_pbp())

    row = result.iloc[0]
    assert row['batter_season_o_swing_rate'] == 0.25
    assert row['batter_season_z_swing_rate'] == 0.25
    assert row['batter_season_swinging_strike_rate'] == 0.25
    assert row['batter_season_zone_rate'] == 0.5


def test_create_batter_plate_discipline_stats_contact_rate_uses_swings_only_denominator():
    """Contact% is contact-made / swings — distinct from SwStr%, which divides
    by all pitches. Of the 2 swings in the fixture (1 whiff, 1 foul), exactly
    1 made contact."""

    result = _create_batter_plate_discipline_stats(_make_plate_discipline_pbp())

    row = result.iloc[0]
    assert row['batter_season_contact_rate'] == 0.5


def test_shift_to_last_season_renames_role_prefixed_columns():
    """_shift_to_last_season only renamed columns starting with the literal
    'season_' prefix, so role-prefixed columns like pitcher_season_foo or
    batter_season_bar (which don't start with 'season_') were silently left
    unrenamed after the shift, even though they now represent last season's
    value. The rename should catch 'season_' anywhere in the column name."""

    df = pd.DataFrame({
        'pitcher_id': ['1'],
        'game_season': [2023],
        'pitcher_season_foo': [0.5],
    })

    result = _shift_to_last_season(df)

    assert result['game_season'].tolist() == [2024]
    assert 'pitcher_last_season_foo' in result.columns
    assert 'pitcher_season_foo' not in result.columns


def _make_in_play_contact_pbp():
    return pd.DataFrame([
        {'batter_id': '1', 'game_season': 2023, 'is_in_play': True,
         'launch_speed': 100.0, 'launch_angle': 20.0, 'trajectory': 'Line Drive'},
        {'batter_id': '1', 'game_season': 2023, 'is_in_play': True,
         'launch_speed': 80.0, 'launch_angle': 5.0, 'trajectory': 'Ground Ball'},
        {'batter_id': '1', 'game_season': 2023, 'is_in_play': True,
         'launch_speed': np.nan, 'launch_angle': np.nan, 'trajectory': 'Fly Ball'},
        {'batter_id': '1', 'game_season': 2023, 'is_in_play': True,
         'launch_speed': np.nan, 'launch_angle': np.nan, 'trajectory': 'Fly Ball'},
        # not in play at all — must be excluded from every stat
        {'batter_id': '1', 'game_season': 2023, 'is_in_play': False,
         'launch_speed': 999.0, 'launch_angle': 999.0, 'trajectory': 'Line Drive'},
    ])


def test_create_batter_in_play_contact_stats_returns_batter_id_and_game_season_as_columns():
    pbp = pd.DataFrame({
        'batter_id': ['1'],
        'game_season': [2023],
        'is_in_play': [True],
        'launch_speed': [100.0],
        'launch_angle': [20.0],
        'trajectory': ['Line Drive'],
    })

    result = _create_batter_in_play_contact_stats(pbp)

    assert 'batter_id' in result.columns
    assert 'game_season' in result.columns


def test_create_batter_in_play_contact_stats_filters_to_in_play_pitches_only():
    """A non-in-play row is given adversarial trajectory='Ground Ball' data on
    purpose — if the is_in_play filter were missing, gb_rate/ld_rate would be
    diluted (0.5/0.5) instead of correctly reflecting only the in-play row."""

    pbp = pd.DataFrame([
        {'batter_id': '1', 'game_season': 2023, 'is_in_play': True,
         'launch_speed': 100.0, 'launch_angle': 20.0, 'trajectory': 'Line Drive'},
        {'batter_id': '1', 'game_season': 2023, 'is_in_play': False,
         'launch_speed': np.nan, 'launch_angle': np.nan, 'trajectory': 'Ground Ball'},
    ])

    result = _create_batter_in_play_contact_stats(pbp)

    row = result.iloc[0]
    assert row['batter_season_contact_gb_rate'] == 0.0
    assert row['batter_season_contact_ld_rate'] == 1.0


def test_create_batter_in_play_contact_stats_hard_hit_and_sweet_spot_exclude_nulls_from_denominator():
    """2 of 4 in-play rows have null launch_speed/launch_angle (unclassified
    batted balls). A naive (launch_speed >= 95).mean() over all 4 rows would
    treat those nulls as False and understate hard_hit_rate as 0.25; the
    correct null-excluded rate over the 2 classified rows is 0.5."""

    result = _create_batter_in_play_contact_stats(_make_in_play_contact_pbp())

    row = result.iloc[0]
    assert row['batter_season_contact_hard_hit_rate'] == 0.5
    assert row['batter_season_contact_sweet_spot_rate'] == 0.5


def test_create_batter_in_play_contact_stats_computes_trajectory_rates_and_averages():
    result = _create_batter_in_play_contact_stats(_make_in_play_contact_pbp())

    row = result.iloc[0]
    assert row['batter_season_contact_gb_rate'] == 0.25
    assert row['batter_season_contact_fb_rate'] == 0.5
    assert row['batter_season_contact_ld_rate'] == 0.25
    assert row['batter_season_contact_avg_launch_speed'] == 90.0
    assert row['batter_season_contact_avg_launch_angle'] == 12.5


def _make_foul_contact_pbp():
    return pd.DataFrame([
        # foul, not in play — a swing that stayed foul
        {'batter_id': '1', 'game_season': 2023, 'is_swing': True,
         'pitch_call': 'Foul', 'is_in_play': False},
        # in play — a swing that got put in play
        {'batter_id': '1', 'game_season': 2023, 'is_swing': True,
         'pitch_call': 'In play, out(s)', 'is_in_play': True},
        # whiff — a swing with no contact at all
        {'batter_id': '1', 'game_season': 2023, 'is_swing': True,
         'pitch_call': 'Swinging Strike', 'is_in_play': False},
        # Foul Tip — out of scope for this story; must not count as a 'Foul'
        # match in either numerator, and must not enter the contact_only denominator
        {'batter_id': '1', 'game_season': 2023, 'is_swing': True,
         'pitch_call': 'Foul Tip', 'is_in_play': False},
        # taken pitch — not a swing at all, excluded from every denominator
        {'batter_id': '1', 'game_season': 2023, 'is_swing': False,
         'pitch_call': 'Ball', 'is_in_play': False},
    ])


def test_create_batter_foul_contact_stats_returns_batter_id_and_game_season_as_columns():
    result = _create_batter_foul_contact_stats(_make_foul_contact_pbp())

    assert 'batter_id' in result.columns
    assert 'game_season' in result.columns


def test_create_batter_foul_contact_stats_foul_rate_uses_swings_only_denominator():
    """foul_rate = fouls / swings. Of the 4 swings in the fixture (foul, in-play,
    whiff, foul-tip), exactly 1 is a literal 'Foul' -> 1/4 = 0.25."""

    result = _create_batter_foul_contact_stats(_make_foul_contact_pbp())

    row = result.iloc[0]
    assert row['batter_season_foul_rate'] == 0.25


def test_create_batter_foul_contact_stats_contact_foul_rate_excludes_whiffs_from_denominator():
    """contact_foul_rate = fouls / (fouls + in-play), isolating contact composition
    from swing-and-miss tendency. Of the 2 contact events (foul, in-play — the
    whiff and foul-tip are excluded entirely), exactly 1 is foul -> 1/2 = 0.5.
    This must differ from foul_rate (0.25) — proving the denominators are scoped
    independently, not just two views of the same number."""

    result = _create_batter_foul_contact_stats(_make_foul_contact_pbp())

    row = result.iloc[0]
    assert row['batter_season_contact_foul_rate'] == 0.5
    assert row['batter_season_contact_foul_rate'] != row['batter_season_foul_rate']


def _make_two_strike_foul_pbp():
    return pd.DataFrame([
        # two-strike swings (4 of them — the correct denominator)
        {'batter_id': '1', 'game_season': 2023, 'count_strikes': 2, 'is_swing': True,
         'pitch_call': 'Foul'},
        {'batter_id': '1', 'game_season': 2023, 'count_strikes': 2, 'is_swing': True,
         'pitch_call': 'Swinging Strike'},
        {'batter_id': '1', 'game_season': 2023, 'count_strikes': 2, 'is_swing': True,
         'pitch_call': 'In play, out(s)'},
        {'batter_id': '1', 'game_season': 2023, 'count_strikes': 2, 'is_swing': True,
         'pitch_call': 'In play, no out'},
        # two strikes but not a swing — excluded
        {'batter_id': '1', 'game_season': 2023, 'count_strikes': 2, 'is_swing': False,
         'pitch_call': 'Ball'},
        # fouls at other counts — must NOT leak into the two-strike denominator/numerator
        {'batter_id': '1', 'game_season': 2023, 'count_strikes': 1, 'is_swing': True,
         'pitch_call': 'Foul'},
        {'batter_id': '1', 'game_season': 2023, 'count_strikes': 0, 'is_swing': True,
         'pitch_call': 'Foul'},
    ])


def test_create_batter_two_strike_foul_stats_returns_batter_id_and_game_season_as_columns():
    result = _create_batter_two_strike_foul_stats(_make_two_strike_foul_pbp())

    assert 'batter_id' in result.columns
    assert 'game_season' in result.columns


def test_create_batter_two_strike_foul_stats_filters_to_two_strike_counts_before_grouping():
    """4 two-strike swings, 1 of which is foul -> 0.25. If the count_strikes==2
    pre-filter were missing (or used all swings regardless of count), the two
    extra fouls at 0- and 1-strike counts would inflate this to 3/6 = 0.5 —
    the two values must differ."""

    result = _create_batter_two_strike_foul_stats(_make_two_strike_foul_pbp())

    row = result.iloc[0]
    assert row['batter_season_two_strike_foul_rate'] == 0.25


def _make_full_batter_pbp():
    return pd.DataFrame([
        # PA 1, pitch 1: taken ball outside zone
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 1,
         'play_result': None, 'count_balls': 0, 'count_strikes': 0,
         'zone': 12, 'is_chase': False, 'is_zone_swing': False, 'is_swinging_strike': False,
         'is_swing': False, 'pitch_call': 'Ball', 'is_in_play': False,
         'launch_speed': np.nan, 'launch_angle': np.nan, 'trajectory': np.nan},
        # PA 1, pitch 2: in-play single
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 2,
         'play_result': 'Single', 'count_balls': 1, 'count_strikes': 1,
         'zone': 5, 'is_chase': False, 'is_zone_swing': True, 'is_swinging_strike': False,
         'is_swing': True, 'pitch_call': 'In play, no out', 'is_in_play': True,
         'launch_speed': 100.0, 'launch_angle': 15.0, 'trajectory': 'Line Drive'},
        # PA 2, pitch 1: two-strike foul, then strikeout on pitch 2
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 2, 'pitch_number': 1,
         'play_result': None, 'count_balls': 0, 'count_strikes': 2,
         'zone': 5, 'is_chase': False, 'is_zone_swing': True, 'is_swinging_strike': False,
         'is_swing': True, 'pitch_call': 'Foul', 'is_in_play': False,
         'launch_speed': np.nan, 'launch_angle': np.nan, 'trajectory': np.nan},
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 2, 'pitch_number': 2,
         'play_result': 'Strikeout', 'count_balls': 0, 'count_strikes': 2,
         'zone': 13, 'is_chase': True, 'is_zone_swing': False, 'is_swinging_strike': True,
         'is_swing': True, 'pitch_call': 'Swinging Strike', 'is_in_play': False,
         'launch_speed': np.nan, 'launch_angle': np.nan, 'trajectory': np.nan},
    ])


def test_build_pbp_batter_feats_shifts_game_season_and_merges_all_substats():
    result = build_pbp_batter_feats(_make_full_batter_pbp())

    row = result.iloc[0]
    # game_season shifted +1 for next-season lookup
    assert row['game_season'] == 2024
    # one representative column from each of the 5 merged sub-functions, correctly
    # renamed to last_season_* now that Part A1's _shift_to_last_season fix is in place
    assert 'batter_last_season_pa_hit_rate' in result.columns
    assert 'batter_last_season_o_swing_rate' in result.columns
    assert 'batter_last_season_contact_hard_hit_rate' in result.columns
    assert 'batter_last_season_foul_rate' in result.columns
    assert 'batter_last_season_two_strike_foul_rate' in result.columns


def test_build_pbp_batter_feats_has_no_role_parameter():
    """Unlike build_pbp_pitcher_feats, there is no batter-side role filter —
    calling with just pbp must work with no second positional/keyword arg."""

    sig = inspect.signature(build_pbp_batter_feats)
    assert list(sig.parameters) == ['pbp']
