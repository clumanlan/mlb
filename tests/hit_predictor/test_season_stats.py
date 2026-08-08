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
    _create_pitcher_last_inning_stats,
    _create_pitcher_pa_outcome_stats,
    _pivot_by_hand,
    _shift_to_last_season,
    build_batter_stats,
    build_league_handedness_stats,
    build_pbp_batter_feats,
    build_pbp_batter_feats_by_pitcher_hand,
    _pitcher_role_lookup,
    build_pbp_pitcher_feats,
    build_pbp_pitcher_feats_all_roles,
    build_pbp_pitcher_feats_by_batter_hand,
    build_pitcher_stats,
    build_pitcher_stats_all_roles,
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


def _make_pitcher_boxscore_row(**overrides):
    row = {
        'personId': '1',
        'game_season': 2023,
        'h': 5, 'r': 2, 'er': 2, 'bb': 1, 'hr': 1, 'k': 8, 'p': 90, 's': 60, 'ip': 6.0,
    }
    row.update(overrides)
    return row


def test_build_pitcher_stats_preserves_person_id_as_a_column():
    """build_pitcher_stats groups by personId but previously passed
    key_cols=['pitcher_id', 'game_season'] to _prefix_stat_cols — since
    'pitcher_id' doesn't exist yet at that point, personId itself (the only
    actual join key) fell through and got prefixed away into
    pitcher_season_personId, leaving no usable key for downstream merges."""

    pitcher_boxscore = pd.DataFrame([_make_pitcher_boxscore_row()])

    result = build_pitcher_stats(pitcher_boxscore)

    assert 'personId' in result.columns
    assert 'pitcher_season_personId' not in result.columns
    assert 'pitcher_last_season_personId' not in result.columns


def _make_last_inning_pbp_row(**overrides):
    row = {
        'pitcher_id': '1',
        'gamepk': '1',
        'game_season': 2023,
        'play_id': 1,
        'pitch_number': 1,
        'inning': 9,
        'start_speed': 90.0,
        'is_ball': False,
        'is_strike': True,
        'count_balls': 0,
        'count_strikes': 2,
        'count_outs': 3,
    }
    row.update(overrides)
    return row


def test_create_pitcher_last_inning_stats_uses_true_last_pitch_of_final_pa():
    """The pitcher's final PA of the game (play_id=2) has two pitches: an
    opening ball at 95mph, then the actual final pitch — a 90mph strikeout
    pitch. Selecting via groupby(...)['play_id'].idxmax() alone (without
    first collapsing each PA to its true last pitch) ties-break to whichever
    row appears first in the frame, silently picking the *opening* pitch of
    the last PA instead of the true last pitch — velo 95 and ball_rate 1.0
    instead of the correct velo 90 and ball_rate 0.0."""

    pbp = pd.DataFrame([
        # earlier PA, not the last one
        _make_last_inning_pbp_row(play_id=1, pitch_number=1, inning=8, start_speed=88.0),
        # final PA: opening pitch (a ball) ...
        _make_last_inning_pbp_row(play_id=2, pitch_number=1, inning=9, start_speed=95.0,
                                   is_ball=True, is_strike=False, count_balls=0, count_strikes=0),
        # ... then the true last pitch (a called strikeout)
        _make_last_inning_pbp_row(play_id=2, pitch_number=2, inning=9, start_speed=90.0,
                                   is_ball=False, is_strike=True, count_balls=0, count_strikes=2, count_outs=3),
    ])

    result = _create_pitcher_last_inning_stats(pbp)

    row = result.iloc[0]
    assert row['pitcher_season_avg_last_inning_velo'] == 90.0
    assert row['pitcher_season_last_inning_ball_rate'] == 0.0
    assert row['pitcher_season_last_inning_strike_rate'] == 1.0


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


# --------------------------- HANDEDNESS SPLIT: BATTER VS PITCHER HAND --------------------------- #

def _make_hand_split_pa_pbp():
    return pd.DataFrame([
        # vs LHP: 2 PAs, both singles -> hit_rate 1.0
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 1,
         'play_result': 'Single', 'count_balls': 1, 'count_strikes': 1, 'pitcher_throw_hand': 'L'},
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 2, 'pitch_number': 1,
         'play_result': 'Single', 'count_balls': 1, 'count_strikes': 1, 'pitcher_throw_hand': 'L'},
        # vs RHP: 1 PA, a strikeout -> hit_rate 0.0
        {'batter_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 3, 'pitch_number': 1,
         'play_result': 'Strikeout', 'count_balls': 0, 'count_strikes': 2, 'pitcher_throw_hand': 'R'},
    ])


def test_create_batter_pa_outcome_stats_extra_group_cols_splits_rates_by_hand():
    """extra_group_cols must produce one row per (batter_id, game_season, hand) with
    rates computed independently per hand, not pooled across both."""

    result = _create_batter_pa_outcome_stats(
        _make_hand_split_pa_pbp(), extra_group_cols=['pitcher_throw_hand']
    )

    assert set(result.columns) >= {'batter_id', 'game_season', 'pitcher_throw_hand'}
    assert len(result) == 2

    vs_l = result[result['pitcher_throw_hand'] == 'L'].iloc[0]
    vs_r = result[result['pitcher_throw_hand'] == 'R'].iloc[0]
    assert vs_l['batter_season_pa_total'] == 2
    assert vs_l['batter_season_pa_hit_rate'] == 1.0
    assert vs_r['batter_season_pa_total'] == 1
    assert vs_r['batter_season_pa_hit_rate'] == 0.0


def test_create_batter_pa_outcome_stats_extra_group_cols_defaults_to_no_split():
    """Passing no extra_group_cols must reproduce the original ungrouped-by-hand
    behavior exactly — existing call sites must not change output."""

    result = _create_batter_pa_outcome_stats(_make_hand_split_pa_pbp())

    assert 'pitcher_throw_hand' not in result.columns
    assert len(result) == 1
    assert result.iloc[0]['batter_season_pa_total'] == 3


def test_pivot_by_hand_inserts_hand_tag_after_entity_prefix():
    """_pivot_by_hand turns a long (entity, season, hand) table into one wide
    row per (entity, season), inserting each hand's tag right after the
    entity_prefix in every stat column name."""

    long_df = pd.DataFrame([
        {'batter_id': '1', 'game_season': 2023, 'pitcher_throw_hand': 'L', 'batter_season_pa_hit_rate': 0.5},
        {'batter_id': '1', 'game_season': 2023, 'pitcher_throw_hand': 'R', 'batter_season_pa_hit_rate': 0.2},
    ])

    result = _pivot_by_hand(
        long_df,
        hand_col='pitcher_throw_hand',
        hand_tags={'L': 'vs_lhp', 'R': 'vs_rhp'},
        key_cols=['batter_id', 'game_season'],
        entity_prefix='batter_season_',
    )

    row = result.iloc[0]
    assert row['batter_season_vs_lhp_pa_hit_rate'] == 0.5
    assert row['batter_season_vs_rhp_pa_hit_rate'] == 0.2
    assert 'pitcher_throw_hand' not in result.columns


def test_pivot_by_hand_drops_rows_with_unmapped_hand_codes():
    """A hand code missing from hand_tags (e.g. NaN from an unmatched
    player_info join) must be dropped, not silently produce a garbage
    'batter_season_Nonepa_hit_rate'-style column."""

    long_df = pd.DataFrame([
        {'batter_id': '1', 'game_season': 2023, 'pitcher_throw_hand': 'L', 'batter_season_pa_hit_rate': 0.5},
        {'batter_id': '1', 'game_season': 2023, 'pitcher_throw_hand': None, 'batter_season_pa_hit_rate': 0.9},
    ])

    result = _pivot_by_hand(
        long_df,
        hand_col='pitcher_throw_hand',
        hand_tags={'L': 'vs_lhp', 'R': 'vs_rhp'},
        key_cols=['batter_id', 'game_season'],
        entity_prefix='batter_season_',
    )

    assert not any('0.9' in str(v) for v in result.iloc[0].values)
    assert result.iloc[0]['batter_season_vs_lhp_pa_hit_rate'] == 0.5


def test_build_pbp_batter_feats_by_pitcher_hand_splits_rates_correctly():
    """Reuses _make_full_batter_pbp's 2 PAs (PA1: single, PA2: strikeout) but
    tags PA1 as vs-LHP and PA2 as vs-RHP, so the split must show a 1.0 hit
    rate vs LHP and 0.0 vs RHP — proving the split isn't just pooling both."""

    pbp = _make_full_batter_pbp()
    pbp.loc[pbp['play_id'] == 1, 'pitcher_throw_hand'] = 'L'
    pbp.loc[pbp['play_id'] == 2, 'pitcher_throw_hand'] = 'R'

    result = build_pbp_batter_feats_by_pitcher_hand(pbp)

    row = result.iloc[0]
    assert row['game_season'] == 2024
    assert row['batter_last_season_vs_lhp_pa_hit_rate'] == 1.0
    assert row['batter_last_season_vs_rhp_pa_hit_rate'] == 0.0


def test_build_pbp_batter_feats_by_pitcher_hand_merges_all_substats():
    """Same 5-category merge as build_pbp_batter_feats, but each category now
    split into vs_lhp/vs_rhp column pairs."""

    pbp = _make_full_batter_pbp().assign(pitcher_throw_hand='L')
    other_hand = _make_full_batter_pbp().assign(
        pitcher_throw_hand='R',
        gamepk='2',
        play_id=lambda x: x['play_id'] + 10,
    )
    result = build_pbp_batter_feats_by_pitcher_hand(pd.concat([pbp, other_hand], ignore_index=True))

    row = result.iloc[0]
    for tag in ('vs_lhp', 'vs_rhp'):
        assert f'batter_last_season_{tag}_pa_hit_rate' in result.columns
        assert f'batter_last_season_{tag}_o_swing_rate' in result.columns
        assert f'batter_last_season_{tag}_contact_hard_hit_rate' in result.columns
        assert f'batter_last_season_{tag}_foul_rate' in result.columns
        assert f'batter_last_season_{tag}_two_strike_foul_rate' in result.columns


# --------------------------- HANDEDNESS SPLIT: PITCHER VS BATTER HAND --------------------------- #

def _make_hand_split_pitcher_pa_pbp():
    return pd.DataFrame([
        # vs LHB: 1 PA, a strikeout -> hit_rate 0.0
        {'pitcher_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 1,
         'play_result': 'Strikeout', 'count_balls': 0, 'count_strikes': 2, 'batter_bat_side': 'L'},
        # vs RHB: 1 PA, a single -> hit_rate 1.0
        {'pitcher_id': '1', 'game_season': 2023, 'gamepk': '1', 'play_id': 2, 'pitch_number': 1,
         'play_result': 'Single', 'count_balls': 1, 'count_strikes': 1, 'batter_bat_side': 'R'},
    ])


def test_create_pitcher_pa_outcome_stats_extra_group_cols_splits_rates_by_batter_hand():
    result = _create_pitcher_pa_outcome_stats(
        _make_hand_split_pitcher_pa_pbp(), extra_group_cols=['batter_bat_side']
    )

    assert len(result) == 2
    vs_l = result[result['batter_bat_side'] == 'L'].iloc[0]
    vs_r = result[result['batter_bat_side'] == 'R'].iloc[0]
    assert vs_l['pitcher_season_pa_hit_rate'] == 0.0
    assert vs_r['pitcher_season_pa_hit_rate'] == 1.0


def test_create_pitcher_pa_outcome_stats_extra_group_cols_defaults_to_no_split():
    result = _create_pitcher_pa_outcome_stats(_make_hand_split_pitcher_pa_pbp())

    assert 'batter_bat_side' not in result.columns
    assert len(result) == 1


def _make_full_pitcher_pbp():
    return pd.DataFrame([
        # SP, PA vs LHB: strikeout, not in play
        {'pitcher_id': '1', 'pitcher_team_id': 'T1', 'game_season': 2023, 'gamepk': '1', 'play_id': 1, 'pitch_number': 1,
         'pitcher_role': 'sp', 'batter_bat_side': 'L',
         'is_first_pitch': True, 'is_strike': True, 'is_ball': False, 'is_called_strike': True,
         'is_chase': False, 'is_zone_swing': False, 'is_swinging_strike': False, 'is_in_play': False,
         'start_speed': 95.0, 'end_speed': 87.0, 'perceived_velo': 97.0, 'spin_rate': 2200.0,
         'movement_magnitude': 8.0, 'pfx_z': 10.0, 'extension': 6.5, 'speed_retention': 0.9,
         'plate_x': 0.1, 'plate_z_normalized': 0.5, 'zone': 5,
         'play_result': 'Strikeout', 'count_balls': 0, 'count_strikes': 2,
         'inning': 1, 'count_outs': 1, 'is_pitch': True,
         'hardness': None, 'trajectory': None, 'launch_speed': np.nan, 'launch_angle': np.nan},
        # SP, PA vs RHB: single, in play hard-hit line drive
        {'pitcher_id': '1', 'pitcher_team_id': 'T1', 'game_season': 2023, 'gamepk': '1', 'play_id': 2, 'pitch_number': 1,
         'pitcher_role': 'sp', 'batter_bat_side': 'R',
         'is_first_pitch': True, 'is_strike': False, 'is_ball': False, 'is_called_strike': False,
         'is_chase': False, 'is_zone_swing': True, 'is_swinging_strike': False, 'is_in_play': True,
         'start_speed': 93.0, 'end_speed': 85.0, 'perceived_velo': 95.0, 'spin_rate': 2100.0,
         'movement_magnitude': 7.0, 'pfx_z': 9.0, 'extension': 6.2, 'speed_retention': 0.91,
         'plate_x': -0.2, 'plate_z_normalized': 0.4, 'zone': 4,
         'play_result': 'Single', 'count_balls': 1, 'count_strikes': 1,
         'inning': 1, 'count_outs': 1, 'is_pitch': True,
         'hardness': 'Hard', 'trajectory': 'Line Drive', 'launch_speed': 100.0, 'launch_angle': 15.0},
        # Bullpen appearance, different game, vs LHB — must be excluded when pitcher_role='sp'
        {'pitcher_id': '1', 'pitcher_team_id': 'T1', 'game_season': 2023, 'gamepk': '2', 'play_id': 1, 'pitch_number': 1,
         'pitcher_role': 'bullpen', 'batter_bat_side': 'L',
         'is_first_pitch': True, 'is_strike': False, 'is_ball': True, 'is_called_strike': False,
         'is_chase': False, 'is_zone_swing': False, 'is_swinging_strike': False, 'is_in_play': False,
         'start_speed': 91.0, 'end_speed': 83.0, 'perceived_velo': 93.0, 'spin_rate': 2000.0,
         'movement_magnitude': 6.0, 'pfx_z': 8.0, 'extension': 6.0, 'speed_retention': 0.9,
         'plate_x': 1.0, 'plate_z_normalized': 1.5, 'zone': 12,
         'play_result': 'Walk', 'count_balls': 4, 'count_strikes': 1,
         'inning': 9, 'count_outs': 1, 'is_pitch': True,
         'hardness': None, 'trajectory': None, 'launch_speed': np.nan, 'launch_angle': np.nan},
    ])


def _make_bullpen_team_pool_pbp():
    """Pitcher '1' starts a game (gamepk='1', sp), then relieves in a later game
    (gamepk='2', bullpen) for team 'T1', alongside a DIFFERENT reliever ('2') on
    the same team in that same game. Proves entity_col='pitcher_team_id' pools
    bullpen appearances across different pitcher_ids into one row, while sp
    stays individually keyed — self-contained (no row collisions if reused
    alongside other fixtures)."""
    return pd.DataFrame([
        {'pitcher_id': '1', 'pitcher_team_id': 'T1', 'game_season': 2023, 'gamepk': '1',
         'play_id': 1, 'pitch_number': 1,
         'pitcher_role': 'sp', 'batter_bat_side': 'L',
         'is_first_pitch': True, 'is_strike': True, 'is_ball': False, 'is_called_strike': True,
         'is_chase': False, 'is_zone_swing': False, 'is_swinging_strike': False, 'is_in_play': False,
         'start_speed': 95.0, 'end_speed': 87.0, 'perceived_velo': 97.0, 'spin_rate': 2200.0,
         'movement_magnitude': 8.0, 'pfx_z': 10.0, 'extension': 6.5, 'speed_retention': 0.9,
         'plate_x': 0.1, 'plate_z_normalized': 0.5, 'zone': 5,
         'play_result': 'Strikeout', 'count_balls': 0, 'count_strikes': 2,
         'inning': 1, 'count_outs': 1, 'is_pitch': True,
         'hardness': None, 'trajectory': None, 'launch_speed': np.nan, 'launch_angle': np.nan},
        {'pitcher_id': '1', 'pitcher_team_id': 'T1', 'game_season': 2023, 'gamepk': '2',
         'play_id': 1, 'pitch_number': 1,
         'pitcher_role': 'bullpen', 'batter_bat_side': 'L',
         'is_first_pitch': True, 'is_strike': False, 'is_ball': True, 'is_called_strike': False,
         'is_chase': False, 'is_zone_swing': False, 'is_swinging_strike': False, 'is_in_play': False,
         'start_speed': 91.0, 'end_speed': 83.0, 'perceived_velo': 93.0, 'spin_rate': 2000.0,
         'movement_magnitude': 6.0, 'pfx_z': 8.0, 'extension': 6.0, 'speed_retention': 0.9,
         'plate_x': 1.0, 'plate_z_normalized': 1.5, 'zone': 12,
         'play_result': 'Walk', 'count_balls': 4, 'count_strikes': 1,
         'inning': 9, 'count_outs': 1, 'is_pitch': True,
         'hardness': None, 'trajectory': None, 'launch_speed': np.nan, 'launch_angle': np.nan},
        {'pitcher_id': '2', 'pitcher_team_id': 'T1', 'game_season': 2023, 'gamepk': '2',
         'play_id': 2, 'pitch_number': 1,
         'pitcher_role': 'bullpen', 'batter_bat_side': 'R',
         'is_first_pitch': True, 'is_strike': False, 'is_ball': False, 'is_called_strike': False,
         'is_chase': False, 'is_zone_swing': True, 'is_swinging_strike': False, 'is_in_play': True,
         'start_speed': 90.0, 'end_speed': 82.0, 'perceived_velo': 92.0, 'spin_rate': 1950.0,
         'movement_magnitude': 5.0, 'pfx_z': 7.0, 'extension': 5.8, 'speed_retention': 0.91,
         'plate_x': -0.1, 'plate_z_normalized': 0.4, 'zone': 4,
         'play_result': 'Single', 'count_balls': 1, 'count_strikes': 1,
         'inning': 9, 'count_outs': 2, 'is_pitch': True,
         'hardness': 'Hard', 'trajectory': 'Line Drive', 'launch_speed': 98.0, 'launch_angle': 12.0},
    ])


def test_build_pbp_pitcher_feats_pools_bullpen_by_team_when_entity_col_given():
    """A team's bullpen isn't one identity — entity_col lets the same aggregation
    code pool ALL of a team's bullpen pitchers into one row instead of keeping
    them separate by individual pitcher_id."""

    pbp = _make_bullpen_team_pool_pbp()

    result = build_pbp_pitcher_feats(pbp, pitcher_role='bullpen', entity_col='pitcher_team_id')

    assert len(result) == 1
    assert 'pitcher_team_id' in result.columns
    assert 'pitcher_id' not in result.columns
    assert result.iloc[0]['pitcher_last_season_pa_total'] == 2


def test_build_pbp_pitcher_feats_by_batter_hand_splits_rates_correctly():
    result = build_pbp_pitcher_feats_by_batter_hand(_make_full_pitcher_pbp(), pitcher_role='sp')

    row = result.iloc[0]
    assert row['game_season'] == 2024
    assert row['pitcher_last_season_vs_lhb_pa_hit_rate'] == 0.0
    assert row['pitcher_last_season_vs_rhb_pa_hit_rate'] == 1.0


def test_build_pbp_pitcher_feats_by_batter_hand_respects_pitcher_role_filter():
    """pitcher_role='sp' must exclude the bullpen row entirely — its
    batter_bat_side='L' PA must not leak into vs_lhb stats."""

    result = build_pbp_pitcher_feats_by_batter_hand(_make_full_pitcher_pbp(), pitcher_role='sp')

    row = result.iloc[0]
    # only 1 vs-LHB PA counted (the SP one), not 2 (which would include the bullpen PA)
    assert row['pitcher_last_season_vs_lhb_pa_total'] == 1


def test_pitcher_role_lookup_returns_one_row_per_pitcher_per_game():
    """pitcher_boxscore has no pitcher_role column of its own — this lookup,
    derived from pbp, is how role gets tagged onto boxscore rows for the
    never-role-aware pitcher_season_stats/pitcher_rolling_*_stats fix. Must
    dedupe down to one row per (gamepk, pitcher_id) even though pbp has
    multiple pitch-level rows per pitcher per game."""

    result = _pitcher_role_lookup(_make_full_pitcher_pbp())

    assert len(result) == 2
    assert set(result.columns) == {'gamepk', 'pitcher_id', 'pitcher_team_id', 'pitcher_role'}

    bullpen_row = result[result['gamepk'] == '2'].iloc[0]
    assert bullpen_row['pitcher_role'] == 'bullpen'
    assert bullpen_row['pitcher_team_id'] == 'T1'


def test_build_pitcher_stats_all_roles_pools_bullpen_by_team():
    """pitcher_season_stats (boxscore-derived) has never been role-aware at
    all — it blends a pitcher's sp+bullpen appearances into one aggregate and
    gets joined onto every PA for that pitcher_id regardless of role, same
    serving-time bug as the pbp-derived stats had, on a different code path
    (pitcher_boxscore has no pitcher_role column of its own). Two relievers
    on the same team must pool into one bullpen row keyed by team, not stay
    separate individual-pitcher rows."""

    pitcher_boxscore = pd.DataFrame([
        {'personId': '10', 'gamepk': 'g1', 'team_id': 'T1', 'game_season': 2023,
         'h': 1, 'r': 1, 'er': 1, 'bb': 1, 'hr': 0, 'k': 2, 'p': 20, 's': 12, 'ip': 1.0},
        {'personId': '11', 'gamepk': 'g1', 'team_id': 'T1', 'game_season': 2023,
         'h': 2, 'r': 2, 'er': 2, 'bb': 0, 'hr': 1, 'k': 1, 'p': 15, 's': 9, 'ip': 1.0},
    ])
    pbp = pd.DataFrame([
        {'gamepk': 'g1', 'pitcher_id': '10', 'pitcher_team_id': 'T1', 'pitcher_role': 'bullpen'},
        {'gamepk': 'g1', 'pitcher_id': '11', 'pitcher_team_id': 'T1', 'pitcher_role': 'bullpen'},
    ])

    result = build_pitcher_stats_all_roles(pitcher_boxscore, pbp)

    bullpen_rows = result[result['pitcher_role'] == 'bullpen']
    assert len(bullpen_rows) == 1
    assert bullpen_rows.iloc[0]['pitcher_key_id'] == 'T1'
    assert bullpen_rows.iloc[0]['pitcher_last_season_h'] == 3  # 1 + 2, pooled across both relievers


def test_build_pbp_pitcher_feats_all_roles_pools_bullpen_by_team():
    """The bullpen half must be team-pooled, not individual-pitcher-keyed —
    two different relievers on the same team collapse into one bullpen row
    under a common pitcher_key_id column, while the sp row keeps its own
    individual pitcher_id under that same column name so both halves can be
    concatenated and later joined onto PA-level data uniformly."""

    pbp = _make_bullpen_team_pool_pbp()

    result = build_pbp_pitcher_feats_all_roles(pbp)

    assert 'pitcher_key_id' in result.columns
    assert 'pitcher_id' not in result.columns
    assert 'pitcher_team_id' not in result.columns

    sp_row = result[result['pitcher_role'] == 'sp'].iloc[0]
    assert sp_row['pitcher_key_id'] == '1'

    bullpen_rows = result[result['pitcher_role'] == 'bullpen']
    assert len(bullpen_rows) == 1
    assert bullpen_rows.iloc[0]['pitcher_key_id'] == 'T1'


def test_build_pbp_pitcher_feats_all_roles_tags_and_stacks_both_roles():
    """The bug this fixes: sp_season_stats was previously joined onto every
    PA for a pitcher_id regardless of that PA's actual game-context role, so
    a swingman's starter stats bled into his relief-appearance rows (and no
    bullpen table existed to give relief PAs their own signal at all).
    build_pbp_pitcher_feats_all_roles must produce one row per role that
    actually occurred, each tagged and computed from ONLY that role's PAs —
    not a blended average across both."""

    result = build_pbp_pitcher_feats_all_roles(_make_full_pitcher_pbp())

    assert len(result) == 2
    assert set(result['pitcher_role']) == {'sp', 'bullpen'}

    sp_row = result[result['pitcher_role'] == 'sp'].iloc[0]
    bullpen_row = result[result['pitcher_role'] == 'bullpen'].iloc[0]

    # SP: 1 hit (Single) out of 2 PAs (Strikeout, Single) = 0.5
    assert sp_row['pitcher_last_season_pa_hit_rate'] == 0.5
    # bullpen: 1 PA (Walk), not a hit = 0.0 — must not be contaminated by the SP PAs
    assert bullpen_row['pitcher_last_season_pa_hit_rate'] == 0.0


def test_build_pbp_pitcher_feats_by_batter_hand_merges_all_substats():
    result = build_pbp_pitcher_feats_by_batter_hand(_make_full_pitcher_pbp())

    for tag in ('vs_lhb', 'vs_rhb'):
        assert f'pitcher_last_season_{tag}_pa_hit_rate' in result.columns
        assert f'pitcher_last_season_{tag}_stuff_start_speed_mean' in result.columns
        assert f'pitcher_last_season_{tag}_game_avg_pitch_count' in result.columns
        assert f'pitcher_last_season_{tag}_contact_hard_hit_rate' in result.columns


# --------------------------- LEAGUE-WIDE HANDEDNESS CONTEXT TABLE --------------------------- #

def _make_league_handedness_pbp():
    # batter '1' vs LHP: PA1 single (hit), PA2 strikeout (not hit) -> 0.5 hit rate alone
    batter_1 = _make_full_batter_pbp().assign(pitcher_throw_hand='L', batter_bat_side='R')

    # batter '2', same hand pair (L vs R), both PAs are hits -> 1.0 hit rate alone.
    # Pooled with batter 1: 3 hits / 4 PAs = 0.75 — proves league table aggregates
    # across ALL batters sharing a hand pair, not per-batter.
    batter_2 = _make_full_batter_pbp().assign(
        pitcher_throw_hand='L', batter_bat_side='R',
        batter_id='2', gamepk='2', play_id=lambda x: x['play_id'] + 10,
        play_result=lambda x: x['play_result'].replace({'Strikeout': 'Single'}),
    )

    return pd.concat([batter_1, batter_2], ignore_index=True)


def test_build_league_handedness_stats_pools_across_all_batters_sharing_a_hand_pair():
    result = build_league_handedness_stats(_make_league_handedness_pbp())

    assert set(result.columns) >= {'game_season', 'pitcher_throw_hand', 'batter_bat_side'}
    row = result[(result['pitcher_throw_hand'] == 'L') & (result['batter_bat_side'] == 'R')].iloc[0]
    assert row['game_season'] == 2024
    assert row['league_last_season_pa_hit_rate'] == 0.75
    assert row['league_last_season_pa_total'] == 4


def test_build_league_handedness_stats_has_no_per_player_id_columns():
    """This table is a population-level aggregate — batter_id/pitcher_id must
    not appear, otherwise it isn't actually the coarse, stable 'high-level'
    context table it's meant to be alongside the noisier per-player splits."""

    result = build_league_handedness_stats(_make_league_handedness_pbp())

    assert 'batter_id' not in result.columns
    assert 'pitcher_id' not in result.columns


def test_build_league_handedness_stats_merges_all_substats():
    result = build_league_handedness_stats(_make_league_handedness_pbp())

    assert 'league_last_season_pa_hit_rate' in result.columns
    assert 'league_last_season_o_swing_rate' in result.columns
    assert 'league_last_season_contact_hard_hit_rate' in result.columns
    assert 'league_last_season_foul_rate' in result.columns
    assert 'league_last_season_two_strike_foul_rate' in result.columns
