import pandas as pd

from models.hit_predictor.processing.pipeline import (
    _add_pbp_handedness,
    _initial_pbp_processing,
    create_pa_outcome,
)


def _make_pbp_row(**overrides):
    row = {
        'gamepk': '1',
        'batter_id': '1',
        'pitcher_id': '10',
        'play_result': None,
        'end_speed': 85.0,
        'start_speed': 90.0,
        'extension': 6.0,
        'pitch_call': 'Ball',
        'zone': 12,
        'is_in_play': False,
        'plate_z': 3.0,
        'strike_zone_bottom': 1.5,
        'strike_zone_top': 3.5,
        'pfx_x': 2.0,
        'pfx_z': 5.0,
    }
    row.update(overrides)
    return row


def test_is_chase_counts_a_foul_ball_outside_the_zone():
    """is_chase was gated on is_swinging_strike, so a foul ball on a pitch
    outside the zone (a real chase — the batter swung at a non-strike) was
    never counted. Any swing outside the zone should count as a chase,
    regardless of whether contact was made."""

    pbp = pd.DataFrame([_make_pbp_row(pitch_call='Foul', zone=12, is_in_play=False)])

    result = _initial_pbp_processing(pbp, target_col='is_hit')

    assert result.loc[0, 'is_chase'] == True


def test_is_zone_swing_counts_a_ball_put_in_play_inside_the_zone():
    """is_zone_swing checked pitch_call.isin([..., 'In Play']), but real MLB
    Stats API pitch_call values are 'In play, out(s)' / 'In play, no out' /
    'In play, run(s)' — the literal string 'In Play' never matches, so every
    ball put in play inside the zone was silently excluded from zone-swing
    rate. is_in_play (already a reliable boolean column) should be used
    instead of string-matching pitch_call for the in-play case."""

    pbp = pd.DataFrame([_make_pbp_row(pitch_call='In play, no out', zone=5, is_in_play=True)])

    result = _initial_pbp_processing(pbp, target_col='is_hit')

    assert result.loc[0, 'is_zone_swing'] == True


def test_is_zone_swing_counts_foul_tip_inside_the_zone():
    """The swing-call list was also missing Foul Tip, Foul Bunt, Swinging
    Strike (Blocked), and Missed Bunt — all real swing outcomes."""

    pbp = pd.DataFrame([_make_pbp_row(pitch_call='Foul Tip', zone=5, is_in_play=False)])

    result = _initial_pbp_processing(pbp, target_col='is_hit')

    assert result.loc[0, 'is_zone_swing'] == True


def test_is_chase_false_for_a_take_outside_the_zone():
    """A pitch outside the zone that the batter did NOT swing at (a ball) is
    not a chase — this must stay False after the fix."""

    pbp = pd.DataFrame([_make_pbp_row(pitch_call='Ball', zone=12, is_in_play=False)])

    result = _initial_pbp_processing(pbp, target_col='is_hit')

    assert result.loc[0, 'is_chase'] == False


def test_is_swinging_strike_counts_a_missed_bunt():
    """A missed bunt is a swing-and-miss just like a whiffed full swing, but
    is_swinging_strike only checked for 'Swinging Strike'/'Swinging Strike
    (Blocked)' — Contact%-style metrics built on this flag would silently
    count a missed bunt as contact."""

    pbp = pd.DataFrame([_make_pbp_row(pitch_call='Missed Bunt', zone=5, is_in_play=False)])

    result = _initial_pbp_processing(pbp, target_col='is_hit')

    assert result.loc[0, 'is_swinging_strike'] == True


def test_add_pbp_handedness_pulls_hand_from_the_correct_role():
    """pitcher_throw_hand must come from the PITCHER's player_info row and
    batter_bat_side from the BATTER's row — not swapped. Each fixture player
    is given a distinct hand value per role specifically so a swap bug (using
    pitchHand for the batter or batSide for the pitcher) would be caught."""

    pbp = pd.DataFrame([{'pitcher_id': '10', 'batter_id': '1'}])
    player_info = pd.DataFrame([
        {'person_id': 10, 'pitchHand': 'L', 'batSide': 'R'},
        {'person_id': 1, 'pitchHand': 'R', 'batSide': 'S'},
    ])

    result = _add_pbp_handedness(pbp, player_info)

    assert result.loc[0, 'pitcher_throw_hand'] == 'L'
    assert result.loc[0, 'batter_bat_side'] == 'S'


def test_add_pbp_handedness_joins_despite_int_vs_str_id_dtype_mismatch():
    """player_info.person_id comes from the MLB API as a raw int, while pbp's
    batter_id/pitcher_id are cast to str earlier in the pipeline. A merge
    without explicit dtype alignment silently produces an all-NaN join
    instead of erroring, so this must be guarded explicitly."""

    pbp = pd.DataFrame([{'pitcher_id': '10', 'batter_id': '1'}])
    player_info = pd.DataFrame([
        {'person_id': 10, 'pitchHand': 'L', 'batSide': 'R'},
        {'person_id': 1, 'pitchHand': 'R', 'batSide': 'L'},
    ])

    result = _add_pbp_handedness(pbp, player_info)

    assert result.loc[0, 'pitcher_throw_hand'] == 'L'
    assert result.loc[0, 'batter_bat_side'] == 'L'


def test_create_pa_outcome_includes_batter_and_pitcher_hand():
    """pitcher_throw_hand and batter_bat_side are computed onto pbp by
    _add_pbp_handedness before create_pa_outcome runs, but create_pa_outcome's
    column selection dropped them — the strongest, cheapest platoon-split
    signal in the dataset never made it to the model. Both must survive into
    pa_outcome's output, with values matching the source pbp row (not
    swapped between batter/pitcher)."""

    pbp = pd.DataFrame([{
        'gamepk': '1',
        'batter_team_name': 'Cubs',
        'play_id': 'p1',
        'pitcher_id': '10',
        'pitcher_name': 'Pitcher One',
        'batter_id': '1',
        'batter_name': 'Batter One',
        'is_hit': 1,
        'pitcher_throw_hand': 'L',
        'batter_bat_side': 'S',
        'pitcher_role': 'sp',
        'pitcher_team_id': 'T1',
    }])
    batter_boxscore = pd.DataFrame([{
        'gamepk': '1',
        'personId': '1',
        'batting_order': 3,
    }])
    game_info = pd.DataFrame([{
        'gamepk': '1',
        'game_season': 2023,
        'weather_condition': 'Sunny',
        'weather_temp': '75',
    }])
    schedule = pd.DataFrame([{
        'gamepk': '1',
        'game_date': pd.Timestamp('2023-05-01'),
    }])

    result = create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

    assert result.loc[0, 'pitcher_throw_hand'] == 'L'
    assert result.loc[0, 'batter_bat_side'] == 'S'


def test_create_pa_outcome_includes_pitcher_role():
    """pitcher_role ('sp'/'bullpen') is computed onto pbp by
    _add_pbp_starting_pitcher before create_pa_outcome runs, but was never
    selected into pa_outcome's output — without it, downstream code has no
    way to tell whether a given PA happened while this pitcher was starting
    or relieving, which is what causes starter stats to get misapplied to
    relief appearances for swingman pitchers."""

    pbp = pd.DataFrame([{
        'gamepk': '1',
        'batter_team_name': 'Cubs',
        'play_id': 'p1',
        'pitcher_id': '10',
        'pitcher_name': 'Pitcher One',
        'batter_id': '1',
        'batter_name': 'Batter One',
        'is_hit': 1,
        'pitcher_throw_hand': 'L',
        'batter_bat_side': 'S',
        'pitcher_role': 'bullpen',
        'pitcher_team_id': 'T1',
    }])
    batter_boxscore = pd.DataFrame([{
        'gamepk': '1',
        'personId': '1',
        'batting_order': 3,
    }])
    game_info = pd.DataFrame([{
        'gamepk': '1',
        'game_season': 2023,
        'weather_condition': 'Sunny',
        'weather_temp': '75',
    }])
    schedule = pd.DataFrame([{
        'gamepk': '1',
        'game_date': pd.Timestamp('2023-05-01'),
    }])

    result = create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

    assert result.loc[0, 'pitcher_role'] == 'bullpen'


def test_create_pa_outcome_includes_pitcher_team_id():
    """pitcher_team_id is needed to compute pitcher_key_id (sp -> pitcher_id,
    bullpen -> pitcher_team_id) for the team-pooled bullpen feature join —
    without it, bullpen-role PAs have no way to look up their team's pooled
    stats."""

    pbp = pd.DataFrame([{
        'gamepk': '1',
        'batter_team_name': 'Cubs',
        'play_id': 'p1',
        'pitcher_id': '10',
        'pitcher_name': 'Pitcher One',
        'batter_id': '1',
        'batter_name': 'Batter One',
        'is_hit': 1,
        'pitcher_throw_hand': 'L',
        'batter_bat_side': 'S',
        'pitcher_role': 'bullpen',
        'pitcher_team_id': 'T1',
    }])
    batter_boxscore = pd.DataFrame([{
        'gamepk': '1',
        'personId': '1',
        'batting_order': 3,
    }])
    game_info = pd.DataFrame([{
        'gamepk': '1',
        'game_season': 2023,
        'weather_condition': 'Sunny',
        'weather_temp': '75',
    }])
    schedule = pd.DataFrame([{
        'gamepk': '1',
        'game_date': pd.Timestamp('2023-05-01'),
    }])

    result = create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

    assert result.loc[0, 'pitcher_team_id'] == 'T1'
