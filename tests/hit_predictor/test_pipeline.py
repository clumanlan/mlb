import pandas as pd

from models.hit_predictor.processing.pipeline import _initial_pbp_processing


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
