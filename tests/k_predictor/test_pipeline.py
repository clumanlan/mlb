import pandas as pd

from models.k_predictor.processing.pipeline import (
    build_pbp_features_strikeout,
    create_pa_outcome_strikeout,
)


def _make_pbp_row(**overrides):
    row = {
        'gamepk': '1',
        'play_id': 'p1',
        'event_index': 1,
        'pitch_number': 1,
        'count_balls': 0,
        'count_strikes': 0,
        'batter_id': '1',
        'pitcher_id': '10',
        'pitcher_team_id': 'T1',
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


def _schedule():
    return pd.DataFrame([{'gamepk': '1', 'game_date': pd.Timestamp('2023-05-01'), 'venue_id': 'V1'}])


def _player_info():
    return pd.DataFrame([
        {'person_id': '10', 'pitchHand': 'R', 'batSide': None},
        {'person_id': '1', 'pitchHand': None, 'batSide': 'L'},
        {'person_id': '2', 'pitchHand': None, 'batSide': 'R'},
    ])


def test_build_pbp_features_strikeout_adds_is_strikeout_column():
    """build_pbp_features_strikeout composes hit_predictor's own
    build_pbp_features (unmodified) rather than duplicating its pitch-level
    logic — is_hit (hit_predictor's own target) must still be present
    alongside the new is_strikeout column, proving composition not
    reimplementation."""

    pbp = pd.DataFrame([
        _make_pbp_row(play_id='p1', batter_id='1', play_result='Strikeout', pitch_call='Swinging Strike'),
        _make_pbp_row(play_id='p2', batter_id='2', play_result='Single', pitch_call='In play, no out', is_in_play=True),
    ])

    result = build_pbp_features_strikeout(pbp, _schedule(), _player_info())

    result = result.set_index('play_id')
    assert result.loc['p1', 'is_strikeout'] == 1
    assert result.loc['p2', 'is_strikeout'] == 0
    assert 'is_hit' in result.columns
    assert result.loc['p1', 'is_hit'] == 0
    assert result.loc['p2', 'is_hit'] == 1


def _make_pa_outcome_row(**overrides):
    row = {
        'gamepk': '1',
        'batter_team_name': 'Cubs',
        'batter_team_id': 'T2',
        'play_id': 'p1',
        'pitcher_id': '10',
        'pitcher_name': 'Pitcher One',
        'batter_id': '1',
        'batter_name': 'Batter One',
        'is_strikeout': 1,
        'pitcher_throw_hand': 'L',
        'batter_bat_side': 'S',
        'pitcher_role': 'sp',
        'pitcher_team_id': 'T1',
        'starting_pitcher_id': '10',
        'batter_pa_number': 1,
    }
    row.update(overrides)
    return row


def _game_info():
    return pd.DataFrame([{
        'gamepk': '1', 'game_season': 2023, 'weather_condition': 'Sunny', 'weather_temp': '75',
    }])


def test_create_pa_outcome_strikeout_label_correct():
    pbp = pd.DataFrame([_make_pa_outcome_row(is_strikeout=1)])
    batter_boxscore = pd.DataFrame([{'gamepk': '1', 'personId': '1', 'batting_order': 3}])

    result = create_pa_outcome_strikeout(pbp, batter_boxscore, _game_info(), _schedule())

    assert result.iloc[0]['is_strikeout'] == 1


def test_create_pa_outcome_strikeout_excludes_non_starters():
    """Same inner join on batting_order as hit_predictor's create_pa_outcome
    — a batter with no batting_order row (pinch hitter/sub) is dropped."""
    pbp = pd.DataFrame([
        _make_pa_outcome_row(batter_id='1', play_id='p1'),
        _make_pa_outcome_row(batter_id='2', play_id='p2'),
    ])
    batter_boxscore = pd.DataFrame([{'gamepk': '1', 'personId': '1', 'batting_order': 3}])

    result = create_pa_outcome_strikeout(pbp, batter_boxscore, _game_info(), _schedule())

    assert list(result['batter_id']) == ['1']


def test_create_pa_outcome_strikeout_excludes_bullpen_role_pas():
    """K-prop's real-world target is a named starting pitcher's strikeout
    total — a bullpen PA's actual pitcher isn't identifiable pre-game
    (pooled by team, see expected_role.py), so it isn't part of that
    population at all, unlike hit_predictor's own is_hit model which pools
    both roles. Scope to REALIZED pitcher_role=='sp' only, same convention
    as season_stats._create_pitcher_start_depth_stats' sp_pbp filter."""
    pbp = pd.DataFrame([
        _make_pa_outcome_row(batter_id='1', play_id='p1', pitcher_role='sp'),
        _make_pa_outcome_row(batter_id='1', play_id='p2', pitcher_role='bullpen'),
    ])
    batter_boxscore = pd.DataFrame([{'gamepk': '1', 'personId': '1', 'batting_order': 3}])

    result = create_pa_outcome_strikeout(pbp, batter_boxscore, _game_info(), _schedule())

    assert list(result['play_id']) == ['p1']
    assert (result['pitcher_role'] == 'sp').all()


def test_create_pa_outcome_strikeout_adds_platoon_matchup():
    """Mirrors hit_predictor's slice_diagnostic.py derivation: an explicit
    same_hand/opposite_hand/switch_hitter column, so the model doesn't have
    to rediscover the batter_bat_side == pitcher_throw_hand equality from
    two separately-encoded categorical columns."""
    pbp = pd.DataFrame([
        _make_pa_outcome_row(batter_id='1', play_id='p1', pitcher_throw_hand='L', batter_bat_side='L'),
        _make_pa_outcome_row(batter_id='1', play_id='p2', pitcher_throw_hand='L', batter_bat_side='R'),
        _make_pa_outcome_row(batter_id='1', play_id='p3', pitcher_throw_hand='L', batter_bat_side='S'),
        _make_pa_outcome_row(batter_id='1', play_id='p4', pitcher_throw_hand=None, batter_bat_side='R'),
    ])
    batter_boxscore = pd.DataFrame([{'gamepk': '1', 'personId': '1', 'batting_order': 3}])

    result = create_pa_outcome_strikeout(pbp, batter_boxscore, _game_info(), _schedule())
    result = result.set_index('play_id')

    assert result.loc['p1', 'platoon_matchup'] == 'same_hand'
    assert result.loc['p2', 'platoon_matchup'] == 'opposite_hand'
    assert result.loc['p3', 'platoon_matchup'] == 'switch_hitter'
    assert pd.isna(result.loc['p4', 'platoon_matchup'])


def test_create_pa_outcome_strikeout_includes_estimated_team_pa_position():
    """Hand-computed: batter_pa_number=2, batting_order=4 -> (2-1)*9+4 = 13."""
    pbp = pd.DataFrame([_make_pa_outcome_row(batter_pa_number=2)])
    batter_boxscore = pd.DataFrame([{'gamepk': '1', 'personId': '1', 'batting_order': 4}])

    result = create_pa_outcome_strikeout(pbp, batter_boxscore, _game_info(), _schedule())

    assert result.iloc[0]['estimated_team_pa_position'] == 13
