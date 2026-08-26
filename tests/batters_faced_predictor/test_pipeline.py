import pandas as pd

from models.batters_faced_predictor.processing.pipeline import create_start_pa_outcome


def _pbp_row(**overrides):
    row = {
        'gamepk': '1', 'pitcher_id': '10', 'pitcher_role': 'sp',
        'play_id': 1, 'pitch_number': 1, 'play_result': 'Single',
        'count_balls': 0, 'count_strikes': 0,
        'game_date': pd.Timestamp('2023-05-01'), 'game_season': 2023,
    }
    row.update(overrides)
    return row


def test_create_start_pa_outcome_labels_realized_batters_faced():
    """3 distinct PAs (by play_id) for one sp start -> realized_batters_faced == 3."""
    pbp = pd.DataFrame([
        _pbp_row(play_id=1, play_result='Strikeout'),
        _pbp_row(play_id=2, play_result='Walk'),
        _pbp_row(play_id=3, play_result='Single'),
    ])

    result = create_start_pa_outcome(pbp)

    assert result.iloc[0]['realized_batters_faced'] == 3


def test_create_start_pa_outcome_excludes_bullpen_role_pas():
    """A bullpen pbp row isn't part of the starter's own start — out of scope
    by construction, same as short_outing_predictor's create_start_outcome."""
    pbp = pd.DataFrame([
        _pbp_row(pitcher_id='10', pitcher_role='sp', play_id=1),
        _pbp_row(pitcher_id='11', pitcher_role='bullpen', play_id=2),
    ])

    result = create_start_pa_outcome(pbp)

    assert list(result['personId']) == ['10']


def test_create_start_pa_outcome_returns_expected_columns():
    pbp = pd.DataFrame([_pbp_row()])

    result = create_start_pa_outcome(pbp)

    assert set(result.columns) == {
        'personId', 'gamepk', 'game_date', 'game_season', 'realized_batters_faced',
    }
