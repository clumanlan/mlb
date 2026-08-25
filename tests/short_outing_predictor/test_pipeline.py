import pandas as pd

from models.short_outing_predictor.processing.pipeline import create_start_outcome
from models.short_outing_predictor.processing.schema import SHORT_OUTING_IP_THRESHOLD


def _pbp_row(**overrides):
    row = {
        'gamepk': '1', 'pitcher_id': '10', 'pitcher_team_id': 'T1', 'pitcher_role': 'sp',
    }
    row.update(overrides)
    return row


def _boxscore_row(**overrides):
    row = {
        'personId': '10', 'gamepk': '1', 'ip': 6.0,
        'game_date': pd.Timestamp('2023-05-01'), 'game_season': 2023,
    }
    row.update(overrides)
    return row


def test_create_start_outcome_labels_short_outing():
    pbp = pd.DataFrame([_pbp_row()])
    pitcher_boxscore = pd.DataFrame([_boxscore_row(ip=3.0)])

    result = create_start_outcome(pitcher_boxscore, pbp)

    assert result.iloc[0]['is_short_outing'] == 1


def test_create_start_outcome_full_start_not_short():
    pbp = pd.DataFrame([_pbp_row()])
    pitcher_boxscore = pd.DataFrame([_boxscore_row(ip=6.0)])

    result = create_start_outcome(pitcher_boxscore, pbp)

    assert result.iloc[0]['is_short_outing'] == 0


def test_create_start_outcome_boundary_is_inclusive():
    """README defines the mechanism as '<=4 IP' — exactly 4.0 IP counts as
    a short outing, not just strictly fewer innings."""
    pbp = pd.DataFrame([_pbp_row()])
    pitcher_boxscore = pd.DataFrame([_boxscore_row(ip=SHORT_OUTING_IP_THRESHOLD)])

    result = create_start_outcome(pitcher_boxscore, pbp)

    assert result.iloc[0]['is_short_outing'] == 1


def test_create_start_outcome_excludes_bullpen_appearances():
    """A bullpen boxscore row isn't a 'start' at all — out of scope by
    construction (unlike k_predictor's/bb_predictor's own sp-only PA
    scoping, which is a deliberate population choice among options that
    could have gone either way; here a bullpen appearance was never a
    candidate 'start' row in the first place)."""
    pbp = pd.DataFrame([
        _pbp_row(pitcher_id='10', pitcher_role='sp'),
        _pbp_row(pitcher_id='11', pitcher_role='bullpen'),
    ])
    pitcher_boxscore = pd.DataFrame([
        _boxscore_row(personId='10', ip=6.0),
        _boxscore_row(personId='11', ip=1.0),
    ])

    result = create_start_outcome(pitcher_boxscore, pbp)

    assert list(result['personId']) == ['10']


def test_create_start_outcome_returns_expected_columns():
    pbp = pd.DataFrame([_pbp_row()])
    pitcher_boxscore = pd.DataFrame([_boxscore_row()])

    result = create_start_outcome(pitcher_boxscore, pbp)

    assert set(result.columns) == {
        'personId', 'gamepk', 'game_date', 'game_season', 'ip', 'is_short_outing',
    }
