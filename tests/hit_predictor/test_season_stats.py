import pandas as pd

from models.hit_predictor.processing.features.season_stats import (
    _create_pitcher_contact_quality_stats,
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
