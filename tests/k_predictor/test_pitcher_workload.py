import numpy as np
import pandas as pd
import pytest

from models.k_predictor.processing.features.pitcher_workload import (
    build_pitcher_shrunk_whip,
    build_pitcher_projected_workload,
)


def _rolling_row(**overrides):
    row = {
        'pitcher_key_id': 'P1',
        'pitcher_role': 'sp',
        'game_season': 2023,
        'pitcher_roll_season_whip': 1.0,
        'pitcher_roll_season_games_n': 10,
    }
    row.update(overrides)
    return row


def _season_row(**overrides):
    row = {
        'pitcher_key_id': 'P1',
        'pitcher_role': 'sp',
        'game_season': 2023,
        'pitcher_last_season_whip': 1.4,
    }
    row.update(overrides)
    return row


def test_build_pitcher_shrunk_whip_blends_toward_rolling_as_sample_grows():
    """shrinkage_weight = games_n / (games_n + k). k=20, games_n=10 ->
    weight=1/3. Hand-computed: (2/3)*1.4 + (1/3)*1.0 = 1.2666..."""

    rolling = pd.DataFrame([_rolling_row(pitcher_roll_season_whip=1.0, pitcher_roll_season_games_n=10)])
    season = pd.DataFrame([_season_row(pitcher_last_season_whip=1.4)])

    result = build_pitcher_shrunk_whip(rolling, season, window='season', k=20.0)

    assert result['pitcher_shrunk_whip'].iloc[0] == pytest.approx((2 / 3) * 1.4 + (1 / 3) * 1.0)
    assert result['pitcher_shrunk_whip_weight'].iloc[0] == pytest.approx(1 / 3)


def test_build_pitcher_shrunk_whip_trusts_last_season_fully_at_season_opener():
    """games_n=0 -> weight=0 -> shrunk value equals last season's WHIP
    exactly, same "trust the prior fully with no in-season sample" behavior
    as game_context.build_expected_start_innings."""

    rolling = pd.DataFrame([_rolling_row(pitcher_roll_season_whip=None, pitcher_roll_season_games_n=0)])
    season = pd.DataFrame([_season_row(pitcher_last_season_whip=1.4)])

    result = build_pitcher_shrunk_whip(rolling, season, window='season', k=20.0)

    assert result['pitcher_shrunk_whip'].iloc[0] == pytest.approx(1.4)
    assert result['pitcher_shrunk_whip_weight'].iloc[0] == 0.0


def test_build_pitcher_shrunk_whip_falls_back_to_rolling_when_no_last_season_stat():
    """A rookie/unseen pitcher has no last-season row — baseline falls back
    to this season's own rolling WHIP rather than producing NaN."""

    rolling = pd.DataFrame([_rolling_row(pitcher_roll_season_whip=0.8, pitcher_roll_season_games_n=5)])
    season = pd.DataFrame([_season_row(pitcher_last_season_whip=None)])

    result = build_pitcher_shrunk_whip(rolling, season, window='season', k=20.0)

    assert result['pitcher_shrunk_whip'].iloc[0] == pytest.approx(0.8)


def test_build_pitcher_projected_workload_multiplies_batters_faced_before_pa_by_pace():
    """estimated_team_pa_position=4 -> 3 batters already faced before this
    PA. At a pace of 15 pitches/PA (trailing-3-game), the pitcher should
    have thrown 3 * 15 = 45 pitches heading into this PA."""

    df = pd.DataFrame([{
        'estimated_team_pa_position': 4,
        'pitcher_roll_last3g_pa_pitch_count_mean': 15.0,
    }])

    result = build_pitcher_projected_workload(df)

    assert result['pitcher_projected_pitches_before_pa'].iloc[0] == pytest.approx(45.0)


def test_build_pitcher_projected_workload_zero_at_first_pa_of_game():
    """estimated_team_pa_position=1 -> 0 batters faced before this PA ->
    0 pitches thrown yet, regardless of pace."""

    df = pd.DataFrame([{
        'estimated_team_pa_position': 1,
        'pitcher_roll_last3g_pa_pitch_count_mean': 15.0,
    }])

    result = build_pitcher_projected_workload(df)

    assert result['pitcher_projected_pitches_before_pa'].iloc[0] == pytest.approx(0.0)


def test_build_pitcher_projected_workload_nan_pace_propagates_to_nan():
    """A pitcher with no trailing-3-game pace yet (e.g. season opener) has
    no basis for a workload projection -- NaN, not a silently wrong 0."""

    df = pd.DataFrame([{
        'estimated_team_pa_position': 4,
        'pitcher_roll_last3g_pa_pitch_count_mean': np.nan,
    }])

    result = build_pitcher_projected_workload(df)

    assert pd.isna(result['pitcher_projected_pitches_before_pa'].iloc[0])
