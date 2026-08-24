import numpy as np
import pandas as pd
import pytest

from models.hit_predictor.processing.features.tier1_feats import (
    build_log5_matchup_features,
    build_velocity_decline_trend_features,
    pitch_type_entropy,
)


def test_pitch_type_entropy_even_two_way_split_is_one_bit():
    """Shannon entropy (base 2) of an evenly split 2-outcome distribution is
    exactly 1.0 bit — a hand-computable literal, independent of how the
    implementation computes it: H = -sum(p*log2(p)) = -(0.5*log2(0.5)*2) = 1.0."""

    pitch_types = pd.Series(['FF', 'FF', 'SL', 'SL'])

    assert pitch_type_entropy(pitch_types) == pytest.approx(1.0)


def test_pitch_type_entropy_single_pitch_type_is_zero():
    """A pitcher who only ever throws one pitch type is maximally
    predictable — zero entropy, the other end of the scale from the
    even-split case above."""

    pitch_types = pd.Series(['FF', 'FF', 'FF'])

    assert pitch_type_entropy(pitch_types) == pytest.approx(0.0)


def test_build_log5_matchup_features_computes_batter_x_pitcher_over_league():
    """log5_matchup_single = batter_rate * pitcher_rate / league_rate, a
    hand-computable formula independent of the implementation:
    0.30 * 0.25 / 0.20 = 0.375."""

    df = pd.DataFrame({
        'batter_season_pa_single_rate': [0.30],
        'pitcher_season_pa_single_rate': [0.25],
        'league_season_pa_single_rate': [0.20],
    })

    result = build_log5_matchup_features(df, outcomes=['single'])

    assert result['log5_matchup_single'].iloc[0] == pytest.approx(0.375)


def test_build_log5_matchup_features_guards_against_zero_league_rate():
    """A zero league rate (degenerate/thin-sample edge case) must produce
    NaN, not a ZeroDivisionError or inf that would silently poison
    downstream model training."""

    df = pd.DataFrame({
        'batter_season_pa_hbp_rate': [0.05],
        'pitcher_season_pa_hbp_rate': [0.03],
        'league_season_pa_hbp_rate': [0.0],
    })

    result = build_log5_matchup_features(df, outcomes=['hbp'])

    assert np.isnan(result['log5_matchup_hbp'].iloc[0])


def test_build_log5_matchup_features_preserves_original_columns():
    df = pd.DataFrame({
        'batter_season_pa_single_rate': [0.30],
        'pitcher_season_pa_single_rate': [0.25],
        'league_season_pa_single_rate': [0.20],
        'unrelated_col': ['x'],
    })

    result = build_log5_matchup_features(df, outcomes=['single'])

    assert 'unrelated_col' in result.columns
    assert result['unrelated_col'].iloc[0] == 'x'


def test_build_velocity_decline_trend_features_computes_ratio_and_direction():
    """short/season = 96.0/94.0, a hand-computable literal independent of
    the trend machinery's own implementation; direction = sign(96-94) = 1
    (hot, i.e. throwing harder recently than the season baseline — a
    velocity DROP would show as trend_ratio < 1 / direction -1, the
    fatigue-decline signal this feature exists to surface)."""

    df = pd.DataFrame({
        'pitcher_roll_last10g_stuff_start_speed_mean': [96.0],
        'pitcher_roll_season_stuff_start_speed_mean': [94.0],
    })

    result = build_velocity_decline_trend_features(df)

    assert result['pitcher_trend_ratio_stuff_start_speed_mean'].iloc[0] == pytest.approx(96.0 / 94.0)
    assert result['pitcher_trend_direction_stuff_start_speed_mean'].iloc[0] == 1.0


def test_build_velocity_decline_trend_features_ignores_unrelated_rolling_pairs():
    """Unlike v3_interaction_feats/train.py's blanket
    find_rolling_trend_pairs(all columns) approach — found to hurt val
    metrics when applied to every rolling stat — this is deliberately
    narrow: an unrelated rolling pair (a PA outcome rate, not velocity/spin)
    must NOT get a trend column."""

    df = pd.DataFrame({
        'pitcher_roll_last10g_stuff_start_speed_mean': [96.0],
        'pitcher_roll_season_stuff_start_speed_mean': [94.0],
        'pitcher_roll_last10g_pa_hit_rate': [0.30],
        'pitcher_roll_season_pa_hit_rate': [0.25],
    })

    result = build_velocity_decline_trend_features(df)

    assert 'pitcher_trend_ratio_pa_hit_rate' not in result.columns
    assert 'pitcher_trend_direction_pa_hit_rate' not in result.columns


def test_build_velocity_decline_trend_features_covers_spin_rate_too():
    df = pd.DataFrame({
        'pitcher_roll_last10g_stuff_spin_rate_mean': [2100.0],
        'pitcher_roll_season_stuff_spin_rate_mean': [2200.0],
    })

    result = build_velocity_decline_trend_features(df)

    assert result['pitcher_trend_ratio_stuff_spin_rate_mean'].iloc[0] == pytest.approx(2100.0 / 2200.0)
    assert result['pitcher_trend_direction_stuff_spin_rate_mean'].iloc[0] == -1.0
