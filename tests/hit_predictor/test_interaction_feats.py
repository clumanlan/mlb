import pandas as pd
import pytest

import numpy as np

from models.hit_predictor.processing.features.interaction_feats import (
    find_rolling_trend_pairs,
    find_sample_size_col,
    build_trend_features,
    build_shrinkage_weight_features,
)


def test_find_rolling_trend_pairs_matches_same_entity_and_stat():
    """A short-window column pairs with its season counterpart only when
    both the entity prefix (batter/pitcher) and the stat name match."""

    columns = ['batter_roll_season_ba', 'batter_roll_last10g_ba']

    result = find_rolling_trend_pairs(columns)

    assert result == [('batter_roll_last10g_ba', 'batter_roll_season_ba')]


def test_find_rolling_trend_pairs_skips_columns_missing_a_counterpart():
    """A season-only or short-only column (no matching pair) must not be
    returned — PDP-ing a trend needs both axes to exist."""

    columns = ['batter_roll_season_ba', 'batter_roll_last10g_iso', 'batting_order']

    result = find_rolling_trend_pairs(columns)

    assert result == []


def test_find_rolling_trend_pairs_does_not_cross_entities():
    """batter's season stat must not pair with pitcher's short-window stat
    even if the stat name happens to match."""

    columns = ['batter_roll_season_ba', 'pitcher_roll_last10g_ba']

    result = find_rolling_trend_pairs(columns)

    assert result == []


def test_find_rolling_trend_pairs_matches_any_short_window_size():
    """The short window's game count is a variable (10, 5, whatever a given
    experiment configures) — the regex must not hardcode '10'."""

    columns = ['pitcher_roll_season_whip', 'pitcher_roll_last5g_whip']

    result = find_rolling_trend_pairs(columns)

    assert result == [('pitcher_roll_last5g_whip', 'pitcher_roll_season_whip')]


def test_find_rolling_trend_pairs_handles_multi_underscore_stat_names():
    """Stat names themselves contain underscores (e.g. pa_hit_rate) — the
    stat-matching group must capture the full remainder, not stop early."""

    columns = ['batter_roll_season_pa_hit_rate', 'batter_roll_last10g_pa_hit_rate']

    result = find_rolling_trend_pairs(columns)

    assert result == [('batter_roll_last10g_pa_hit_rate', 'batter_roll_season_pa_hit_rate')]


def test_find_sample_size_col_prefers_plate_appearances():
    columns = ['batter_roll_last10g_ba', 'batter_roll_last10g_plate_appearances', 'batter_roll_last10g_ab']

    result = find_sample_size_col(columns, rate_col='batter_roll_last10g_ba')

    assert result == 'batter_roll_last10g_plate_appearances'


def test_find_sample_size_col_falls_back_to_pa_total_for_pbp_derived_rates():
    """pbp-derived rolling rates (e.g. o_swing_rate) have no plate_appearances/ab
    column — pa_total (the pbp PA-outcome denominator) is the fallback."""

    columns = ['batter_roll_last10g_o_swing_rate', 'batter_roll_last10g_pa_total', 'batter_roll_last10g_n_pitches']

    result = find_sample_size_col(columns, rate_col='batter_roll_last10g_o_swing_rate')

    assert result == 'batter_roll_last10g_pa_total'


def test_find_sample_size_col_returns_none_when_no_denominator_present():
    columns = ['batter_roll_last10g_ba']

    result = find_sample_size_col(columns, rate_col='batter_roll_last10g_ba')

    assert result is None


def test_find_sample_size_col_matches_season_window_prefix_too():
    columns = ['pitcher_roll_season_whip', 'pitcher_roll_season_ip']

    result = find_sample_size_col(columns, rate_col='pitcher_roll_season_whip')

    assert result == 'pitcher_roll_season_ip'


def test_find_sample_size_col_never_returns_the_rate_col_itself():
    """batter_roll_last10g_plate_appearances is itself in _SAMPLE_SIZE_SUFFIXES
    — asking for ITS OWN sample-size column must not match itself (a
    degenerate self-pair), even though it's a valid denominator for every
    other rate in the same window. Found via the real PDP diagnostic run,
    which produced a shrinkage_plate_appearances_x_plate_appearances.png."""

    columns = ['batter_roll_last10g_plate_appearances']

    result = find_sample_size_col(columns, rate_col='batter_roll_last10g_plate_appearances')

    assert result is None


def test_build_trend_features_ratio_is_short_divided_by_season():
    """batter_trend_ratio_ba = last10g_ba / season_ba — >1 running hotter
    than season baseline, <1 colder, 1.0 exactly on pace."""

    df = pd.DataFrame({
        'batter_roll_last10g_ba': [0.400, 0.125],
        'batter_roll_season_ba': [0.250, 0.250],
    })
    pairs = [('batter_roll_last10g_ba', 'batter_roll_season_ba')]

    result = build_trend_features(df, pairs)

    assert result['batter_trend_ratio_ba'].tolist() == pytest.approx([1.6, 0.5])


def test_build_trend_features_ratio_guards_divide_by_zero():
    """A rookie with zero rolled season at-bats behind them (season_ba
    computed as 0/0 -> NaN already, or a genuine 0.0) must not blow up the
    ratio into inf — NaN instead."""

    df = pd.DataFrame({
        'batter_roll_last10g_ba': [0.300],
        'batter_roll_season_ba': [0.0],
    })
    pairs = [('batter_roll_last10g_ba', 'batter_roll_season_ba')]

    result = build_trend_features(df, pairs)

    assert pd.isna(result['batter_trend_ratio_ba'].iloc[0])


def test_build_trend_features_direction_is_sign_of_the_difference():
    """batter_trend_direction_ba is a coarse hot/cold/flat indicator:
    +1 when short > season, -1 when short < season, 0 when exactly equal —
    deliberately throws away the magnitude that made the plain diff feature
    (build_trend_diff_features, removed) get overused by the RF without
    improving held-out metrics."""

    df = pd.DataFrame({
        'batter_roll_last10g_ba': [0.350, 0.100, 0.250],
        'batter_roll_season_ba': [0.250, 0.250, 0.250],
    })
    pairs = [('batter_roll_last10g_ba', 'batter_roll_season_ba')]

    result = build_trend_features(df, pairs)

    assert result['batter_trend_direction_ba'].tolist() == [1.0, -1.0, 0.0]


def test_build_trend_features_preserves_original_columns():
    df = pd.DataFrame({
        'batter_roll_last10g_ba': [0.300],
        'batter_roll_season_ba': [0.250],
        'unrelated_col': [1],
    })
    pairs = [('batter_roll_last10g_ba', 'batter_roll_season_ba')]

    result = build_trend_features(df, pairs)

    assert 'batter_roll_last10g_ba' in result.columns
    assert 'unrelated_col' in result.columns


def test_build_shrinkage_weight_features_weight_is_half_at_k_games():
    """weight = sample / (sample + k) — at sample == k the weight is exactly
    0.5, the crossover point between 'mostly discount this rate' and
    'mostly trust it'."""

    df = pd.DataFrame({
        'batter_roll_last10g_ba': [0.300],
        'batter_roll_last10g_plate_appearances': [10],
    })
    rate_to_sample_col = {'batter_roll_last10g_ba': 'batter_roll_last10g_plate_appearances'}

    result = build_shrinkage_weight_features(df, rate_to_sample_col, k=10.0)

    assert result['batter_shrinkage_weight_ba'].iloc[0] == pytest.approx(0.5)


def test_build_shrinkage_weight_features_shrunk_value_is_rate_times_weight():
    """shrunk = rate * weight — a rate built on a small sample gets pulled
    toward 0 (not toward the season mean, which this function has no access
    to; that's the tree's job to combine with the still-present season
    column), a rate built on a large sample stays close to its raw value."""

    df = pd.DataFrame({
        'batter_roll_last10g_ba': [0.300],
        'batter_roll_last10g_plate_appearances': [90],
    })
    rate_to_sample_col = {'batter_roll_last10g_ba': 'batter_roll_last10g_plate_appearances'}

    result = build_shrinkage_weight_features(df, rate_to_sample_col, k=10.0)

    # weight = 90 / (90 + 10) = 0.9 -> shrunk = 0.3 * 0.9 = 0.27
    assert result['batter_shrunk_ba'].iloc[0] == pytest.approx(0.27)


def test_build_shrinkage_weight_features_preserves_original_columns():
    df = pd.DataFrame({
        'batter_roll_last10g_ba': [0.300],
        'batter_roll_last10g_plate_appearances': [90],
        'unrelated_col': [1],
    })
    rate_to_sample_col = {'batter_roll_last10g_ba': 'batter_roll_last10g_plate_appearances'}

    result = build_shrinkage_weight_features(df, rate_to_sample_col, k=10.0)

    assert 'batter_roll_last10g_ba' in result.columns
    assert 'unrelated_col' in result.columns
