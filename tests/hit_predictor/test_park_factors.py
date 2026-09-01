import pandas as pd
import pytest

from models.hit_predictor.processing.features.park_factors import (
    _create_venue_hit_factor,
    _create_venue_strikeout_factor,
    build_park_factors,
    build_park_strikeout_factor,
)


def _make_schedule():
    return pd.DataFrame([
        {'gamepk': 'g1', 'venue_id': 'V1', 'game_date': pd.Timestamp('2023-04-01')},
        {'gamepk': 'g2', 'venue_id': 'V1', 'game_date': pd.Timestamp('2023-04-02')},
        {'gamepk': 'g3', 'venue_id': 'V2', 'game_date': pd.Timestamp('2023-04-01')},
        {'gamepk': 'g4', 'venue_id': 'V2', 'game_date': pd.Timestamp('2023-04-02')},
    ])


def _make_batter_boxscore():
    # V1 games: 8 hits / 16 AB combined = 0.500 hit rate
    # V2 games: 4 hits / 16 AB combined = 0.250 hit rate
    # League (both venues): 12 hits / 32 AB = 0.375 hit rate
    return pd.DataFrame([
        {'gamepk': 'g1', 'personId': 'A', 'h': 2, 'ab': 4},
        {'gamepk': 'g1', 'personId': 'B', 'h': 2, 'ab': 4},
        {'gamepk': 'g2', 'personId': 'A', 'h': 3, 'ab': 4},
        {'gamepk': 'g2', 'personId': 'B', 'h': 1, 'ab': 4},
        {'gamepk': 'g3', 'personId': 'C', 'h': 1, 'ab': 4},
        {'gamepk': 'g3', 'personId': 'D', 'h': 1, 'ab': 4},
        {'gamepk': 'g4', 'personId': 'C', 'h': 1, 'ab': 4},
        {'gamepk': 'g4', 'personId': 'D', 'h': 1, 'ab': 4},
    ])


def test_create_venue_hit_factor_hitter_friendly_park_above_one():
    """V1's combined hit rate (0.500) is above the league rate (0.375) across
    both venues that season, so its factor should be > 1 — a hitter-friendly
    park. Hand-computed: 0.500 / 0.375 = 1.3333..."""

    result = _create_venue_hit_factor(_make_schedule(), _make_batter_boxscore())
    row = result[(result['venue_id'] == 'V1') & (result['game_season'] == 2023)].iloc[0]

    assert row['park_season_hit_factor'] == pytest.approx(4 / 3)


def test_create_venue_hit_factor_pitcher_friendly_park_below_one():
    """V2's combined hit rate (0.250) is below the league rate (0.375), so
    its factor should be < 1 — a pitcher-friendly park. Hand-computed:
    0.250 / 0.375 = 0.6667..."""

    result = _create_venue_hit_factor(_make_schedule(), _make_batter_boxscore())
    row = result[(result['venue_id'] == 'V2') & (result['game_season'] == 2023)].iloc[0]

    assert row['park_season_hit_factor'] == pytest.approx(2 / 3)


def test_create_venue_hit_factor_guards_zero_ab():
    """A venue-season with zero recorded AB (e.g. a single rained-out/
    forfeited game with no official at-bats) must not raise a
    divide-by-zero error — it should produce NaN instead."""

    schedule = pd.DataFrame([{'gamepk': 'g1', 'venue_id': 'V1', 'game_date': pd.Timestamp('2023-04-01')}])
    batter_boxscore = pd.DataFrame([{'gamepk': 'g1', 'personId': 'A', 'h': 0, 'ab': 0}])

    result = _create_venue_hit_factor(schedule, batter_boxscore)

    assert result['park_season_hit_factor'].isna().all()


def _make_pitcher_boxscore():
    # V1 games: 8 k / 16 ip combined = 0.500 k/ip
    # V2 games: 4 k / 16 ip combined = 0.250 k/ip
    # League (both venues): 12 k / 32 ip = 0.375 k/ip
    return pd.DataFrame([
        {'gamepk': 'g1', 'personId': 'A', 'k': 2, 'ip': 4},
        {'gamepk': 'g1', 'personId': 'B', 'k': 2, 'ip': 4},
        {'gamepk': 'g2', 'personId': 'A', 'k': 3, 'ip': 4},
        {'gamepk': 'g2', 'personId': 'B', 'k': 1, 'ip': 4},
        {'gamepk': 'g3', 'personId': 'C', 'k': 1, 'ip': 4},
        {'gamepk': 'g3', 'personId': 'D', 'k': 1, 'ip': 4},
        {'gamepk': 'g4', 'personId': 'C', 'k': 1, 'ip': 4},
        {'gamepk': 'g4', 'personId': 'D', 'k': 1, 'ip': 4},
    ])


def test_create_venue_strikeout_factor_strikeout_friendly_park_above_one():
    """V1's combined K rate (0.500 k/ip) is above the league rate (0.375
    k/ip) that season, so its factor should be > 1 — a strikeout-friendly
    park. Hand-computed: 0.500 / 0.375 = 1.3333..."""

    result = _create_venue_strikeout_factor(_make_schedule(), _make_pitcher_boxscore())
    row = result[(result['venue_id'] == 'V1') & (result['game_season'] == 2023)].iloc[0]

    assert row['park_season_strikeout_factor'] == pytest.approx(4 / 3)


def test_create_venue_strikeout_factor_strikeout_suppressing_park_below_one():
    """V2's combined K rate (0.250 k/ip) is below the league rate (0.375
    k/ip), so its factor should be < 1 — a strikeout-suppressing park.
    Hand-computed: 0.250 / 0.375 = 0.6667..."""

    result = _create_venue_strikeout_factor(_make_schedule(), _make_pitcher_boxscore())
    row = result[(result['venue_id'] == 'V2') & (result['game_season'] == 2023)].iloc[0]

    assert row['park_season_strikeout_factor'] == pytest.approx(2 / 3)


def test_create_venue_strikeout_factor_guards_zero_ip():
    """A venue-season with zero recorded IP must not raise a divide-by-zero
    error — it should produce NaN instead."""

    schedule = pd.DataFrame([{'gamepk': 'g1', 'venue_id': 'V1', 'game_date': pd.Timestamp('2023-04-01')}])
    pitcher_boxscore = pd.DataFrame([{'gamepk': 'g1', 'personId': 'A', 'k': 0, 'ip': 0}])

    result = _create_venue_strikeout_factor(schedule, pitcher_boxscore)

    assert result['park_season_strikeout_factor'].isna().all()


def test_build_park_strikeout_factor_shifts_to_next_season():
    """Same point-in-time-safe convention as build_park_factors."""

    result = build_park_strikeout_factor(_make_schedule(), _make_pitcher_boxscore())
    row = result[result['venue_id'] == 'V1'].iloc[0]

    assert row['game_season'] == 2024
    assert 'park_last_season_strikeout_factor' in result.columns
    assert 'park_season_strikeout_factor' not in result.columns
    assert row['park_last_season_strikeout_factor'] == pytest.approx(4 / 3)


def test_build_park_factors_shifts_to_next_season():
    """Park factors follow the same point-in-time-safe convention as every
    other season-level feature in this repo (_shift_to_last_season): a
    factor computed from 2023 games can only join onto 2024 games, and the
    raw 'season_' column name becomes 'last_season_' so it's unambiguous
    which season a row is meant to be joined onto."""

    result = build_park_factors(_make_schedule(), _make_batter_boxscore())
    row = result[result['venue_id'] == 'V1'].iloc[0]

    assert row['game_season'] == 2024
    assert 'park_last_season_hit_factor' in result.columns
    assert 'park_season_hit_factor' not in result.columns
    assert row['park_last_season_hit_factor'] == pytest.approx(4 / 3)
