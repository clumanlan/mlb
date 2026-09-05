import numpy as np
import pandas as pd
import pytest

from models.k_predictor.processing.features.in_game_context import (
    build_pitcher_in_game_running_stats,
    build_pitcher_in_game_hot_cold_gap,
)


def _pa_row(**overrides):
    row = {
        'gamepk': 'G1',
        'pitcher_id': 'P1',
        'play_id': 1,
        'is_strikeout': 0,
    }
    row.update(overrides)
    return row


# ── build_pitcher_in_game_running_stats ────────────────────────────────────

def test_running_stats_are_strictly_point_in_time_within_game():
    """4 PAs, one pitcher, one game, play_id 1-4, is_strikeout=[0,1,0,1].
    Each row's running stats must reflect only STRICTLY PRIOR PAs -- the
    current PA's own outcome must never be included in its own count."""
    df = pd.DataFrame([
        _pa_row(play_id=1, is_strikeout=0),
        _pa_row(play_id=2, is_strikeout=1),
        _pa_row(play_id=3, is_strikeout=0),
        _pa_row(play_id=4, is_strikeout=1),
    ])

    result = build_pitcher_in_game_running_stats(df).sort_values('play_id')

    assert result['pitcher_pa_faced_this_game_so_far'].tolist() == [0, 1, 2, 3]
    assert result['pitcher_k_this_game_so_far'].tolist() == [0, 0, 1, 1]
    k_rate = result['pitcher_k_this_game_so_far_rate'].tolist()
    assert pd.isna(k_rate[0])  # 0 PAs faced yet -- no rate to compute
    assert k_rate[1] == pytest.approx(0.0)
    assert k_rate[2] == pytest.approx(0.5)
    assert k_rate[3] == pytest.approx(1 / 3)


def test_running_stats_do_not_cross_contaminate_between_pitchers_same_game():
    """Home SP and away SP in the SAME gamepk must never see each other's
    counts -- the exact team/pitcher-scoping bug class already found
    elsewhere in this repo (build_batter_slot_expansion)."""
    df = pd.DataFrame([
        _pa_row(pitcher_id='HOME_SP', play_id=1, is_strikeout=1),
        _pa_row(pitcher_id='AWAY_SP', play_id=2, is_strikeout=1),
        _pa_row(pitcher_id='HOME_SP', play_id=3, is_strikeout=0),
        _pa_row(pitcher_id='AWAY_SP', play_id=4, is_strikeout=0),
    ])

    result = build_pitcher_in_game_running_stats(df)

    home_last = result[(result['pitcher_id'] == 'HOME_SP') & (result['play_id'] == 3)].iloc[0]
    away_last = result[(result['pitcher_id'] == 'AWAY_SP') & (result['play_id'] == 4)].iloc[0]

    assert home_last['pitcher_pa_faced_this_game_so_far'] == 1
    assert home_last['pitcher_k_this_game_so_far'] == 1
    assert away_last['pitcher_pa_faced_this_game_so_far'] == 1
    assert away_last['pitcher_k_this_game_so_far'] == 1


def test_running_stats_use_play_id_order_not_input_row_order():
    """Rows arrive scrambled (not in chronological order) but play_id still
    encodes the real order -- output must match the CHRONOLOGICAL
    expectation, not whatever order the rows happened to arrive in."""
    df = pd.DataFrame([
        _pa_row(play_id=3, is_strikeout=0),
        _pa_row(play_id=1, is_strikeout=1),
        _pa_row(play_id=2, is_strikeout=0),
    ])

    result = build_pitcher_in_game_running_stats(df)

    row_play_id_3 = result[result['play_id'] == 3].iloc[0]
    assert row_play_id_3['pitcher_pa_faced_this_game_so_far'] == 2
    assert row_play_id_3['pitcher_k_this_game_so_far'] == 1


def test_running_stats_reset_at_the_start_of_a_new_game_same_pitcher():
    """The same pitcher across two different starts (different gamepk) must
    not carry counts over from the earlier game."""
    df = pd.DataFrame([
        _pa_row(gamepk='G1', play_id=1, is_strikeout=1),
        _pa_row(gamepk='G1', play_id=2, is_strikeout=1),
        _pa_row(gamepk='G2', play_id=1, is_strikeout=0),
    ])

    result = build_pitcher_in_game_running_stats(df)

    g2_row = result[result['gamepk'] == 'G2'].iloc[0]
    assert g2_row['pitcher_pa_faced_this_game_so_far'] == 0
    assert g2_row['pitcher_k_this_game_so_far'] == 0


# ── build_pitcher_in_game_hot_cold_gap ─────────────────────────────────────

def test_hot_cold_gap_matches_hand_calculation():
    """pregame rate 0.25, 4 PAs faced so far, 2 strikeouts so far ->
    expected = 0.25 * 4 = 1.0, gap = 2 - 1.0 = 1.0 (running hot)."""
    df = pd.DataFrame([{
        'pitcher_roll_season_pa_strikeout_rate': 0.25,
        'pitcher_pa_faced_this_game_so_far': 4,
        'pitcher_k_this_game_so_far': 2,
    }])

    result = build_pitcher_in_game_hot_cold_gap(df)

    assert result['pitcher_expected_k_this_game_so_far'].iloc[0] == pytest.approx(1.0)
    assert result['pitcher_hot_cold_gap_this_game_so_far'].iloc[0] == pytest.approx(1.0)


def test_hot_cold_gap_is_zero_not_nan_at_first_pa_of_game():
    """0 PAs faced so far -> 0 expected K -> gap is a real 0, unlike the
    k_rate_so_far in build_pitcher_in_game_running_stats, which is NaN at
    this same point (no rate to compute vs. no gap to compute are different
    things -- 0 batters faced means both actual and expected K are truly 0,
    not undefined)."""
    df = pd.DataFrame([{
        'pitcher_roll_season_pa_strikeout_rate': 0.30,
        'pitcher_pa_faced_this_game_so_far': 0,
        'pitcher_k_this_game_so_far': 0,
    }])

    result = build_pitcher_in_game_hot_cold_gap(df)

    assert result['pitcher_expected_k_this_game_so_far'].iloc[0] == pytest.approx(0.0)
    assert result['pitcher_hot_cold_gap_this_game_so_far'].iloc[0] == pytest.approx(0.0)


def test_hot_cold_gap_nan_pregame_rate_propagates_to_nan_not_zero():
    """A pitcher with no pre-game rolling K-rate yet has no basis for an
    'expected' baseline -- NaN, not a silently-wrong 0."""
    df = pd.DataFrame([{
        'pitcher_roll_season_pa_strikeout_rate': np.nan,
        'pitcher_pa_faced_this_game_so_far': 3,
        'pitcher_k_this_game_so_far': 1,
    }])

    result = build_pitcher_in_game_hot_cold_gap(df)

    assert pd.isna(result['pitcher_expected_k_this_game_so_far'].iloc[0])
    assert pd.isna(result['pitcher_hot_cold_gap_this_game_so_far'].iloc[0])
