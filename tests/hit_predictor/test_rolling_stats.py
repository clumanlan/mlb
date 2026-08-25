import numpy as np
import pandas as pd
import pytest

from models.hit_predictor.processing.features.rolling_stats import (
    _rolling_sum,
    _rolling_max,
    _rolling_pooled_std,
    _finalize_rates,
    _validate_window,
    build_batter_rolling_stats,
    build_pitcher_rolling_stats,
    build_pitcher_rolling_stats_all_roles,
    build_pbp_pitcher_rolling_feats,
    build_pbp_pitcher_rolling_feats_all_roles,
    build_pbp_batter_rolling_feats,
    build_team_batter_strikeout_rolling_feats,
)


def _row(personId='1', game_date='2023-04-01', game_season=2023, **overrides):
    row = {'personId': personId, 'game_date': pd.Timestamp(game_date), 'game_season': game_season}
    row.update(overrides)
    return row


def test_rolling_sum_excludes_current_game_own_stats():
    """A 4-hit game shouldn't see that 4 baked into its own rolling feature —
    the shift(1) point. First game of a player's history has nothing prior,
    so its rolled value must be NaN, not 0 or the game's own stat."""

    df = pd.DataFrame([
        _row(game_date='2023-04-01', h=4),
        _row(game_date='2023-04-02', h=1),
    ])

    result = _rolling_sum(df, entity_col='personId', cols=['h'], window=10)

    assert pd.isna(result.loc[result['game_date'] == '2023-04-01', 'h'].iloc[0])
    assert result.loc[result['game_date'] == '2023-04-02', 'h'].iloc[0] == 4


def test_rolling_sum_short_window_sums_exactly_trailing_n_games():
    """window=3 must sum exactly the 3 games immediately before the current
    one — not 2 (off-by-one short) and not 4 (off-by-one long)."""

    df = pd.DataFrame([
        _row(game_date='2023-04-01', h=1),
        _row(game_date='2023-04-02', h=2),
        _row(game_date='2023-04-03', h=4),
        _row(game_date='2023-04-04', h=8),
        _row(game_date='2023-04-05', h=16),
    ])

    result = _rolling_sum(df, entity_col='personId', cols=['h'], window=3)

    # game 5's rolling-3 window should be games 2,3,4 = 2+4+8 = 14
    row5 = result.loc[result['game_date'] == '2023-04-05']
    assert row5['h'].iloc[0] == 14


def test_rolling_sum_min_periods_one_uses_partial_history():
    """A player's 3rd career game with window=10 should reflect the 2 prior
    games it actually has, not NaN until 10 games have accumulated."""

    df = pd.DataFrame([
        _row(game_date='2023-04-01', h=1),
        _row(game_date='2023-04-02', h=2),
        _row(game_date='2023-04-03', h=4),
    ])

    result = _rolling_sum(df, entity_col='personId', cols=['h'], window=10)

    row3 = result.loc[result['game_date'] == '2023-04-03']
    assert row3['h'].iloc[0] == 3  # games 1+2 = 1+2


def test_rolling_sum_season_mode_resets_at_season_boundary():
    """Season-to-date must not carry last season's accumulated stats into
    the first game of a new season, even with a long prior-season history."""

    df = pd.DataFrame([
        _row(game_date='2023-09-01', game_season=2023, h=100),
        _row(game_date='2024-04-01', game_season=2024, h=1),
    ])

    result = _rolling_sum(df, entity_col='personId', cols=['h'], window='season')

    row_2024 = result.loc[result['game_date'] == '2024-04-01']
    assert pd.isna(row_2024['h'].iloc[0])


def test_rolling_sum_sort_col_orders_same_date_rows_by_finer_grained_column():
    """game_date has no time component — two rows sharing the same date
    (e.g. both games of a doubleheader) can't be reliably ordered by it
    alone. sort_col lets a caller pass a finer-grained column (e.g.
    game_datetime) so the earlier game's rolled value doesn't leak the
    later game's own-day result. Default behavior (sort_col unset) is
    unchanged — this only matters when a caller opts in."""

    df = pd.DataFrame([
        _row(game_date='2023-04-01', game_datetime=pd.Timestamp('2023-04-01 19:00'), h=1),
        # Doubleheader day: two rows sharing the same game_date, deliberately
        # inserted in reverse chronological order (game 2's timestamp first)
        # to prove sort_col, not row order, controls the result.
        _row(game_date='2023-04-13', game_datetime=pd.Timestamp('2023-04-13 22:00'), h=100),  # game 2
        _row(game_date='2023-04-13', game_datetime=pd.Timestamp('2023-04-13 16:00'), h=10),   # game 1
    ])

    result = _rolling_sum(df, entity_col='personId', cols=['h'], window=10, sort_col='game_datetime')

    game1 = result.loc[result['game_datetime'] == pd.Timestamp('2023-04-13 16:00')]
    game2 = result.loc[result['game_datetime'] == pd.Timestamp('2023-04-13 22:00')]
    # game 1 must roll forward only the prior day's h=1 — not game 2's h=100,
    # which hasn't happened yet relative to game 1 despite sharing a date
    assert game1['h'].iloc[0] == 1
    # game 2 (later that same day) correctly picks up game 1's h=10 too
    assert game2['h'].iloc[0] == 1 + 10


def test_rolling_sum_short_window_carries_across_season_boundary():
    """Unlike season-to-date, a fixed N-game window has no reason to reset
    at a season boundary — a player's early-April window should still
    include games from the tail of last season."""

    df = pd.DataFrame([
        _row(game_date='2023-09-01', game_season=2023, h=10),
        _row(game_date='2024-04-01', game_season=2024, h=1),
    ])

    result = _rolling_sum(df, entity_col='personId', cols=['h'], window=5)

    row_2024 = result.loc[result['game_date'] == '2024-04-01']
    assert row_2024['h'].iloc[0] == 10


def test_rolling_max_is_max_of_per_game_maxes():
    """Rolling max over a window of per-game maxes must equal the true max
    across every raw value in that window — max-of-maxes is exact, unlike
    sum or mean, so no decomposition into finer components is needed."""

    df = pd.DataFrame([
        _row(game_date='2023-04-01', velo_max=95.0),
        _row(game_date='2023-04-02', velo_max=91.0),
        _row(game_date='2023-04-03', velo_max=97.0),
        _row(game_date='2023-04-04', velo_max=93.0),
    ])

    result = _rolling_max(df, entity_col='personId', cols=['velo_max'], window=10)

    row4 = result.loc[result['game_date'] == '2023-04-04']
    assert row4['velo_max'].iloc[0] == 97.0  # max of games 1-3: 95, 91, 97


def test_rolling_pooled_std_matches_true_std_of_underlying_raw_values():
    """An exact rolling std can't be derived from per-game std alone —
    pooled variance needs (n, sum, sum_of_squares) per game. This test
    proves the pooled-from-summary-stats result equals the std you'd get
    computing directly over every raw pitch value in the window."""

    game_a_pitches = [90.0, 92.0]   # date 1
    game_b_pitches = [88.0]         # date 2
    game_c_pitches = [95.0, 96.0, 94.0]  # date 3 — the one we check

    def _summary(pitches):
        n = len(pitches)
        s = sum(pitches)
        sq = sum(p ** 2 for p in pitches)
        return n, s, sq

    n_a, sum_a, sq_a = _summary(game_a_pitches)
    n_b, sum_b, sq_b = _summary(game_b_pitches)
    n_c, sum_c, sq_c = _summary(game_c_pitches)

    df = pd.DataFrame([
        _row(game_date='2023-04-01', n_pitches=n_a, speed_sum=sum_a, speed_sumsq=sq_a),
        _row(game_date='2023-04-02', n_pitches=n_b, speed_sum=sum_b, speed_sumsq=sq_b),
        _row(game_date='2023-04-03', n_pitches=n_c, speed_sum=sum_c, speed_sumsq=sq_c),
    ])

    result = _rolling_pooled_std(
        df, entity_col='personId',
        n_col='n_pitches', sum_col='speed_sum', sumsq_col='speed_sumsq',
        window=10, out_col='speed_std',
    )

    row3 = result.loc[result['game_date'] == '2023-04-03']
    expected = pd.Series(game_a_pitches + game_b_pitches).std()  # true std of prior raw pitches
    assert np.isclose(row3['speed_std'].iloc[0], expected)


def test_rolling_pooled_std_is_nan_when_fewer_than_two_prior_pitches():
    """Sample std is undefined with n<=1 prior pitches (division by n-1=0) —
    must yield NaN, not a ZeroDivisionError or inf."""

    df = pd.DataFrame([
        _row(game_date='2023-04-01', n_pitches=1, speed_sum=90.0, speed_sumsq=8100.0),
        _row(game_date='2023-04-02', n_pitches=1, speed_sum=91.0, speed_sumsq=8281.0),
    ])

    result = _rolling_pooled_std(
        df, entity_col='personId',
        n_col='n_pitches', sum_col='speed_sum', sumsq_col='speed_sumsq',
        window=10, out_col='speed_std',
    )

    row2 = result.loc[result['game_date'] == '2023-04-02']
    assert pd.isna(row2['speed_std'].iloc[0])


def test_finalize_rates_uses_rolled_sums_not_averaged_per_game_rates():
    """Game 1 is 1-for-1 (rate 1.0), game 2 is 0-for-4 (rate 0.0). Naive
    averaging of those two per-game rates gives 0.5. The correct answer,
    summing counts first, is (1+0)/(1+4) = 0.2 — this test fails if
    _finalize_rates (or whatever feeds it) ever averages per-game rates
    instead of dividing rolled sums."""

    df = pd.DataFrame([
        _row(game_date='2023-04-01', hits=1, ab=1),
        _row(game_date='2023-04-02', hits=0, ab=4),
        _row(game_date='2023-04-03', hits=0, ab=0),
    ])

    rolled = _rolling_sum(df, entity_col='personId', cols=['hits', 'ab'], window=10)
    result = _finalize_rates(rolled, {'hit_rate': ('hits', 'ab')})

    row3 = result.loc[result['game_date'] == '2023-04-03']
    assert row3['hit_rate'].iloc[0] == pytest.approx(0.2)


def test_finalize_rates_guards_divide_by_zero():
    """A player with zero rolled at-bats (e.g. two straight pinch-runner-only
    games) must get NaN, not a ZeroDivisionError or inf, for a rate stat."""

    df = pd.DataFrame([
        _row(game_date='2023-04-01', hits=0, ab=0),
        _row(game_date='2023-04-02', hits=0, ab=0),
    ])

    rolled = _rolling_sum(df, entity_col='personId', cols=['hits', 'ab'], window=10)
    result = _finalize_rates(rolled, {'hit_rate': ('hits', 'ab')})

    row2 = result.loc[result['game_date'] == '2023-04-02']
    assert pd.isna(row2['hit_rate'].iloc[0])


def test_validate_window_accepts_season_and_positive_int():
    _validate_window('season')
    _validate_window(10)


@pytest.mark.parametrize('bad_window', [0, -5, 'bogus', 3.5])
def test_validate_window_rejects_invalid_values(bad_window):
    with pytest.raises(ValueError):
        _validate_window(bad_window)


def _batter_box_row(personId='1', gamepk='g1', game_date='2023-04-01', game_season=2023, **overrides):
    row = {
        'personId': personId, 'gamepk': gamepk,
        'game_date': pd.Timestamp(game_date), 'game_season': game_season,
        'h': 0, 'k': 0, 'bb': 0, 'hr': 0, 'ab': 0, 'plate_appearances': 0, 'total_bases_from_h': 0,
    }
    row.update(overrides)
    return row


def test_build_batter_rolling_stats_ba_uses_rolled_sums_not_averaged_per_game_rates():
    """Game 1 is 1-for-1 (ba 1.0), game 2 is 0-for-4 (ba 0.0). The correct
    rolled ba going into game 3 is (1+0)/(1+4)=0.2, not the naive per-game
    average of 0.5 — same failure mode as the pure-engine finalize test,
    exercised now through the actual public entry point."""

    df = pd.DataFrame([
        _batter_box_row(gamepk='g1', game_date='2023-04-01', h=1, ab=1),
        _batter_box_row(gamepk='g2', game_date='2023-04-02', h=0, ab=4),
        _batter_box_row(gamepk='g3', game_date='2023-04-03', h=0, ab=0),
    ])

    result = build_batter_rolling_stats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['batter_roll_last10g_ba'].iloc[0] == pytest.approx(0.2)


def test_build_batter_rolling_stats_iso_and_babip_from_rolled_sums():
    """iso = slg - ba, babip = (h-hr)/(ab-k-hr) — both composite formulas
    must be computed from already-rolled component sums, mirroring
    season_stats.py's _create_boxscore_batter_stats exactly."""

    df = pd.DataFrame([
        _batter_box_row(gamepk='g1', game_date='2023-04-01', h=2, ab=4, hr=1, k=1, total_bases_from_h=6),
        _batter_box_row(gamepk='g2', game_date='2023-04-02'),
    ])

    result = build_batter_rolling_stats(df, window=10)

    row2 = result.loc[result['gamepk'] == 'g2']
    # rolled: h=2, ab=4, hr=1, k=1, total_bases=6 -> ba=0.5, slg=1.5, iso=1.0
    assert row2['batter_roll_last10g_ba'].iloc[0] == pytest.approx(0.5)
    assert row2['batter_roll_last10g_slg'].iloc[0] == pytest.approx(1.5)
    assert row2['batter_roll_last10g_iso'].iloc[0] == pytest.approx(1.0)
    # babip = (h-hr)/(ab-k-hr) = (2-1)/(4-1-1) = 1/2 = 0.5
    assert row2['batter_roll_last10g_babip'].iloc[0] == pytest.approx(0.5)


def test_build_batter_rolling_stats_season_mode_uses_roll_season_prefix():
    """window='season' must produce batter_roll_season_* columns, distinct
    from both season_stats.py's batter_season_*/_last_season_* columns and
    from this module's own batter_roll_last{N}g_* short-window columns."""

    df = pd.DataFrame([
        _batter_box_row(gamepk='g1', game_date='2023-04-01', h=1, ab=1),
        _batter_box_row(gamepk='g2', game_date='2023-04-02'),
    ])

    result = build_batter_rolling_stats(df, window='season')

    assert 'batter_roll_season_ba' in result.columns
    assert 'batter_roll_last10g_ba' not in result.columns
    assert 'batter_season_ba' not in result.columns


def test_build_batter_rolling_stats_preserves_key_columns():
    df = pd.DataFrame([_batter_box_row()])

    result = build_batter_rolling_stats(df, window=10)

    for key_col in ('personId', 'gamepk', 'game_date', 'game_season'):
        assert key_col in result.columns
    assert 'h' not in result.columns  # raw stat col replaced by prefixed rolled version


def _pitcher_box_row(personId='1', gamepk='g1', game_date='2023-04-01', game_season=2023, **overrides):
    row = {
        'personId': personId, 'gamepk': gamepk,
        'game_date': pd.Timestamp(game_date), 'game_season': game_season,
        'h': 0, 'r': 0, 'er': 0, 'bb': 0, 'hr': 0, 'k': 0, 'p': 0, 's': 0, 'ip': 0.0,
    }
    row.update(overrides)
    return row


def test_build_pitcher_rolling_stats_whip_from_rolled_bb_and_h():
    """whip = (bb+h)/ip, a composite numerator across two rolled columns —
    must equal (rolled_bb + rolled_h) / rolled_ip, not a per-game average."""

    df = pd.DataFrame([
        _pitcher_box_row(gamepk='g1', game_date='2023-04-01', bb=1, h=1, ip=6.0),
        _pitcher_box_row(gamepk='g2', game_date='2023-04-02', bb=2, h=1, ip=3.0),
        _pitcher_box_row(gamepk='g3', game_date='2023-04-03'),
    ])

    result = build_pitcher_rolling_stats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    # rolled bb=3, h=2, ip=9 -> whip = 5/9
    assert row3['pitcher_roll_last10g_whip'].iloc[0] == pytest.approx(5 / 9)


def test_build_pitcher_rolling_stats_divide_by_zero_guarded():
    """A pitcher's first career game has zero rolled IP behind it — rate
    stats must come back NaN, not raise."""

    df = pd.DataFrame([_pitcher_box_row(gamepk='g1', game_date='2023-04-01')])

    result = build_pitcher_rolling_stats(df, window=10)

    assert pd.isna(result['pitcher_roll_last10g_whip'].iloc[0])


def test_build_pitcher_rolling_stats_includes_games_n_count():
    """games_n: rolling count of games pitched so far this window — a
    workload sample-size signal, and the shrinkage-weight input for
    k_predictor's rolling-stats-shrunk-to-last-season blend."""

    df = pd.DataFrame([
        _pitcher_box_row(gamepk='g1', game_date='2023-04-01'),
        _pitcher_box_row(gamepk='g2', game_date='2023-04-02'),
        _pitcher_box_row(gamepk='g3', game_date='2023-04-03'),
    ])

    result = build_pitcher_rolling_stats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_games_n'].iloc[0] == 2


def test_build_pitcher_rolling_stats_games_n_zero_for_first_game():
    """A real, meaningful 0 for a pitcher's first game — not NaN, same
    convention as build_pitcher_start_ip_this_season's starts_n."""

    df = pd.DataFrame([_pitcher_box_row(gamepk='g1', game_date='2023-04-01')])

    result = build_pitcher_rolling_stats(df, window=10)

    assert result['pitcher_roll_last10g_games_n'].iloc[0] == 0


def _pitch_row(pitcher_id='1', gamepk='g1', game_date='2023-04-01', game_season=2023,
                play_id=1, pitch_number=1, **overrides):
    row = {
        'pitcher_id': pitcher_id, 'pitcher_team_id': 'T1', 'gamepk': gamepk,
        'game_date': pd.Timestamp(game_date), 'game_season': game_season,
        'play_id': play_id, 'pitch_number': pitch_number,
        'pitcher_role': 'sp',
        'start_speed': 90.0, 'end_speed': 85.0, 'perceived_velo': 92.0,
        'spin_rate': 2200.0, 'movement_magnitude': 10.0, 'pfx_z': 5.0,
        'extension': 6.0, 'speed_retention': 0.94,
        'is_in_play': False, 'is_swinging_strike': False,
        'plate_x': 0.0, 'plate_z_normalized': 0.5, 'zone': 5,
        'is_ball': False, 'is_strike': True, 'is_called_strike': False,
        'is_chase': False, 'is_zone_swing': False,
        'is_first_pitch': False, 'count_balls': 0, 'count_strikes': 0, 'count_outs': 0,
        'play_result': 'Ball', 'inning': 1,
        'hardness': None, 'trajectory': None, 'launch_speed': np.nan, 'launch_angle': np.nan,
    }
    row.update(overrides)
    return row


def test_build_pbp_pitcher_rolling_feats_mean_stat_weighted_by_pitch_count():
    """Game 1 has 2 pitches averaging 91 (sum 182), game 2 has 1 pitch at 94.
    Naive per-game-mean averaging gives 92.5; the correct pitch-weighted
    answer is (182+94)/3 = 92.0 — this is the pbp analogue of the
    rolled-sums-not-averaged-rates rule, for a continuous mean instead of
    a boolean rate."""

    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1, start_speed=90.0),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2, start_speed=92.0),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, start_speed=94.0),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_stuff_start_speed_mean'].iloc[0] == pytest.approx(92.0)


def test_build_pbp_pitcher_rolling_feats_max_stat_is_true_max_of_window():
    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1, start_speed=90.0),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2, start_speed=95.0),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, start_speed=97.0),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_stuff_start_speed_max'].iloc[0] == 97.0


def test_build_pbp_pitcher_rolling_feats_std_stat_matches_true_std_of_window():
    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1, start_speed=90.0),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2, start_speed=92.0),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, start_speed=88.0),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    expected = pd.Series([90.0, 92.0, 88.0]).std()
    assert np.isclose(row3['pitcher_roll_last10g_stuff_start_speed_std'].iloc[0], expected)


def test_build_pbp_pitcher_rolling_feats_boolean_rate_uses_rolled_pitch_counts():
    """command_zone_rate: game 1 is 1-of-2 in zone, game 2 is 1-of-1 in
    zone. Naive per-game-rate averaging gives 0.75; sum-then-divide over
    pitches gives (1+1)/(2+1) = 2/3 — the same pattern as box-score rate
    stats, now for a pbp boolean command rate."""

    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1, zone=5),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2, zone=12),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, zone=3),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_command_zone_rate'].iloc[0] == pytest.approx(2 / 3)


def test_build_pbp_pitcher_rolling_feats_pa_outcome_uses_last_pitch_of_pa_and_rolled_totals():
    """Game 1: PA1 ends in a strikeout on its 2nd (final) pitch — the
    opening ball on pitch 1 must not be counted as its own PA outcome.
    PA2 ends in a single. Game 2: a walk. Rolled into game 3: strikeout_rate
    should be 1 strikeout / 3 total PAs = 1/3, not a naive average of
    game-level rates (0.5 and 0.0 -> 0.25)."""

    df = pd.DataFrame([
        # game 1, PA 1: opening ball, then strikeout on pitch 2 (the true outcome pitch)
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                    play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                    play_result='Strikeout', is_ball=False, is_strike=True,
                    count_balls=1, count_strikes=2),
        # game 1, PA 2: single
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=2, pitch_number=1,
                    play_result='Single'),
        # game 2, PA 1: walk
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1,
                    play_result='Walk'),
        # game 3 — the row under test
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1,
                    play_result='Single'),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_pa_strikeout_rate'].iloc[0] == pytest.approx(1 / 3)


def test_build_pbp_pitcher_rolling_feats_last_inning_rolls_per_game_single_events():
    """Each game's 'last inning pitched' collapses to a single event at
    per-game grain. Rolling this across games (a genuinely new feature
    season-level aggregation couldn't express) should average those
    per-game values, weighted by number of games, not pitches."""

    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1, inning=7),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, inning=9),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1, inning=5),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_last_inning_avg'].iloc[0] == pytest.approx((7 + 9) / 2)


def test_build_pbp_pitcher_rolling_feats_pitch_count_rolls_game_totals():
    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    # game 1 had 2 pitches, game 2 had 1 -> avg 1.5, max 2
    assert row3['pitcher_roll_last10g_pitch_count_avg'].iloc[0] == pytest.approx(1.5)
    assert row3['pitcher_roll_last10g_pitch_count_max'].iloc[0] == 2


def test_build_pbp_pitcher_rolling_feats_contact_quality_rolls_rates_over_balls_in_play():
    """A game with zero balls in play must contribute 0/0 to the rolling
    numerator/denominator (not silently vanish or error), and the rate
    must still be sum-then-divide over the games that did have contact."""

    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                    is_in_play=True, hardness='Hard', trajectory='Line Drive'),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                    is_in_play=True, hardness='Medium', trajectory='Fly Ball'),
        # game 2: no balls in play at all
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, is_in_play=False),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_contact_hard_hit_rate'].iloc[0] == pytest.approx(0.5)


def test_build_pbp_pitcher_rolling_feats_pitcher_role_filter():
    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                    pitcher_role='bullpen', start_speed=80.0),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1,
                    pitcher_role='sp', start_speed=95.0),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1, pitcher_role='sp'),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10, pitcher_role='sp')

    # only the sp game (g2) should feed g3's rolled stat -> exactly 95.0, not blended with bullpen's 80.0
    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_stuff_start_speed_mean'].iloc[0] == pytest.approx(95.0)


def test_build_pitcher_rolling_stats_all_roles_pools_bullpen_by_team():
    """Rolling equivalent of season_stats.build_pitcher_stats_all_roles:
    pitcher_rolling_season_stats/pitcher_rolling_short_stats have never been
    role-aware — bullpen rows must roll forward pooled by team, not stay
    separate by individual reliever."""

    pitcher_boxscore = pd.DataFrame([
        {'personId': '99', 'gamepk': 'g0', 'team_id': 'T1', 'game_season': 2023,
         'game_date': pd.Timestamp('2023-03-30'),
         'h': 5, 'r': 3, 'er': 3, 'bb': 2, 'hr': 1, 'k': 6, 'p': 90, 's': 60, 'ip': 6.0},
        {'personId': '10', 'gamepk': 'g1', 'team_id': 'T1', 'game_season': 2023,
         'game_date': pd.Timestamp('2023-04-01'),
         'h': 1, 'r': 1, 'er': 1, 'bb': 1, 'hr': 0, 'k': 2, 'p': 20, 's': 12, 'ip': 1.0},
        {'personId': '11', 'gamepk': 'g2', 'team_id': 'T1', 'game_season': 2023,
         'game_date': pd.Timestamp('2023-04-02'),
         'h': 2, 'r': 2, 'er': 2, 'bb': 0, 'hr': 1, 'k': 1, 'p': 15, 's': 9, 'ip': 1.0},
        {'personId': '10', 'gamepk': 'g3', 'team_id': 'T1', 'game_season': 2023,
         'game_date': pd.Timestamp('2023-04-03'),
         'h': 0, 'r': 0, 'er': 0, 'bb': 0, 'hr': 0, 'k': 1, 'p': 10, 's': 7, 'ip': 1.0},
    ])
    pbp = pd.DataFrame([
        {'gamepk': 'g0', 'pitcher_id': '99', 'pitcher_team_id': 'T1', 'pitcher_role': 'sp'},
        {'gamepk': 'g1', 'pitcher_id': '10', 'pitcher_team_id': 'T1', 'pitcher_role': 'bullpen'},
        {'gamepk': 'g2', 'pitcher_id': '11', 'pitcher_team_id': 'T1', 'pitcher_role': 'bullpen'},
        {'gamepk': 'g3', 'pitcher_id': '10', 'pitcher_team_id': 'T1', 'pitcher_role': 'bullpen'},
    ])

    result = build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window=10)

    bullpen_rows = result[result['pitcher_role'] == 'bullpen']
    row_g3 = bullpen_rows[bullpen_rows['gamepk'] == 'g3'].iloc[0]
    # g3 is team T1's 3rd bullpen game -> rolls forward BOTH g1 (h=1, pitcher '10')
    # and g2 (h=2, pitcher '11') pooled together: sum = 3, not just pitcher '10's own history
    assert row_g3['pitcher_roll_last10g_h'] == 3
    assert row_g3['pitcher_key_id'] == 'T1'


def test_build_pitcher_rolling_stats_all_roles_collapses_multiple_relievers_per_game_before_rolling():
    """pitcher_boxscore has ONE ROW PER INDIVIDUAL PITCHER PER GAME, not one
    row per team per game — a real bullpen outing routinely uses 2+
    relievers in the same game. build_pitcher_rolling_stats/_rolling_sum
    operate via .transform(), which preserves one output row per INPUT row
    rather than collapsing to one per (team, game) first. Without an
    explicit per-team-per-game collapse before rolling: (a) every downstream
    PA-level join fans out (multiple duplicate team-game rows all matching
    the same join key), and (b) shift(1) corrupts across teammates sharing
    the same game_date instead of skipping the whole game. Must collapse to
    one row per (team, game) — summing all relievers' stats — before rolling,
    the same way the pbp-derived per-game layer already does."""

    pitcher_boxscore = pd.DataFrame([
        {'personId': '99', 'gamepk': 'g0', 'team_id': 'T1', 'game_season': 2023,
         'game_date': pd.Timestamp('2023-03-30'),
         'h': 5, 'r': 3, 'er': 3, 'bb': 2, 'hr': 1, 'k': 6, 'p': 90, 's': 60, 'ip': 6.0},
        # game g1: TWO relievers for team T1 on the SAME day
        {'personId': '10', 'gamepk': 'g1', 'team_id': 'T1', 'game_season': 2023,
         'game_date': pd.Timestamp('2023-04-01'),
         'h': 1, 'r': 1, 'er': 1, 'bb': 1, 'hr': 0, 'k': 2, 'p': 20, 's': 12, 'ip': 1.0},
        {'personId': '11', 'gamepk': 'g1', 'team_id': 'T1', 'game_season': 2023,
         'game_date': pd.Timestamp('2023-04-01'),
         'h': 2, 'r': 2, 'er': 2, 'bb': 0, 'hr': 1, 'k': 1, 'p': 15, 's': 9, 'ip': 1.0},
        # game g2: ONE reliever for team T1, later
        {'personId': '12', 'gamepk': 'g2', 'team_id': 'T1', 'game_season': 2023,
         'game_date': pd.Timestamp('2023-04-02'),
         'h': 0, 'r': 0, 'er': 0, 'bb': 0, 'hr': 0, 'k': 1, 'p': 10, 's': 7, 'ip': 1.0},
    ])
    pbp = pd.DataFrame([
        {'gamepk': 'g0', 'pitcher_id': '99', 'pitcher_team_id': 'T1', 'pitcher_role': 'sp'},
        {'gamepk': 'g1', 'pitcher_id': '10', 'pitcher_team_id': 'T1', 'pitcher_role': 'bullpen'},
        {'gamepk': 'g1', 'pitcher_id': '11', 'pitcher_team_id': 'T1', 'pitcher_role': 'bullpen'},
        {'gamepk': 'g2', 'pitcher_id': '12', 'pitcher_team_id': 'T1', 'pitcher_role': 'bullpen'},
    ])

    result = build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window=10)

    bullpen_rows = result[result['pitcher_role'] == 'bullpen']
    # exactly one row per (team, game) — not one per individual reliever
    assert len(bullpen_rows[bullpen_rows['gamepk'] == 'g1']) == 1
    row_g2 = bullpen_rows[bullpen_rows['gamepk'] == 'g2'].iloc[0]
    # g2 rolls forward g1's TEAM TOTAL (h = 1 + 2 = 3), not a per-reliever fragment
    assert row_g2['pitcher_roll_last10g_h'] == 3


def test_build_pbp_pitcher_rolling_feats_pools_bullpen_by_team_when_entity_col_given():
    """Two different bullpen pitchers on the same team must roll forward
    pooled together under entity_col='pitcher_team_id', not stay separate by
    individual pitcher_id — at prediction time you know the team but not
    which specific reliever will face a given batter."""

    df = pd.DataFrame([
        _pitch_row(pitcher_id='1', gamepk='g1', game_date='2023-04-01', pitcher_role='bullpen', start_speed=80.0),
        _pitch_row(pitcher_id='2', gamepk='g2', game_date='2023-04-02', pitcher_role='bullpen', start_speed=90.0),
        _pitch_row(pitcher_id='1', gamepk='g3', game_date='2023-04-03', pitcher_role='bullpen', start_speed=100.0),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10, pitcher_role='bullpen', entity_col='pitcher_team_id')

    # g3 is the TEAM's 3rd bullpen game -> rolls forward BOTH g1 (80, pitcher '1')
    # and g2 (90, pitcher '2') pooled together: mean = 85.0, not just pitcher '1's own history
    row3 = result[result['gamepk'] == 'g3'].iloc[0]
    assert row3['pitcher_roll_last10g_stuff_start_speed_mean'] == pytest.approx(85.0)


def test_build_pbp_pitcher_rolling_feats_all_roles_pools_bullpen_by_team():
    """The bullpen half must be team-pooled (entity_col='pitcher_team_id'),
    not individual-pitcher-keyed — two different relievers on the same team
    roll forward together under a common pitcher_key_id column."""

    df = pd.DataFrame([
        _pitch_row(pitcher_id='1', gamepk='g1', game_date='2023-04-01', pitcher_role='sp'),
        _pitch_row(pitcher_id='2', gamepk='g2', game_date='2023-04-02', pitcher_role='bullpen', start_speed=80.0),
        _pitch_row(pitcher_id='3', gamepk='g3', game_date='2023-04-03', pitcher_role='bullpen', start_speed=90.0),
        _pitch_row(pitcher_id='2', gamepk='g4', game_date='2023-04-04', pitcher_role='bullpen', start_speed=100.0),
    ])

    result = build_pbp_pitcher_rolling_feats_all_roles(df, window=10)

    assert 'pitcher_key_id' in result.columns
    assert 'pitcher_id' not in result.columns
    assert 'pitcher_team_id' not in result.columns

    sp_row = result[result['pitcher_role'] == 'sp'].iloc[0]
    assert sp_row['pitcher_key_id'] == '1'

    # g4 is team T1's 3rd bullpen game -> rolls forward BOTH g2 (80, pitcher '2')
    # and g3 (90, pitcher '3') pooled together, keyed by team not individual pitcher
    bullpen_g4 = result[(result['pitcher_role'] == 'bullpen') & (result['pitcher_key_id'] == 'T1')]
    assert len(bullpen_g4[bullpen_g4['gamepk'] == 'g4']) == 1
    row_g4 = bullpen_g4[bullpen_g4['gamepk'] == 'g4'].iloc[0]
    assert row_g4['pitcher_roll_last10g_stuff_start_speed_mean'] == pytest.approx(85.0)


def test_build_pbp_pitcher_rolling_feats_all_roles_tags_and_stacks_both_roles():
    """The rolling equivalent of season_stats' build_pbp_pitcher_feats_all_roles
    fix: a swingman's SP-role rolling stat and bullpen-role rolling stat must
    each roll forward from ONLY that role's own prior games — not blended
    together — and both roles must appear as separately tagged rows so a PA
    can be joined to the one matching its own game-context role."""

    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', pitcher_role='bullpen', start_speed=80.0),
        _pitch_row(gamepk='g2', game_date='2023-04-02', pitcher_role='sp', start_speed=95.0),
        _pitch_row(gamepk='g3', game_date='2023-04-03', pitcher_role='sp', start_speed=100.0),
        _pitch_row(gamepk='g4', game_date='2023-04-04', pitcher_role='bullpen', start_speed=85.0),
    ])

    result = build_pbp_pitcher_rolling_feats_all_roles(df, window=10)

    assert set(result['pitcher_role']) == {'sp', 'bullpen'}

    # g3 is this pitcher's 2nd SP game -> rolls forward only g2's (sp) 95.0
    sp_row = result[(result['gamepk'] == 'g3') & (result['pitcher_role'] == 'sp')].iloc[0]
    assert sp_row['pitcher_roll_last10g_stuff_start_speed_mean'] == pytest.approx(95.0)

    # g4 is this pitcher's 2nd bullpen game -> rolls forward only g1's (bullpen) 80.0,
    # not contaminated by the sp games at g2/g3
    bullpen_row = result[(result['gamepk'] == 'g4') & (result['pitcher_role'] == 'bullpen')].iloc[0]
    assert bullpen_row['pitcher_roll_last10g_stuff_start_speed_mean'] == pytest.approx(80.0)


def test_build_pbp_pitcher_rolling_feats_pa_pitch_count_mean_uses_rolled_totals():
    """season_stats.py's PA-outcome category also tracks pitch_count_mean/std
    (pitches per PA) alongside the outcome rates. Game 1: PA1 takes 3
    pitches, PA2 takes 1 (sum 4 over 2 PAs). Game 2: one 5-pitch PA. Rolled
    into game 3, mean pitches/PA should be (4+5)/(2+1)=3.0 — sum-then-divide
    over PAs, not the naive per-game average of 2.0 and 5.0 (=3.5)."""

    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=3, play_result='Single'),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=2, pitch_number=1, play_result='Single'),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=2, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=3, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=4, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=5, play_result='Single'),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1, play_result='Single'),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_pa_pitch_count_mean'].iloc[0] == pytest.approx(3.0)


def test_build_pbp_pitcher_rolling_feats_exposes_pitch_and_pa_count_denominators():
    """n_pitches/pa_total are the sample-size denominators behind every
    command/stuff and PA-outcome rate in this table — they must survive as
    their own output columns (not just get consumed internally to compute a
    rate) so a model can learn to trust a rate less when its denominator is
    small, instead of that signal being silently thrown away. Same dataset
    as test_build_pbp_pitcher_rolling_feats_pa_pitch_count_mean_uses_rolled_totals,
    which already establishes n_pitches=9, pa_total=3 rolled into g3."""

    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=3, play_result='Single'),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=2, pitch_number=1, play_result='Single'),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=2, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=3, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=4, play_result='Ball', is_ball=True, is_strike=False),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=5, play_result='Single'),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1, play_result='Single'),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_n_pitches'].iloc[0] == 9
    assert row3['pitcher_roll_last10g_pa_total'].iloc[0] == 3


def test_build_pbp_pitcher_rolling_feats_exposes_contact_and_games_count_denominators():
    """contact_n/games_n are the sample-size denominators behind the
    contact-quality rates and last-inning averages respectively — same
    dataset as test_build_pbp_pitcher_rolling_feats_contact_quality_rolls_rates_over_balls_in_play,
    which already establishes contact_hard_hit_rate=0.5 (i.e. contact_n=2)
    and games_n=2 (two prior games) rolled into g3."""

    df = pd.DataFrame([
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                    is_in_play=True, hardness='Hard', trajectory='Line Drive'),
        _pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                    is_in_play=True, hardness='Medium', trajectory='Fly Ball'),
        _pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, is_in_play=False),
        _pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_pitcher_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['pitcher_roll_last10g_contact_n'].iloc[0] == 2
    assert row3['pitcher_roll_last10g_games_n'].iloc[0] == 2


def test_build_pbp_pitcher_rolling_feats_column_names_no_collision_with_season_stats():
    df = pd.DataFrame([_pitch_row()])

    result_short = build_pbp_pitcher_rolling_feats(df, window=10)
    result_season = build_pbp_pitcher_rolling_feats(df, window='season')

    assert 'pitcher_roll_last10g_stuff_start_speed_mean' in result_short.columns
    assert 'pitcher_roll_season_stuff_start_speed_mean' in result_season.columns
    assert 'pitcher_season_stuff_start_speed_mean' not in result_short.columns
    assert 'pitcher_last_season_stuff_start_speed_mean' not in result_short.columns


def _batter_pitch_row(batter_id='1', gamepk='g1', game_date='2023-04-01', game_season=2023,
                       play_id=1, pitch_number=1, **overrides):
    row = {
        'batter_id': batter_id, 'gamepk': gamepk,
        'game_date': pd.Timestamp(game_date), 'game_season': game_season,
        'play_id': play_id, 'pitch_number': pitch_number,
        'play_result': 'Ball', 'count_balls': 0, 'count_strikes': 0,
        'is_chase': False, 'is_zone_swing': False, 'is_swinging_strike': False,
        'zone': 5, 'is_swing': False, 'pitch_call': 'Ball',
        'is_in_play': False, 'launch_speed': np.nan, 'launch_angle': np.nan, 'trajectory': None,
    }
    row.update(overrides)
    return row


def test_build_pbp_batter_rolling_feats_pa_outcome_and_pitch_count_from_rolled_totals():
    """Mirrors the pitcher PA-outcome test: game 1 is a 2-pitch PA ending in
    strikeout plus a 1-pitch single (sum 3 pitches / 2 PAs); game 2 is a
    walk. Rolled into game 3: strikeout_rate = 1/3 (not naive avg 0.25),
    pitch_count_mean = 3/3 = 1.0-pitch average is wrong on purpose to check
    sum-then-divide, actual expected mean = (2+1)/2 = 1.5 pitches/PA over
    games 1's PAs only for that sub-check; full totals asserted directly."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           play_result='Ball'),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           play_result='Strikeout', count_strikes=2),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=2, pitch_number=1,
                           play_result='Single'),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1,
                           play_result='Walk'),
        _batter_pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1,
                           play_result='Single'),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    # rolled: strikeout_n=1, total PAs=3 -> 1/3, not naive avg of (0.5, 0.0)=0.25
    assert row3['batter_roll_last10g_pa_strikeout_rate'].iloc[0] == pytest.approx(1 / 3)
    # rolled: pitch sum = 2+1+1 = 4 over 3 PAs = 4/3
    assert row3['batter_roll_last10g_pa_pitch_count_mean'].iloc[0] == pytest.approx(4 / 3)


def test_build_pbp_batter_rolling_feats_chase_rate_weighted_by_pitch_count():
    """Game 1: 2 pitches, 1 chased. Game 2: 1 pitch, 1 chased. Naive avg of
    per-game rates (0.5, 1.0) = 0.75; correct sum-then-divide = (1+1)/(2+1)
    = 2/3."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1, is_chase=True),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2, is_chase=False),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1, is_chase=True),
        _batter_pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['batter_roll_last10g_o_swing_rate'].iloc[0] == pytest.approx(2 / 3)


def test_build_pbp_batter_rolling_feats_contact_rate_denominator_is_swings_not_all_pitches():
    """contact_rate's denominator is swings taken, not all pitches seen —
    a batter who took 10 pitches but only swung at 2 (making contact on
    both) should show contact_rate=1.0, not 2/10."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           is_swing=False),  # taken, not a swing
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           is_swing=True, is_swinging_strike=False),  # swing, contact
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1,
                           is_swing=True, is_swinging_strike=False),  # swing, contact
        _batter_pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['batter_roll_last10g_contact_rate'].iloc[0] == pytest.approx(1.0)


def test_build_pbp_batter_rolling_feats_hard_hit_rate_excludes_nulls_from_denominator():
    """hard_hit_rate must dropna launch_speed before computing its rate —
    a null launch_speed (no Statcast read) shouldn't silently deflate the
    rate by inflating the denominator."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           is_in_play=True, launch_speed=100.0, trajectory='Line Drive'),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           is_in_play=True, launch_speed=np.nan, trajectory='Ground Ball'),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1,
                           is_in_play=True, launch_speed=80.0, trajectory='Ground Ball'),
        _batter_pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    # 1 hard hit (100mph) out of 2 valid launch_speed readings (100, 80) -> 0.5, not 1/3
    assert row3['batter_roll_last10g_contact_hard_hit_rate'].iloc[0] == pytest.approx(0.5)


def test_build_pbp_batter_rolling_feats_foul_rate_vs_contact_foul_rate_use_different_denominators():
    """foul_rate's denominator is all swings taken (including whiffs);
    contact_foul_rate's denominator is only foul-or-in-play events (excludes
    whiffs). One game: 3 swings — a whiff, a foul, and a ball in play.
    foul_rate = 1/3 (of all swings); contact_foul_rate = 1/2 (of contact
    events only, excluding the whiff)."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           is_swing=True, pitch_call='Swinging Strike', is_swinging_strike=True),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           is_swing=True, pitch_call='Foul'),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=3,
                           is_swing=True, pitch_call='In Play, Out(s)', is_in_play=True, trajectory='Fly Ball'),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row2 = result.loc[result['gamepk'] == 'g2']
    assert row2['batter_roll_last10g_foul_rate'].iloc[0] == pytest.approx(1 / 3)
    assert row2['batter_roll_last10g_contact_foul_rate'].iloc[0] == pytest.approx(1 / 2)


def test_build_pbp_batter_rolling_feats_two_strike_foul_rate():
    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           count_strikes=2, is_swing=True, pitch_call='Foul'),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           count_strikes=2, is_swing=True, pitch_call='Swinging Strike', is_swinging_strike=True),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row2 = result.loc[result['gamepk'] == 'g2']
    assert row2['batter_roll_last10g_two_strike_foul_rate'].iloc[0] == pytest.approx(0.5)


def test_build_pbp_batter_rolling_feats_exposes_pitch_and_pa_count_denominators():
    """n_pitches/pa_total are the sample-size denominators behind the
    plate-discipline and PA-outcome rates — same dataset as
    test_build_pbp_batter_rolling_feats_pa_outcome_and_pitch_count_from_rolled_totals,
    which already establishes n_pitches=4, pa_total=3 rolled into g3."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           play_result='Ball'),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           play_result='Strikeout', count_strikes=2),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=2, pitch_number=1,
                           play_result='Single'),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1,
                           play_result='Walk'),
        _batter_pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1,
                           play_result='Single'),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['batter_roll_last10g_n_pitches'].iloc[0] == 4
    assert row3['batter_roll_last10g_pa_total'].iloc[0] == 3


def test_build_pbp_batter_rolling_feats_exposes_swing_count_denominator():
    """swing_n is the denominator behind contact_rate — same dataset as
    test_build_pbp_batter_rolling_feats_contact_rate_denominator_is_swings_not_all_pitches,
    which already establishes contact_rate=1.0 (2 contacts / 2 swings)
    rolled into g3."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           is_swing=False),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           is_swing=True, is_swinging_strike=False),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1,
                           is_swing=True, is_swinging_strike=False),
        _batter_pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['batter_roll_last10g_swing_n'].iloc[0] == 2


def test_build_pbp_batter_rolling_feats_exposes_contact_trajectory_count_denominator():
    """contact_trajectory_n is the denominator behind the contact-quality
    rates (gb/fb/ld rate) — same dataset as
    test_build_pbp_batter_rolling_feats_hard_hit_rate_excludes_nulls_from_denominator,
    which has 2 in-play pitches in g1 and 1 in g2, all with a trajectory
    value, rolled into g3."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           is_in_play=True, launch_speed=100.0, trajectory='Line Drive'),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           is_in_play=True, launch_speed=np.nan, trajectory='Ground Ball'),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1,
                           is_in_play=True, launch_speed=80.0, trajectory='Ground Ball'),
        _batter_pitch_row(gamepk='g3', game_date='2023-04-03', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row3 = result.loc[result['gamepk'] == 'g3']
    assert row3['batter_roll_last10g_contact_trajectory_n'].iloc[0] == 3


def test_build_pbp_batter_rolling_feats_exposes_foul_count_denominators():
    """foul_swing_n (denominator of foul_rate: all swings) and
    foul_or_inplay_n (denominator of contact_foul_rate: foul-or-in-play
    events only, excluding whiffs) are two genuinely different sample
    sizes — same dataset as
    test_build_pbp_batter_rolling_feats_foul_rate_vs_contact_foul_rate_use_different_denominators,
    which already establishes foul_rate=1/3, contact_foul_rate=1/2 rolled
    into g2 (a whiff + a foul + a ball in play = 3 swings, 2 contact events)."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           is_swing=True, pitch_call='Swinging Strike', is_swinging_strike=True),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           is_swing=True, pitch_call='Foul'),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=3,
                           is_swing=True, pitch_call='In Play, Out(s)', is_in_play=True, trajectory='Fly Ball'),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row2 = result.loc[result['gamepk'] == 'g2']
    assert row2['batter_roll_last10g_foul_swing_n'].iloc[0] == 3
    assert row2['batter_roll_last10g_foul_or_inplay_n'].iloc[0] == 2


def test_build_pbp_batter_rolling_feats_exposes_two_strike_swing_count_denominator():
    """two_strike_swing_n is the denominator behind two_strike_foul_rate —
    same dataset as test_build_pbp_batter_rolling_feats_two_strike_foul_rate,
    which already establishes two_strike_foul_rate=0.5 (1 foul / 2 two-strike
    swings) rolled into g2."""

    df = pd.DataFrame([
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=1,
                           count_strikes=2, is_swing=True, pitch_call='Foul'),
        _batter_pitch_row(gamepk='g1', game_date='2023-04-01', play_id=1, pitch_number=2,
                           count_strikes=2, is_swing=True, pitch_call='Swinging Strike', is_swinging_strike=True),
        _batter_pitch_row(gamepk='g2', game_date='2023-04-02', play_id=1, pitch_number=1),
    ])

    result = build_pbp_batter_rolling_feats(df, window=10)

    row2 = result.loc[result['gamepk'] == 'g2']
    assert row2['batter_roll_last10g_two_strike_swing_n'].iloc[0] == 2


def test_build_pbp_batter_rolling_feats_column_names_no_collision_with_season_stats():
    df = pd.DataFrame([_batter_pitch_row()])

    result_short = build_pbp_batter_rolling_feats(df, window=10)
    result_season = build_pbp_batter_rolling_feats(df, window='season')

    assert 'batter_roll_last10g_pa_strikeout_rate' in result_short.columns
    assert 'batter_roll_season_pa_strikeout_rate' in result_season.columns
    assert 'batter_season_pa_strikeout_rate' not in result_short.columns
    assert 'batter_last_season_pa_strikeout_rate' not in result_short.columns


# --------------------- TEAM-LEVEL BATTER STRIKEOUT ROLLING FEATS --------------------- #
# k_predictor v2: opposing-lineup rolling K rate. Mirrors
# build_pitcher_rolling_stats_all_roles' bullpen-pooling pattern (pool per-entity
# per-game rows into one team-game row before rolling), but pools STARTING-LINEUP
# batters only (those with a batting_order that game) rather than pooling by role.

def _lineup_slot_row(personId='1', gamepk='g1', batting_order=1, **overrides):
    row = {'personId': personId, 'gamepk': gamepk, 'batting_order': batting_order}
    row.update(overrides)
    return row


def test_build_team_batter_strikeout_rolling_feats_pools_starting_lineup_by_team():
    """Team-level rolling K rate must pool ONLY starting-lineup batters
    (those carrying a batting_order that game) into the team-game total.
    Batter '2' starts in g1 (batting_order=2) but has NO batting_order row
    in g2 (simulates a bench/pinch-hit appearance) despite appearing in
    pbp — g2's PA must be excluded from the team pool even though the
    batter is present in pbp that game."""

    pbp = pd.DataFrame([
        _batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g1',
                           game_date='2023-04-01', play_id=1, play_result='Strikeout', count_strikes=2),
        _batter_pitch_row(batter_id='2', batter_team_id='T1', gamepk='g1',
                           game_date='2023-04-01', play_id=2, play_result='Single'),
        # g2: batter '2' pinch-hits (present in pbp) but has no lineup slot that game
        _batter_pitch_row(batter_id='2', batter_team_id='T1', gamepk='g2',
                           game_date='2023-04-02', play_id=1, play_result='Strikeout', count_strikes=2),
        _batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g3',
                           game_date='2023-04-03', play_id=1, play_result='Single'),
    ])
    batter_boxscore = pd.DataFrame([
        _lineup_slot_row(personId='1', gamepk='g1', batting_order=1),
        _lineup_slot_row(personId='2', gamepk='g1', batting_order=2),
        # no row at all for batter '2' in g2 -> no batting_order that game
        _lineup_slot_row(personId='1', gamepk='g3', batting_order=1),
    ])

    result = build_team_batter_strikeout_rolling_feats(pbp, batter_boxscore, window=10)

    row_g3 = result.loc[result['gamepk'] == 'g3'].iloc[0]
    # g1's starting lineup pooled: pa_total=2 (batter 1's K + batter 2's single), strikeout_n=1.
    # g2's pinch-hit K (batter 2, no batting_order) must be EXCLUDED from the pool.
    assert row_g3['team_roll_last10g_pa_total'] == 2
    assert row_g3['team_roll_last10g_pa_strikeout_n'] == 1


def test_build_team_batter_strikeout_rolling_feats_excludes_current_game():
    """g3's rolled totals must reflect only g1+g2, not g3's own PAs — same
    point-in-time guarantee _rolling_sum already provides elsewhere in this
    file; this verifies the team-pooling step doesn't break it."""

    pbp = pd.DataFrame([
        _batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g1',
                           game_date='2023-04-01', play_id=1, play_result='Strikeout', count_strikes=2),
        _batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g2',
                           game_date='2023-04-02', play_id=1, play_result='Single'),
        _batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g3',
                           game_date='2023-04-03', play_id=1, play_result='Strikeout', count_strikes=2),
    ])
    batter_boxscore = pd.DataFrame([
        _lineup_slot_row(personId='1', gamepk='g1', batting_order=1),
        _lineup_slot_row(personId='1', gamepk='g2', batting_order=1),
        _lineup_slot_row(personId='1', gamepk='g3', batting_order=1),
    ])

    result = build_team_batter_strikeout_rolling_feats(pbp, batter_boxscore, window=10)

    row_g3 = result.loc[result['gamepk'] == 'g3'].iloc[0]
    # g1 (1 PA, 1 K) + g2 (1 PA, 0 K) -> pa_total=2, strikeout_n=1; g3's own K excluded
    assert row_g3['team_roll_last10g_pa_total'] == 2
    assert row_g3['team_roll_last10g_pa_strikeout_n'] == 1


def test_build_team_batter_strikeout_rolling_feats_rate_divides_rolled_sums_not_per_game_avg():
    """Rate must come from rolled numerator/denominator sums, not an
    average of each game's own rate — the 'roll counts, not rates' rule
    this whole file follows elsewhere. Also: a team's first game has zero
    rolled PA and must produce NaN, not a ZeroDivisionError/inf."""

    pbp_rows = [
        # g1: 3 PA, 1 K -> rate 1/3
        _batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g1', game_date='2023-04-01',
                           play_id=1, play_result='Strikeout', count_strikes=2),
        _batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g1', game_date='2023-04-01',
                           play_id=2, play_result='Single'),
        _batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g1', game_date='2023-04-01',
                           play_id=3, play_result='Single'),
    ]
    # g2: 6 PA, 1 K -> rate 1/6
    for i in range(1, 7):
        pbp_rows.append(_batter_pitch_row(
            batter_id='1', batter_team_id='T1', gamepk='g2', game_date='2023-04-02',
            play_id=i, play_result='Strikeout' if i == 1 else 'Single',
            count_strikes=2 if i == 1 else 0,
        ))
    pbp_rows.append(_batter_pitch_row(batter_id='1', batter_team_id='T1', gamepk='g3',
                                       game_date='2023-04-03', play_id=1, play_result='Single'))
    pbp = pd.DataFrame(pbp_rows)

    batter_boxscore = pd.DataFrame([
        _lineup_slot_row(personId='1', gamepk='g1', batting_order=1),
        _lineup_slot_row(personId='1', gamepk='g2', batting_order=1),
        _lineup_slot_row(personId='1', gamepk='g3', batting_order=1),
    ])

    result = build_team_batter_strikeout_rolling_feats(pbp, batter_boxscore, window=10)

    row_g1 = result.loc[result['gamepk'] == 'g1'].iloc[0]
    assert pd.isna(row_g1['team_roll_last10g_pa_strikeout_rate'])

    row_g3 = result.loc[result['gamepk'] == 'g3'].iloc[0]
    # rolled: strikeout_n=1+1=2, total PA=3+6=9 -> 2/9, NOT naive avg of (1/3, 1/6)=0.25
    assert row_g3['team_roll_last10g_pa_strikeout_rate'] == pytest.approx(2 / 9)
