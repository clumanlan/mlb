import pandas as pd

from models.n_pa_predictor.processing.features.batter_playing_time import build_batter_pa_rolling_stats


def _make_game_row(**overrides):
    row = {
        'batter_id': '1',
        'gamepk': '1',
        'game_date': pd.Timestamp('2023-04-01'),
        'game_season': 2023,
        'n_pa': 4,
    }
    row.update(overrides)
    return row


def test_first_game_of_season_has_no_prior_games_n():
    """Point-in-time safety: a batter's very first game has nothing prior to
    average — games_n must be 0 (not leaking this game's own n_pa) and
    avg_n_pa_per_game must be NaN, not this game's own realized value."""

    df = pd.DataFrame([_make_game_row()])

    result = build_batter_pa_rolling_stats(df, window='season')

    row = result.iloc[0]
    assert row['batter_n_pa_roll_season_games_n'] == 0
    assert pd.isna(row['batter_n_pa_roll_season_avg_n_pa_per_game'])


def test_second_game_averages_only_the_prior_game():
    """The 2nd game's rolling average reflects only game 1's n_pa (4),
    excluding game 2's own (realized only after the fact)."""

    df = pd.DataFrame([
        _make_game_row(gamepk='1', game_date=pd.Timestamp('2023-04-01'), n_pa=4),
        _make_game_row(gamepk='2', game_date=pd.Timestamp('2023-04-02'), n_pa=2),
    ])

    result = build_batter_pa_rolling_stats(df, window='season')

    row2 = result[result['gamepk'] == '2'].iloc[0]
    assert row2['batter_n_pa_roll_season_games_n'] == 1
    assert row2['batter_n_pa_roll_season_avg_n_pa_per_game'] == 4.0


def test_season_window_resets_at_season_boundary():
    """window='season' is an expanding sum within game_season — a new
    season starts with no prior games, same convention as
    rolling_stats._rolling_sum."""

    df = pd.DataFrame([
        _make_game_row(gamepk='1', game_date=pd.Timestamp('2023-09-01'), game_season=2023, n_pa=5),
        _make_game_row(gamepk='2', game_date=pd.Timestamp('2024-04-01'), game_season=2024, n_pa=3),
    ])

    result = build_batter_pa_rolling_stats(df, window='season')

    row2 = result[result['gamepk'] == '2'].iloc[0]
    assert row2['batter_n_pa_roll_season_games_n'] == 0
    assert pd.isna(row2['batter_n_pa_roll_season_avg_n_pa_per_game'])


def test_short_window_carries_across_season_boundary():
    """window=<int> is a trailing-N-game sum that carries across season
    boundaries, same convention as rolling_stats._rolling_sum's short
    windows (recent form has no reason to reset on Opening Day)."""

    df = pd.DataFrame([
        _make_game_row(gamepk='1', game_date=pd.Timestamp('2023-09-01'), game_season=2023, n_pa=5),
        _make_game_row(gamepk='2', game_date=pd.Timestamp('2024-04-01'), game_season=2024, n_pa=3),
    ])

    result = build_batter_pa_rolling_stats(df, window=5)

    row2 = result[result['gamepk'] == '2'].iloc[0]
    assert row2['batter_n_pa_roll_last5g_games_n'] == 1
    assert row2['batter_n_pa_roll_last5g_avg_n_pa_per_game'] == 5.0


def test_per_batter_not_pooled_across_batters():
    """Two different batters' rolling averages don't leak into each other."""

    df = pd.DataFrame([
        _make_game_row(batter_id='1', gamepk='1', game_date=pd.Timestamp('2023-04-01'), n_pa=5),
        _make_game_row(batter_id='2', gamepk='2', game_date=pd.Timestamp('2023-04-01'), n_pa=1),
        _make_game_row(batter_id='1', gamepk='3', game_date=pd.Timestamp('2023-04-02'), n_pa=4),
    ])

    result = build_batter_pa_rolling_stats(df, window='season')

    row3 = result[result['gamepk'] == '3'].iloc[0]
    assert row3['batter_n_pa_roll_season_avg_n_pa_per_game'] == 5.0
