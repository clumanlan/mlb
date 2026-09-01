import numpy as np
import pandas as pd
import pytest

from models.hit_predictor.processing.features.game_context import (
    build_datetime_features,
    build_doubleheader_flag,
    _team_game_long,
    build_team_win_loss_record,
    build_team_rest_days,
    build_probable_starters,
    build_pitcher_start_ip_this_season,
    build_expected_start_innings,
    build_pitcher_start_pa_this_season,
    build_expected_batters_faced,
    build_batter_slot_expansion,
    build_pitcher_role_by_inning,
    build_batters_faced_residual_bins,
    build_batters_faced_distribution,
    build_pitcher_rest_days,
    build_pitcher_workload_density,
    build_pitcher_anomaly_count_this_season,
    build_team_scoring_volatility,
)


def _schedule_row(gamepk='g1', game_datetime='2024-06-01 20:05:00'):
    ts = pd.Timestamp(game_datetime)
    ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts
    return {'gamepk': gamepk, 'game_datetime': ts}


def test_build_datetime_features_extracts_calendar_parts():
    """Sat June 1 2024, 20:05 UTC — the standard fastai add_datepart calendar
    breakdown (no venue-timezone conversion; hour/minute are UTC as stored)."""

    df = pd.DataFrame([_schedule_row(game_datetime='2024-06-01 20:05:00')])

    result = build_datetime_features(df)

    row = result.iloc[0]
    assert row['game_dt_year'] == 2024
    assert row['game_dt_month'] == 6
    assert row['game_dt_day'] == 1
    assert row['game_dt_dayofweek'] == 5  # Saturday, Monday=0
    assert row['game_dt_hour'] == 20
    assert row['game_dt_minute'] == 5


def test_build_datetime_features_month_end_flag():
    df = pd.DataFrame([_schedule_row(gamepk='g1', game_datetime='2024-03-31 18:00:00')])

    result = build_datetime_features(df)

    assert bool(result.iloc[0]['game_dt_is_month_end']) is True
    assert bool(result.iloc[0]['game_dt_is_month_start']) is False


def test_build_datetime_features_quarter_start_flag():
    df = pd.DataFrame([_schedule_row(gamepk='g1', game_datetime='2024-04-01 18:00:00')])

    result = build_datetime_features(df)

    assert bool(result.iloc[0]['game_dt_is_quarter_start']) is True
    assert bool(result.iloc[0]['game_dt_is_month_start']) is True


def test_build_datetime_features_dayofyear_matches_pandas_accessor():
    """Cross-check against pandas' own .dt.dayofyear directly, rather than
    hand-computing a day count — same pattern as this repo's other 'derived
    from a known-correct library accessor' tests (e.g. rolling_stats.py's
    pooled-std test against pd.Series.std())."""

    ts = pd.Timestamp('2024-07-15 15:30:00', tz='UTC')
    df = pd.DataFrame([_schedule_row(gamepk='g1', game_datetime=ts)])

    result = build_datetime_features(df)

    assert result.iloc[0]['game_dt_dayofyear'] == ts.dayofyear
    assert result.iloc[0]['game_dt_week'] == ts.isocalendar().week


def test_build_datetime_features_preserves_key_columns_and_row_count():
    df = pd.DataFrame([
        _schedule_row(gamepk='g1', game_datetime='2024-06-01 20:05:00'),
        _schedule_row(gamepk='g2', game_datetime='2024-06-02 17:10:00'),
    ])

    result = build_datetime_features(df)

    assert len(result) == 2
    assert set(result['gamepk']) == {'g1', 'g2'}


def test_build_datetime_features_does_not_include_raw_epoch():
    """No 'Elapsed'/raw-timestamp column — a strictly-increasing epoch value
    is useless (even harmful) to a model whose val/test seasons are always
    chronologically after every fit season, same reasoning game_season is
    excluded from NUM_FEATS in every experiment train.py."""

    df = pd.DataFrame([_schedule_row()])

    result = build_datetime_features(df)

    elapsed_like = [c for c in result.columns if 'elapsed' in c.lower() or 'epoch' in c.lower()]
    assert elapsed_like == []


def test_build_doubleheader_flag_true_for_traditional_and_split_codes():
    """MLB API's doubleheader code: 'Y' = traditional doubleheader, 'S' =
    split doubleheader (different times/tickets, same two teams same day) —
    both count as True, only 'N' is False."""

    df = pd.DataFrame({'gamepk': ['g1', 'g2', 'g3'], 'doubleheader': ['Y', 'S', 'N']})

    result = build_doubleheader_flag(df)

    assert result['is_doubleheader'].tolist() == [True, True, False]


def test_build_doubleheader_flag_both_games_flagged_not_just_game_two():
    """Both halves of a Y-coded doubleheader carry the SAME code ('Y') on
    both rows — game_num alone (1 vs 2) isn't the input here, the shared
    per-game doubleheader code already applies to both."""

    df = pd.DataFrame({
        'gamepk': ['g1', 'g2'], 'game_num': [1, 2], 'doubleheader': ['Y', 'Y'],
    })

    result = build_doubleheader_flag(df)

    assert result['is_doubleheader'].tolist() == [True, True]


def test_build_doubleheader_flag_preserves_original_columns():
    df = pd.DataFrame({'gamepk': ['g1'], 'doubleheader': ['N']})

    result = build_doubleheader_flag(df)

    assert 'doubleheader' in result.columns
    assert 'gamepk' in result.columns


def _schedule_game(gamepk='g1', game_date='2023-04-01', game_datetime=None,
                    home_id='H', away_id='A', home_score=5, away_score=2):
    dt = pd.Timestamp(game_datetime or f'{game_date} 19:00', tz='UTC')
    return {
        'gamepk': gamepk, 'game_date': pd.Timestamp(game_date), 'game_datetime': dt,
        'home_id': home_id, 'away_id': away_id,
        'home_score': home_score, 'away_score': away_score,
    }


def test_team_game_long_splits_each_game_into_a_home_row_and_an_away_row():
    schedule = pd.DataFrame([_schedule_game(home_id='H', away_id='A', home_score=5, away_score=2)])

    result = _team_game_long(schedule)

    assert len(result) == 2
    home_row = result[result['team_id'] == 'H'].iloc[0]
    away_row = result[result['team_id'] == 'A'].iloc[0]
    assert home_row['opp_id'] == 'A'
    assert home_row['team_score'] == 5 and home_row['opp_score'] == 2
    assert away_row['opp_id'] == 'H'
    assert away_row['team_score'] == 2 and away_row['opp_score'] == 5


def test_team_game_long_win_n_reflects_who_actually_won():
    schedule = pd.DataFrame([_schedule_game(home_id='H', away_id='A', home_score=5, away_score=2)])

    result = _team_game_long(schedule)

    assert result.loc[result['team_id'] == 'H', 'win_n'].iloc[0] == 1
    assert result.loc[result['team_id'] == 'A', 'win_n'].iloc[0] == 0


def test_build_team_win_loss_record_season_window_rolls_forward_wins_and_run_diff():
    """Team H: game1 win 5-2 (run diff +3), game2 loss 1-4 (run diff -3).
    Rolled into game3: win_n=1, games_n=2, win_pct=0.5, run_diff=0 — not
    the current game's own result."""

    schedule = pd.DataFrame([
        _schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X', home_score=5, away_score=2),
        _schedule_game(gamepk='g2', game_date='2023-04-02', home_id='H', away_id='X', home_score=1, away_score=4),
        _schedule_game(gamepk='g3', game_date='2023-04-03', home_id='H', away_id='X', home_score=9, away_score=0),
    ])

    result = build_team_win_loss_record(schedule, window='season')

    row3 = result[(result['team_id'] == 'H') & (result['gamepk'] == 'g3')].iloc[0]
    assert row3['team_roll_season_win_n'] == 1
    assert row3['team_roll_season_games_n'] == 2
    assert row3['team_roll_season_win_pct'] == pytest.approx(0.5)
    assert row3['team_roll_season_run_diff'] == 0


def test_build_team_win_loss_record_short_window_is_trailing_n_games():
    schedule = pd.DataFrame([
        _schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X', home_score=5, away_score=0),
        _schedule_game(gamepk='g2', game_date='2023-04-02', home_id='H', away_id='X', home_score=5, away_score=0),
        _schedule_game(gamepk='g3', game_date='2023-04-03', home_id='H', away_id='X', home_score=0, away_score=5),
    ])

    result = build_team_win_loss_record(schedule, window=2)

    row3 = result[(result['team_id'] == 'H') & (result['gamepk'] == 'g3')].iloc[0]
    # trailing 2 games (g1, g2) -> 2 wins out of 2
    assert row3['team_roll_last2g_win_n'] == 2
    assert row3['team_roll_last2g_games_n'] == 2


def test_build_team_win_loss_record_first_game_of_season_is_nan():
    schedule = pd.DataFrame([
        _schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X', home_score=5, away_score=0),
    ])

    result = build_team_win_loss_record(schedule, window='season')

    row1 = result[result['team_id'] == 'H'].iloc[0]
    assert pd.isna(row1['team_roll_season_win_pct'])


def test_build_team_scoring_volatility_season_window_computes_mean_std_max():
    """Team H scores 5, 1, 9 in g1/g2/g3. Rolled into g3 (uses g1, g2 only):
    mean=(5+1)/2=3.0, sample std of [5,1]=sqrt(((5-3)^2+(1-3)^2)/(2-1))=sqrt(8),
    max=5 — not the current game's own 9."""
    schedule = pd.DataFrame([
        _schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X', home_score=5, away_score=0),
        _schedule_game(gamepk='g2', game_date='2023-04-02', home_id='H', away_id='X', home_score=1, away_score=0),
        _schedule_game(gamepk='g3', game_date='2023-04-03', home_id='H', away_id='X', home_score=9, away_score=0),
    ])

    result = build_team_scoring_volatility(schedule, window='season')

    row3 = result[(result['team_id'] == 'H') & (result['gamepk'] == 'g3')].iloc[0]
    assert row3['team_roll_season_runs_scored_mean'] == pytest.approx(3.0)
    assert row3['team_roll_season_runs_scored_std'] == pytest.approx(np.sqrt(8))
    assert row3['team_roll_season_runs_scored_max'] == 5


def test_build_team_scoring_volatility_short_window_is_trailing_n_games():
    """Team H scores 10, 2, 4 in g1/g2/g3. Rolled into g4 at window=2 uses
    only g2/g3 (mean=3, max=4) — a different answer than season, which
    would use g1/g2/g3 (mean≈5.33, max=10)."""
    schedule = pd.DataFrame([
        _schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X', home_score=10, away_score=0),
        _schedule_game(gamepk='g2', game_date='2023-04-02', home_id='H', away_id='X', home_score=2, away_score=0),
        _schedule_game(gamepk='g3', game_date='2023-04-03', home_id='H', away_id='X', home_score=4, away_score=0),
        _schedule_game(gamepk='g4', game_date='2023-04-04', home_id='H', away_id='X', home_score=1, away_score=0),
    ])

    result = build_team_scoring_volatility(schedule, window=2)

    row4 = result[(result['team_id'] == 'H') & (result['gamepk'] == 'g4')].iloc[0]
    assert row4['team_roll_last2g_runs_scored_mean'] == pytest.approx(3.0)
    assert row4['team_roll_last2g_runs_scored_max'] == 4


def test_build_team_scoring_volatility_first_game_of_season_is_nan():
    schedule = pd.DataFrame([
        _schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X', home_score=5, away_score=0),
    ])

    result = build_team_scoring_volatility(schedule, window='season')

    row1 = result[result['team_id'] == 'H'].iloc[0]
    assert pd.isna(row1['team_roll_season_runs_scored_mean'])
    assert pd.isna(row1['team_roll_season_runs_scored_std'])
    assert pd.isna(row1['team_roll_season_runs_scored_max'])


def test_build_team_scoring_volatility_std_is_nan_with_only_one_prior_game():
    """Sample std is undefined with fewer than 2 prior games — mean and max
    are still defined off that single prior game."""
    schedule = pd.DataFrame([
        _schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X', home_score=5, away_score=0),
        _schedule_game(gamepk='g2', game_date='2023-04-02', home_id='H', away_id='X', home_score=9, away_score=0),
    ])

    result = build_team_scoring_volatility(schedule, window='season')

    row2 = result[(result['team_id'] == 'H') & (result['gamepk'] == 'g2')].iloc[0]
    assert row2['team_roll_season_runs_scored_mean'] == pytest.approx(5.0)
    assert row2['team_roll_season_runs_scored_max'] == 5
    assert pd.isna(row2['team_roll_season_runs_scored_std'])


def test_build_team_rest_days_computes_calendar_days_since_last_game():
    schedule = pd.DataFrame([
        _schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X'),
        _schedule_game(gamepk='g2', game_date='2023-04-05', home_id='H', away_id='X'),
    ])

    result = build_team_rest_days(schedule)

    row2 = result[(result['team_id'] == 'H') & (result['gamepk'] == 'g2')].iloc[0]
    assert row2['team_days_since_last_game'] == 4


def test_build_team_rest_days_first_game_is_nan():
    schedule = pd.DataFrame([_schedule_game(gamepk='g1', game_date='2023-04-01', home_id='H', away_id='X')])

    result = build_team_rest_days(schedule)

    assert pd.isna(result.iloc[0]['team_days_since_last_game'])


def test_build_team_rest_days_orders_doubleheader_games_by_game_datetime_not_date():
    """Both games share game_date — game 1 (earlier game_datetime) must not
    see 0 rest from game 2 (which hasn't been played yet relative to game 1),
    and game 2 correctly sees 0 days rest since game 1 was earlier that day."""

    schedule = pd.DataFrame([
        _schedule_game(gamepk='g0', game_date='2023-04-10', game_datetime='2023-04-10 19:00',
                        home_id='H', away_id='X'),
        # doubleheader on 4/13, inserted out of chronological order deliberately
        _schedule_game(gamepk='g2', game_date='2023-04-13', game_datetime='2023-04-13 22:00',
                        home_id='H', away_id='X'),
        _schedule_game(gamepk='g1', game_date='2023-04-13', game_datetime='2023-04-13 16:00',
                        home_id='H', away_id='X'),
    ])

    result = build_team_rest_days(schedule)

    game1 = result[(result['team_id'] == 'H') & (result['gamepk'] == 'g1')].iloc[0]
    game2 = result[(result['team_id'] == 'H') & (result['gamepk'] == 'g2')].iloc[0]
    assert game1['team_days_since_last_game'] == 3  # since g0 on 4/10
    assert game2['team_days_since_last_game'] == 0  # same day as game1


def test_build_probable_starters_splits_home_and_away_onto_team_rows():
    """schedule has home_id/away_id (team identity); game_info has the
    probable_pitcher_home_id/away_id (schedule only has pitcher NAMES, not
    IDs — see GAME_INFO_COLUMNS vs SCHEDULE_COLUMNS). Joined on gamepk, then
    reshaped to the same team-centric grain as _team_game_long."""

    schedule = pd.DataFrame({'gamepk': ['g1'], 'home_id': ['H'], 'away_id': ['A']})
    game_info = pd.DataFrame({
        'gamepk': ['g1'], 'probable_pitcher_home_id': ['P_H'], 'probable_pitcher_away_id': ['P_A'],
    })

    result = build_probable_starters(schedule, game_info)

    assert result.loc[result['team_id'] == 'H', 'probable_starter_id'].iloc[0] == 'P_H'
    assert result.loc[result['team_id'] == 'A', 'probable_starter_id'].iloc[0] == 'P_A'


def test_build_probable_starters_missing_game_info_yields_null_starter():
    """A gamepk with no matching game_info row (e.g. not yet ingested) must
    not silently drop the game — team_id row still exists, starter is null."""

    schedule = pd.DataFrame({'gamepk': ['g1'], 'home_id': ['H'], 'away_id': ['A']})
    game_info = pd.DataFrame({
        'gamepk': [], 'probable_pitcher_home_id': [], 'probable_pitcher_away_id': [],
    })

    result = build_probable_starters(schedule, game_info)

    assert len(result) == 2
    assert result['probable_starter_id'].isna().all()


def _start_ip_this_season_pitcher_boxscore():
    return pd.DataFrame([
        {'personId': '10', 'gamepk': 'g1', 'game_date': pd.Timestamp('2024-04-01'),
         'game_datetime': pd.Timestamp('2024-04-01 19:00', tz='UTC'), 'game_season': 2024, 'ip': 6.0},
        {'personId': '10', 'gamepk': 'g2', 'game_date': pd.Timestamp('2024-04-06'),
         'game_datetime': pd.Timestamp('2024-04-06 19:00', tz='UTC'), 'game_season': 2024, 'ip': 4.0},
        {'personId': '10', 'gamepk': 'g3', 'game_date': pd.Timestamp('2024-04-11'),
         'game_datetime': pd.Timestamp('2024-04-11 19:00', tz='UTC'), 'game_season': 2024, 'ip': 5.0},
    ])


def _start_ip_this_season_pbp():
    return pd.DataFrame([
        {'pitcher_id': '10', 'pitcher_team_id': 'T1', 'gamepk': g, 'pitcher_role': 'sp'}
        for g in ['g1', 'g2', 'g3']
    ])


def test_build_pitcher_start_ip_this_season_rolls_forward_avg_and_start_count():
    """Two prior starts this season (6.0, 4.0 IP) rolled into start 3:
    avg = 5.0, starts_n = 2 — this season's own emerging sample, distinct
    from season_stats.py's fixed last-season baseline."""

    result = build_pitcher_start_ip_this_season(
        _start_ip_this_season_pitcher_boxscore(), _start_ip_this_season_pbp()
    )
    row3 = result[result['gamepk'] == 'g3'].iloc[0]

    assert row3['pitcher_this_season_start_ip_starts_n'] == 2
    assert row3['pitcher_this_season_start_ip_avg_ip_per_start'] == pytest.approx(5.0)


def test_build_pitcher_start_ip_this_season_first_start_is_nan():
    """A pitcher's first start of the season has nothing prior this season
    to average — must be NaN, not 0 (0 would look like a real, terrible
    average rather than 'no in-season sample yet')."""

    result = build_pitcher_start_ip_this_season(
        _start_ip_this_season_pitcher_boxscore(), _start_ip_this_season_pbp()
    )
    row1 = result[result['gamepk'] == 'g1'].iloc[0]

    assert row1['pitcher_this_season_start_ip_starts_n'] == 0
    assert pd.isna(row1['pitcher_this_season_start_ip_avg_ip_per_start'])


def _start_ip_four_starts_pitcher_boxscore():
    return pd.DataFrame([
        {'personId': '10', 'gamepk': 'g1', 'game_date': pd.Timestamp('2024-04-01'),
         'game_datetime': pd.Timestamp('2024-04-01 19:00', tz='UTC'), 'game_season': 2024, 'ip': 6.0},
        {'personId': '10', 'gamepk': 'g2', 'game_date': pd.Timestamp('2024-04-06'),
         'game_datetime': pd.Timestamp('2024-04-06 19:00', tz='UTC'), 'game_season': 2024, 'ip': 4.0},
        {'personId': '10', 'gamepk': 'g3', 'game_date': pd.Timestamp('2024-04-11'),
         'game_datetime': pd.Timestamp('2024-04-11 19:00', tz='UTC'), 'game_season': 2024, 'ip': 5.0},
        {'personId': '10', 'gamepk': 'g4', 'game_date': pd.Timestamp('2024-04-16'),
         'game_datetime': pd.Timestamp('2024-04-16 19:00', tz='UTC'), 'game_season': 2024, 'ip': 7.0},
    ])


def _start_ip_four_starts_pbp():
    return pd.DataFrame([
        {'pitcher_id': '10', 'pitcher_team_id': 'T1', 'gamepk': g, 'pitcher_role': 'sp'}
        for g in ['g1', 'g2', 'g3', 'g4']
    ])


def test_build_pitcher_start_ip_this_season_trailing_window_uses_only_last_n_starts():
    """window=2 (trailing, not expanding): start 4's avg should reflect only
    starts 2-3 (4.0, 5.0 -> 4.5), not all of starts 1-3 (6.0, 4.0, 5.0 -> 5.0,
    what the default expanding-season window would give) — the whole point of
    a trailing window being more responsive to a recent workload trend than
    a season-to-date average."""

    result = build_pitcher_start_ip_this_season(
        _start_ip_four_starts_pitcher_boxscore(), _start_ip_four_starts_pbp(), window=2,
    )
    row4 = result[result['gamepk'] == 'g4'].iloc[0]

    assert row4['pitcher_last2_start_ip_starts_n'] == 2
    assert row4['pitcher_last2_start_ip_avg_ip_per_start'] == pytest.approx(4.5)


def test_build_pitcher_start_ip_this_season_default_window_is_season_and_unchanged():
    """window defaults to 'season' — existing callers (build_expected_start_innings,
    every model's baseline/run.py) must see identical column names/values to
    before this parameter was added."""

    result = build_pitcher_start_ip_this_season(
        _start_ip_this_season_pitcher_boxscore(), _start_ip_this_season_pbp()
    )
    row3 = result[result['gamepk'] == 'g3'].iloc[0]

    assert row3['pitcher_this_season_start_ip_starts_n'] == 2
    assert row3['pitcher_this_season_start_ip_avg_ip_per_start'] == pytest.approx(5.0)


def _this_season_row(personId='10', gamepk='g1', game_season=2024, starts_n=0, avg_ip=None):
    return {
        'personId': personId, 'gamepk': gamepk, 'game_season': game_season,
        'pitcher_this_season_start_ip_starts_n': starts_n,
        'pitcher_this_season_start_ip_avg_ip_per_start': avg_ip,
    }


def test_build_expected_start_innings_uses_last_season_baseline_when_no_starts_yet_this_season():
    """Season opener: this-season starts_n=0 -> shrinkage weight=0 ->
    expected innings equals last season's baseline exactly, untouched by
    this season's (nonexistent) sample."""

    this_season = pd.DataFrame([_this_season_row(starts_n=0, avg_ip=None)])
    last_season = pd.DataFrame([{
        'personId': '10', 'game_season': 2024,
        'pitcher_last_season_start_ip_avg_ip_per_start': 6.0,
        'pitcher_last_season_start_ip_n_starts': 20,
    }])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_ip_per_start': 5.0}])

    result = build_expected_start_innings(last_season, this_season, league)
    row = result.iloc[0]

    assert row['expected_start_innings_weight'] == pytest.approx(0.0)
    assert row['expected_start_innings'] == pytest.approx(6.0)


def test_build_expected_start_innings_blends_toward_this_season_as_starts_accumulate():
    """last-season baseline=6.0, this-season avg=4.0 over 5 starts, k=5 ->
    weight = 5/(5+5) = 0.5 -> expected = 0.5*6.0 + 0.5*4.0 = 5.0. Proves the
    blend actually shrinks toward the emerging in-season number, not just
    averaging the two blindly regardless of sample size."""

    this_season = pd.DataFrame([_this_season_row(starts_n=5, avg_ip=4.0)])
    last_season = pd.DataFrame([{
        'personId': '10', 'game_season': 2024,
        'pitcher_last_season_start_ip_avg_ip_per_start': 6.0,
        'pitcher_last_season_start_ip_n_starts': 20,
    }])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_ip_per_start': 5.0}])

    result = build_expected_start_innings(last_season, this_season, league, k=5.0)
    row = result.iloc[0]

    assert row['expected_start_innings_weight'] == pytest.approx(0.5)
    assert row['expected_start_innings'] == pytest.approx(5.0)


def test_build_expected_start_innings_falls_back_to_league_avg_for_rookie_with_no_last_season():
    """A pitcher with no row in the last-season table at all (rookie call-up)
    falls back to the league-wide average IP/start rather than NaN."""

    this_season = pd.DataFrame([_this_season_row(personId='99', starts_n=0, avg_ip=None)])
    last_season = pd.DataFrame([], columns=[
        'personId', 'game_season', 'pitcher_last_season_start_ip_avg_ip_per_start',
        'pitcher_last_season_start_ip_n_starts',
    ])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_ip_per_start': 5.0}])

    result = build_expected_start_innings(last_season, this_season, league)
    row = result.iloc[0]

    assert row['expected_start_innings'] == pytest.approx(5.0)


def test_build_expected_start_innings_exposes_raw_components_not_just_the_blend():
    """Every input to the formula survives as its own column, not just the
    final blended number — same 'expose raw denominators' principle as
    rolling_stats.py's sample-size columns."""

    this_season = pd.DataFrame([_this_season_row(starts_n=5, avg_ip=4.0)])
    last_season = pd.DataFrame([{
        'personId': '10', 'game_season': 2024,
        'pitcher_last_season_start_ip_avg_ip_per_start': 6.0,
        'pitcher_last_season_start_ip_n_starts': 20,
    }])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_ip_per_start': 5.0}])

    result = build_expected_start_innings(last_season, this_season, league)

    for col in [
        'pitcher_last_season_start_ip_avg_ip_per_start', 'pitcher_last_season_start_ip_n_starts',
        'pitcher_this_season_start_ip_avg_ip_per_start', 'pitcher_this_season_start_ip_starts_n',
        'league_last_season_avg_ip_per_start',
        'expected_start_innings_weight', 'expected_start_innings',
    ]:
        assert col in result.columns


# --------------------------- PITCHER START PA (batters-faced estimate, k_predictor) --------------------------- #
# k_predictor's own 3-level pitcher -> team -> league cascade, mirroring the pitcher ->
# league IP cascade above in shape but pbp-derived (build_pitcher_start_pa_this_season
# takes pbp only, no pitcher_boxscore) and with a team-level middle rung this file's
# existing IP cascade deliberately does not have — see season_stats.py's own note on
# why that one wasn't retrofitted instead.

def _start_pa_this_season_pbp_row(pitcher_id='10', pitcher_team_id='T1', gamepk='g1',
                                   game_date='2024-04-01', game_datetime='2024-04-01 19:00',
                                   play_id=1, pitcher_role='sp', play_result='Single'):
    return {
        'pitcher_id': pitcher_id, 'pitcher_team_id': pitcher_team_id, 'gamepk': gamepk,
        'game_date': pd.Timestamp(game_date), 'game_datetime': pd.Timestamp(game_datetime, tz='UTC'),
        'game_season': 2024, 'pitcher_role': pitcher_role, 'play_id': play_id, 'pitch_number': 1,
        'play_result': play_result,
    }


def _start_pa_this_season_pbp():
    rows = []
    # g1 on T1: 3 batters faced
    for pid in range(1, 4):
        rows.append(_start_pa_this_season_pbp_row(gamepk='g1', play_id=pid))
    # g2 on T1: 5 batters faced
    for pid in range(1, 6):
        rows.append(_start_pa_this_season_pbp_row(
            gamepk='g2', game_date='2024-04-06', game_datetime='2024-04-06 19:00', play_id=pid,
        ))
    return pd.DataFrame(rows)


def test_build_pitcher_start_pa_this_season_rolls_forward_avg_and_start_count():
    """One prior start (3 batters faced) rolled into start 2: avg = 3.0,
    starts_n = 1 — mirrors the IP version's same rolling/shift(1) behavior,
    just on pbp-derived batter counts."""

    result = build_pitcher_start_pa_this_season(_start_pa_this_season_pbp())
    row_g2 = result[result['gamepk'] == 'g2'].iloc[0]

    assert row_g2['pitcher_this_season_start_pa_starts_n'] == 1
    assert row_g2['pitcher_this_season_start_pa_avg_pa_per_start'] == pytest.approx(3.0)


def test_build_pitcher_start_pa_this_season_first_start_is_nan():
    """A pitcher's first start of the season has nothing prior this season
    to average — must be NaN, not 0."""

    result = build_pitcher_start_pa_this_season(_start_pa_this_season_pbp())
    row_g1 = result[result['gamepk'] == 'g1'].iloc[0]

    assert row_g1['pitcher_this_season_start_pa_starts_n'] == 0
    assert pd.isna(row_g1['pitcher_this_season_start_pa_avg_pa_per_start'])


def test_build_pitcher_start_pa_this_season_carries_pitchers_current_team_through_a_trade():
    """New behavior the IP cascade's this-season table doesn't need: a
    pitcher's pitcher_team_id must reflect whichever team he's on for THAT
    start, not a single season-long team — so build_expected_batters_faced
    can look up the team fallback for his CURRENT team (post-trade), not
    whichever team his last-season stats belonged to."""

    rows = [_start_pa_this_season_pbp_row(gamepk='g1', pitcher_team_id='T1', play_id=1)]
    rows.append(_start_pa_this_season_pbp_row(
        gamepk='g2', pitcher_team_id='T2', game_date='2024-05-01', game_datetime='2024-05-01 19:00', play_id=1,
    ))

    result = build_pitcher_start_pa_this_season(pd.DataFrame(rows))

    assert result[result['gamepk'] == 'g1'].iloc[0]['pitcher_team_id'] == 'T1'
    assert result[result['gamepk'] == 'g2'].iloc[0]['pitcher_team_id'] == 'T2'


# --------------------------- PITCHER REST DAYS / WORKLOAD DENSITY (batters_faced_predictor v2) --------------------------- #
# Mirrors build_team_rest_days's shape (sort by game_datetime, .diff() per entity)
# but keyed on the pitcher (personId) instead of team_id, and scoped to
# pitcher_role == 'sp' like every other start-grain builder in this file — a
# relief appearance in between two starts isn't "his last start."

def test_build_pitcher_rest_days_computes_calendar_days_since_last_start():
    rows = [
        _start_pa_this_season_pbp_row(gamepk='g1', play_id=1),
        _start_pa_this_season_pbp_row(
            gamepk='g2', game_date='2024-04-05', game_datetime='2024-04-05 19:00', play_id=1,
        ),
    ]

    result = build_pitcher_rest_days(pd.DataFrame(rows))

    row_g2 = result[result['gamepk'] == 'g2'].iloc[0]
    assert row_g2['pitcher_days_since_last_start'] == 4


def test_build_pitcher_rest_days_first_start_is_nan():
    rows = [_start_pa_this_season_pbp_row(gamepk='g1', play_id=1)]

    result = build_pitcher_rest_days(pd.DataFrame(rows))

    assert pd.isna(result.iloc[0]['pitcher_days_since_last_start'])


def test_build_pitcher_rest_days_ignores_bullpen_appearances_between_starts():
    """A relief outing between two starts must not count as "his last start"
    — g3's rest is measured from g1 (the last SP start), not g2 (a bullpen
    appearance 2 days before g3)."""

    rows = [
        _start_pa_this_season_pbp_row(gamepk='g1', game_date='2024-04-01', game_datetime='2024-04-01 19:00', play_id=1),
        _start_pa_this_season_pbp_row(
            gamepk='g2', game_date='2024-04-03', game_datetime='2024-04-03 19:00', play_id=1,
            pitcher_role='bullpen',
        ),
        _start_pa_this_season_pbp_row(gamepk='g3', game_date='2024-04-06', game_datetime='2024-04-06 19:00', play_id=1),
    ]

    result = build_pitcher_rest_days(pd.DataFrame(rows))

    row_g3 = result[result['gamepk'] == 'g3'].iloc[0]
    assert row_g3['pitcher_days_since_last_start'] == 5


def _rest_days_row(personId='10', gamepk='g1', pitcher_days_since_last_start=None):
    return {'personId': personId, 'gamepk': gamepk, 'pitcher_days_since_last_start': pitcher_days_since_last_start}


def _workload_pitcher_boxscore(personId='10', gamepk='g1', game_date='2024-04-01',
                                game_datetime='2024-04-01 19:00', p=90):
    ts = pd.Timestamp(game_datetime)
    ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts
    return {
        'personId': personId, 'gamepk': gamepk, 'game_date': pd.Timestamp(game_date),
        'game_datetime': ts, 'p': p,
    }


def _workload_pbp_row(pitcher_id='10', gamepk='g1', game_date='2024-04-01',
                       game_datetime='2024-04-01 19:00', play_id=1):
    return {
        'pitcher_id': pitcher_id, 'gamepk': gamepk, 'game_date': pd.Timestamp(game_date),
        'game_datetime': pd.Timestamp(game_datetime, tz='UTC'), 'pitcher_role': 'sp',
        'play_id': play_id, 'pitch_number': 1, 'play_result': 'Single',
    }


def test_build_pitcher_workload_density_carries_forward_last_start_pitch_count():
    boxscore = pd.DataFrame([
        _workload_pitcher_boxscore(gamepk='g1', p=90),
        _workload_pitcher_boxscore(gamepk='g2', game_date='2024-04-05', game_datetime='2024-04-05 19:00', p=100),
    ])
    pbp = pd.DataFrame([
        _workload_pbp_row(gamepk='g1'),
        _workload_pbp_row(gamepk='g2', game_date='2024-04-05', game_datetime='2024-04-05 19:00'),
    ])
    rest_days = pd.DataFrame([
        _rest_days_row(gamepk='g1', pitcher_days_since_last_start=None),
        _rest_days_row(gamepk='g2', pitcher_days_since_last_start=4),
    ])

    result = build_pitcher_workload_density(boxscore, pbp, rest_days)

    row_g2 = result[result['gamepk'] == 'g2'].iloc[0]
    assert row_g2['pitcher_last_start_pitches'] == 90


def test_build_pitcher_workload_density_first_start_is_nan():
    boxscore = pd.DataFrame([_workload_pitcher_boxscore(gamepk='g1', p=90)])
    pbp = pd.DataFrame([_workload_pbp_row(gamepk='g1')])
    rest_days = pd.DataFrame([_rest_days_row(gamepk='g1', pitcher_days_since_last_start=None)])

    result = build_pitcher_workload_density(boxscore, pbp, rest_days)

    assert pd.isna(result.iloc[0]['pitcher_last_start_pitches'])


def test_build_pitcher_workload_density_divides_by_rest_days_and_guards_zero():
    boxscore = pd.DataFrame([
        _workload_pitcher_boxscore(gamepk='g1', p=90),
        _workload_pitcher_boxscore(gamepk='g2', game_date='2024-04-04', game_datetime='2024-04-04 19:00', p=80),
        _workload_pitcher_boxscore(gamepk='g3', game_date='2024-04-04', game_datetime='2024-04-04 23:00', p=70),
    ])
    pbp = pd.DataFrame([
        _workload_pbp_row(gamepk='g1'),
        _workload_pbp_row(gamepk='g2', game_date='2024-04-04', game_datetime='2024-04-04 19:00'),
        _workload_pbp_row(gamepk='g3', game_date='2024-04-04', game_datetime='2024-04-04 23:00'),
    ])
    rest_days = pd.DataFrame([
        _rest_days_row(gamepk='g1', pitcher_days_since_last_start=None),
        _rest_days_row(gamepk='g2', pitcher_days_since_last_start=3),
        _rest_days_row(gamepk='g3', pitcher_days_since_last_start=0),
    ])

    result = build_pitcher_workload_density(boxscore, pbp, rest_days)

    row_g2 = result[result['gamepk'] == 'g2'].iloc[0]
    row_g3 = result[result['gamepk'] == 'g3'].iloc[0]
    assert row_g2['pitcher_workload_density'] == pytest.approx(30.0)
    assert pd.isna(row_g3['pitcher_workload_density'])


def _pa_this_season_row(personId='10', gamepk='g1', game_season=2024, pitcher_team_id='T1',
                         starts_n=0, avg_pa=None):
    return {
        'personId': personId, 'gamepk': gamepk, 'game_season': game_season,
        'pitcher_team_id': pitcher_team_id,
        'pitcher_this_season_start_pa_starts_n': starts_n,
        'pitcher_this_season_start_pa_avg_pa_per_start': avg_pa,
    }


def test_build_expected_batters_faced_uses_last_season_baseline_when_no_starts_yet_this_season():
    """Season opener: this-season starts_n=0 -> shrinkage weight=0 ->
    expected batters faced equals last season's baseline exactly."""

    this_season = pd.DataFrame([_pa_this_season_row(starts_n=0, avg_pa=None)])
    last_season = pd.DataFrame([{
        'personId': '10', 'game_season': 2024,
        'pitcher_last_season_start_pa_avg_pa_per_start': 24.0,
        'pitcher_last_season_start_pa_n_starts': 20,
    }])
    team = pd.DataFrame([{'pitcher_team_id': 'T1', 'game_season': 2024, 'team_last_season_avg_pa_per_start': 22.0}])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_pa_per_start': 21.0}])

    result = build_expected_batters_faced(last_season, this_season, team, league)
    row = result.iloc[0]

    assert row['expected_batters_faced_weight'] == pytest.approx(0.0)
    assert row['expected_batters_faced'] == pytest.approx(24.0)


def test_build_expected_batters_faced_blends_toward_this_season_as_starts_accumulate():
    """last-season baseline=24.0, this-season avg=20.0 over 5 starts, k=5 ->
    weight=0.5 -> expected = 0.5*24.0 + 0.5*20.0 = 22.0."""

    this_season = pd.DataFrame([_pa_this_season_row(starts_n=5, avg_pa=20.0)])
    last_season = pd.DataFrame([{
        'personId': '10', 'game_season': 2024,
        'pitcher_last_season_start_pa_avg_pa_per_start': 24.0,
        'pitcher_last_season_start_pa_n_starts': 20,
    }])
    team = pd.DataFrame([{'pitcher_team_id': 'T1', 'game_season': 2024, 'team_last_season_avg_pa_per_start': 22.0}])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_pa_per_start': 21.0}])

    result = build_expected_batters_faced(last_season, this_season, team, league, k=5.0)
    row = result.iloc[0]

    assert row['expected_batters_faced_weight'] == pytest.approx(0.5)
    assert row['expected_batters_faced'] == pytest.approx(22.0)


def test_build_expected_batters_faced_falls_back_to_team_avg_before_league_when_pitcher_missing():
    """The genuinely new middle rung: a pitcher with no last-season row of
    his own (rookie call-up, but on an established team) falls back to his
    CURRENT team's average — not straight to the league-wide average, even
    though both are available. Proves team is tried before league, not just
    that league eventually gets used."""

    this_season = pd.DataFrame([_pa_this_season_row(personId='99', pitcher_team_id='T1', starts_n=0, avg_pa=None)])
    last_season = pd.DataFrame([], columns=[
        'personId', 'game_season', 'pitcher_last_season_start_pa_avg_pa_per_start',
        'pitcher_last_season_start_pa_n_starts',
    ])
    team = pd.DataFrame([{'pitcher_team_id': 'T1', 'game_season': 2024, 'team_last_season_avg_pa_per_start': 22.0}])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_pa_per_start': 21.0}])

    result = build_expected_batters_faced(last_season, this_season, team, league)
    row = result.iloc[0]

    assert row['expected_batters_faced'] == pytest.approx(22.0)


def test_build_expected_batters_faced_falls_back_to_league_avg_when_pitcher_and_team_both_missing():
    """A rookie on a team with no last-season SP-start data at all (e.g. an
    expansion team's first season — rare, but the safety net must hold)
    falls all the way back to the league-wide average."""

    this_season = pd.DataFrame([_pa_this_season_row(personId='99', pitcher_team_id='T9', starts_n=0, avg_pa=None)])
    last_season = pd.DataFrame([], columns=[
        'personId', 'game_season', 'pitcher_last_season_start_pa_avg_pa_per_start',
        'pitcher_last_season_start_pa_n_starts',
    ])
    team = pd.DataFrame([], columns=['pitcher_team_id', 'game_season', 'team_last_season_avg_pa_per_start'])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_pa_per_start': 21.0}])

    result = build_expected_batters_faced(last_season, this_season, team, league)
    row = result.iloc[0]

    assert row['expected_batters_faced'] == pytest.approx(21.0)


def test_build_expected_batters_faced_exposes_raw_components_not_just_the_blend():
    """Every input to the formula survives as its own column, including the
    new team-level rung — same 'expose raw denominators' principle as
    build_expected_start_innings and rolling_stats.py's sample-size columns."""

    this_season = pd.DataFrame([_pa_this_season_row(starts_n=5, avg_pa=20.0)])
    last_season = pd.DataFrame([{
        'personId': '10', 'game_season': 2024,
        'pitcher_last_season_start_pa_avg_pa_per_start': 24.0,
        'pitcher_last_season_start_pa_n_starts': 20,
    }])
    team = pd.DataFrame([{'pitcher_team_id': 'T1', 'game_season': 2024, 'team_last_season_avg_pa_per_start': 22.0}])
    league = pd.DataFrame([{'game_season': 2024, 'league_last_season_avg_pa_per_start': 21.0}])

    result = build_expected_batters_faced(last_season, this_season, team, league)

    for col in [
        'pitcher_last_season_start_pa_avg_pa_per_start', 'pitcher_last_season_start_pa_n_starts',
        'pitcher_this_season_start_pa_avg_pa_per_start', 'pitcher_this_season_start_pa_starts_n',
        'team_last_season_avg_pa_per_start', 'league_last_season_avg_pa_per_start',
        'expected_batters_faced_weight', 'expected_batters_faced',
    ]:
        assert col in result.columns


# --------------------------- BATTER SLOT EXPANSION (count-distribution check) --------------------------- #
# Same expansion shape as build_pitcher_role_by_inning above (cross join a pregame
# estimate against a fixed range, then tag/filter), but on the batter-lineup axis
# instead of innings: expected_batters_faced -> one synthetic row per slot, cycling
# through the real 9-batter lineup order. Feeds k_predictor's total-strikeout
# count-distribution check — see k_predictor/experiments/count_distribution_check/.

def _slot_expansion_pitcher_starts():
    return pd.DataFrame([
        {'gamepk': 'g1', 'expected_pitcher_key_id': '10', 'expected_batters_faced': 11.0},
    ])


def _slot_expansion_batting_order(gamepk='g1'):
    return pd.DataFrame([
        {'gamepk': gamepk, 'batter_id': f'B{i}', 'batting_order': i} for i in range(1, 10)
    ])


def test_build_batter_slot_expansion_cycles_lineup_positions_and_caps_times_through_order():
    """11 expected batters faced = one full 9-batter cycle plus 2 more: slot 10
    must cycle back to lineup position 1 (the leadoff hitter again), 2nd time
    through the order. Slot 1 is the 1st time through."""

    result = build_batter_slot_expansion(_slot_expansion_pitcher_starts(), _slot_expansion_batting_order())

    assert len(result) == 11
    row1 = result[result['slot'] == 1].iloc[0]
    row10 = result[result['slot'] == 10].iloc[0]

    assert row1['lineup_position'] == 1
    assert row1['expected_times_through_order'] == 1
    assert row10['lineup_position'] == 1
    assert row10['expected_times_through_order'] == 2


def test_build_batter_slot_expansion_caps_times_through_order_beyond_third_cycle():
    """A long outing (28 expected batters faced = cycle 4 for the final slots)
    must cap expected_times_through_order at 3 — the same Lichtman-research-
    based cap _add_pbp_times_through_order and expected_role.py already use
    everywhere else this concept appears, so the trained classifier sees the
    same feature shape it was trained on."""

    pitcher_starts = pd.DataFrame([
        {'gamepk': 'g1', 'expected_pitcher_key_id': '10', 'expected_batters_faced': 28.0},
    ])

    result = build_batter_slot_expansion(pitcher_starts, _slot_expansion_batting_order())

    row28 = result[result['slot'] == 28].iloc[0]
    assert row28['expected_times_through_order'] == 3


def test_build_batter_slot_expansion_rounds_estimate_and_joins_real_batter_id():
    """expected_batters_faced is itself a blended float (build_expected_batters_faced's
    output) — must round, not truncate, to decide slot count. Each slot must
    carry the SPECIFIC real batter_id in that lineup position, not just the
    position number."""

    pitcher_starts = pd.DataFrame([
        {'gamepk': 'g1', 'expected_pitcher_key_id': '10', 'expected_batters_faced': 21.6},
    ])

    result = build_batter_slot_expansion(pitcher_starts, _slot_expansion_batting_order())

    assert len(result) == 22  # round(21.6) == 22, not truncated to 21
    row3 = result[result['slot'] == 3].iloc[0]
    assert row3['batter_id'] == 'B3'


def test_build_batter_slot_expansion_deduplicates_lineup_position_collisions():
    """Real MLB batting-order data isn't always a clean 1:1 (gamepk,
    batting_order) -> batter_id mapping — a substitution mid-game is
    sometimes logged sharing the SAME lineup-slot number as the starter it
    replaced, rather than a distinct 501/502-style code (confirmed against
    real 2024 data: ~half of all (gamepk, batting_order) pairs had 2 batter
    rows). A lineup-position lookup must still resolve to exactly ONE batter
    per slot — the join key here is the slot NUMBER, not batter identity, so
    an unresolved collision silently doubles every downstream slot instead
    of erroring. Must dedupe to one row per (gamepk, batting_order),
    deterministically (first occurrence), rather than fan out."""

    batting_order = pd.DataFrame([
        {'gamepk': 'g1', 'batter_id': 'B5_starter', 'batting_order': 5},
        {'gamepk': 'g1', 'batter_id': 'B5_sub', 'batting_order': 5},  # collision — same slot
    ] + [
        {'gamepk': 'g1', 'batter_id': f'B{i}', 'batting_order': i} for i in range(1, 10) if i != 5
    ])
    pitcher_starts = pd.DataFrame([
        {'gamepk': 'g1', 'expected_pitcher_key_id': '10', 'expected_batters_faced': 9.0},
    ])

    result = build_batter_slot_expansion(pitcher_starts, batting_order)

    assert len(result) == 9  # exactly one row per slot, not 10 from the collision
    slot5 = result[result['slot'] == 5].iloc[0]
    assert slot5['batter_id'] == 'B5_starter'


def test_build_pitcher_role_by_inning_rounds_up_and_splits_sp_vs_bullpen():
    """expected_start_innings=5.4 -> ceil=6 -> innings 1-6 are 'sp' (a
    partial inning pitched still counts as that inning belonging to the
    starter), innings 7-9 are 'bullpen'."""

    team_game = pd.DataFrame([{'team_id': 'H', 'gamepk': 'g1', 'expected_start_innings': 5.4}])

    result = build_pitcher_role_by_inning(team_game)

    sp_innings = result[result['pitcher_role'] == 'sp']['inning'].tolist()
    bullpen_innings = result[result['pitcher_role'] == 'bullpen']['inning'].tolist()
    assert sp_innings == [1, 2, 3, 4, 5, 6]
    assert bullpen_innings == [7, 8, 9]


def test_build_pitcher_role_by_inning_produces_exactly_nine_rows_per_game():
    team_game = pd.DataFrame([{'team_id': 'H', 'gamepk': 'g1', 'expected_start_innings': 6.0}])

    result = build_pitcher_role_by_inning(team_game)

    assert len(result) == 9
    assert result['inning'].tolist() == list(range(1, 10))


def test_build_pitcher_role_by_inning_clips_at_nine_for_an_implausibly_high_estimate():
    """A (implausible but not impossible with a small/noisy sample)
    expected_start_innings above 9 must not create a 10th inning or leave
    zero bullpen rows undefined — clipped to the 9-inning game length."""

    team_game = pd.DataFrame([{'team_id': 'H', 'gamepk': 'g1', 'expected_start_innings': 11.0}])

    result = build_pitcher_role_by_inning(team_game)

    assert len(result) == 9
    assert (result['pitcher_role'] == 'sp').all()


def test_build_pitcher_role_by_inning_all_bullpen_when_estimate_is_missing():
    """No expected_start_innings estimate at all (shouldn't happen given
    build_expected_start_innings' league-wide fallback, but documented
    degenerate behavior rather than silently crashing): every inning
    defaults to 'bullpen' rather than guessing a starter depth."""

    team_game = pd.DataFrame([{'team_id': 'H', 'gamepk': 'g1', 'expected_start_innings': np.nan}])

    result = build_pitcher_role_by_inning(team_game)

    assert (result['pitcher_role'] == 'bullpen').all()


def test_build_pitcher_role_by_inning_handles_multiple_games_independently():
    team_game = pd.DataFrame([
        {'team_id': 'H', 'gamepk': 'g1', 'expected_start_innings': 5.0},
        {'team_id': 'A', 'gamepk': 'g1', 'expected_start_innings': 7.0},
    ])

    result = build_pitcher_role_by_inning(team_game)

    assert len(result) == 18
    h_sp = result[(result['team_id'] == 'H') & (result['pitcher_role'] == 'sp')]
    a_sp = result[(result['team_id'] == 'A') & (result['pitcher_role'] == 'sp')]
    assert len(h_sp) == 5
    assert len(a_sp) == 7


# --------------------------- BATTERS-FACED RESIDUAL DISTRIBUTION --------------------------- #
# build_expected_batters_faced (above) is a fixed POINT estimate — k_predictor's
# count-distribution-check diagnostic showed its error scales with
# expected_batters_faced_weight (thin this-season sample = bigger miss). These two
# functions turn that point estimate + known error-correlate into a real distribution:
# fit an empirical residual histogram per weight bin (build_batters_faced_residual_bins),
# then shift/scatter that bin's histogram around a new start's own point estimate
# (build_batters_faced_distribution). See ROADMAP.md's batters-faced-distribution plan.

def test_build_batters_faced_residual_bins_single_bin_is_unconditional():
    """n_bins=1 must collapse to the plain unconditional residual histogram —
    4 starts, residuals [0, 1, 1, 2] -> pmf {0: 0.25, 1: 0.5, 2: 0.25}, and the
    single bin's edges must span the entire real line (no future weight value
    can fall outside it)."""

    fit_starts = pd.DataFrame([
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.1, 'realized_batters_faced': 10},
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.4, 'realized_batters_faced': 11},
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.6, 'realized_batters_faced': 11},
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.9, 'realized_batters_faced': 12},
    ])

    result = build_batters_faced_residual_bins(fit_starts, n_bins=1)

    assert result['weight_bin'].nunique() == 1
    pmf = dict(zip(result['residual'], result['probability']))
    assert pmf == pytest.approx({0: 0.25, 1: 0.5, 2: 0.25})
    assert result['weight_bin_lower'].iloc[0] == -np.inf
    assert result['weight_bin_upper'].iloc[0] == np.inf


def test_build_batters_faced_residual_bins_two_bins_isolate_own_residuals():
    """Two bins must NOT pool their residuals — a low-weight (thin-sample)
    start's residual must never leak into the high-weight bin's histogram,
    and vice versa."""

    fit_starts = pd.DataFrame([
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.0, 'realized_batters_faced': 10},  # residual 0
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.1, 'realized_batters_faced': 12},  # residual 2
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.8, 'realized_batters_faced': 9},   # residual -1
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.9, 'realized_batters_faced': 11},  # residual 1
    ])

    result = build_batters_faced_residual_bins(fit_starts, n_bins=2)

    low_bin = result[result['weight_bin'] == result['weight_bin'].min()]
    high_bin = result[result['weight_bin'] == result['weight_bin'].max()]

    assert dict(zip(low_bin['residual'], low_bin['probability'])) == pytest.approx({0: 0.5, 2: 0.5})
    assert dict(zip(high_bin['residual'], high_bin['probability'])) == pytest.approx({-1: 0.5, 1: 0.5})
    assert -1 not in low_bin['residual'].tolist()
    assert 1 not in low_bin['residual'].tolist()


def test_build_batters_faced_residual_bins_edges_are_contiguous_and_span_full_range():
    """Bin edges must be contiguous (no gap a future weight value could fall
    into) and the outer edges must be -inf/+inf so every possible weight
    resolves to exactly one bin — same 'always resolves' contract
    build_expected_batters_faced's own fallback cascade guarantees."""

    fit_starts = pd.DataFrame([
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.0, 'realized_batters_faced': 10},
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.1, 'realized_batters_faced': 12},
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.8, 'realized_batters_faced': 9},
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.9, 'realized_batters_faced': 11},
    ])

    result = build_batters_faced_residual_bins(fit_starts, n_bins=2)

    bins = result[['weight_bin', 'weight_bin_lower', 'weight_bin_upper']].drop_duplicates().sort_values('weight_bin')
    assert bins['weight_bin_lower'].iloc[0] == -np.inf
    assert bins['weight_bin_upper'].iloc[-1] == np.inf
    assert bins['weight_bin_upper'].iloc[0] == bins['weight_bin_lower'].iloc[1]


def test_build_batters_faced_residual_bins_drops_rows_with_missing_expected_batters_faced():
    """Real data has a small number of starts with no expected_batters_faced
    at all — build_expected_batters_faced's 3-level cascade (pitcher -> team
    -> league) still returns NaN when even the this-season fallback has no
    starts yet (a pitcher's true first-ever MLB start with no league data
    for that season, confirmed against real 2018-2023 fit data). These rows
    must be dropped before computing residuals, not crash the int cast —
    a NaN residual can't be assigned to any bin anyway."""

    fit_starts = pd.DataFrame([
        {'expected_batters_faced': 10.0, 'expected_batters_faced_weight': 0.5, 'realized_batters_faced': 11},
        {'expected_batters_faced': np.nan, 'expected_batters_faced_weight': 0.5, 'realized_batters_faced': 8},
    ])

    result = build_batters_faced_residual_bins(fit_starts, n_bins=1)

    assert dict(zip(result['residual'], result['probability'])) == pytest.approx({1: 1.0})


# --------------------------- BATTERS-FACED DISTRIBUTION --------------------------- #

def test_build_batters_faced_distribution_shifts_residual_pmf_by_point_estimate():
    """A bin's residual pmf {0: 0.5, 2: 0.5} applied to a start with
    expected_batters_faced=10.4 (rounds to 10) must land mass at n=10 and
    n=12, nowhere else."""

    residual_bins = pd.DataFrame([
        {'weight_bin': 0, 'weight_bin_lower': -np.inf, 'weight_bin_upper': np.inf, 'residual': 0, 'probability': 0.5},
        {'weight_bin': 0, 'weight_bin_lower': -np.inf, 'weight_bin_upper': np.inf, 'residual': 2, 'probability': 0.5},
    ])
    expected_pa = pd.DataFrame([{'expected_batters_faced': 10.4, 'expected_batters_faced_weight': 0.05}])

    result = build_batters_faced_distribution(expected_pa, residual_bins, max_slots=15)
    pmf = result['batters_faced_pmf'].iloc[0]

    assert len(pmf) == 16
    assert pmf[10] == pytest.approx(0.5)
    assert pmf[12] == pytest.approx(0.5)
    assert pmf.sum() == pytest.approx(1.0)


def test_build_batters_faced_distribution_clips_and_accumulates_at_boundary():
    """Residuals that push below 0 must clip to n=0 and ACCUMULATE mass
    there, not overwrite each other — {-5: 0.4, -3: 0.6} against
    expected_batters_faced=2.0 gives raw values -3 and -1, both < 0, both
    clip to n=0: pmf[0] must be 0.4+0.6=1.0, not just the last write."""

    residual_bins = pd.DataFrame([
        {'weight_bin': 0, 'weight_bin_lower': -np.inf, 'weight_bin_upper': np.inf, 'residual': -5, 'probability': 0.4},
        {'weight_bin': 0, 'weight_bin_lower': -np.inf, 'weight_bin_upper': np.inf, 'residual': -3, 'probability': 0.6},
    ])
    expected_pa = pd.DataFrame([{'expected_batters_faced': 2.0, 'expected_batters_faced_weight': 0.05}])

    result = build_batters_faced_distribution(expected_pa, residual_bins, max_slots=15)
    pmf = result['batters_faced_pmf'].iloc[0]

    assert pmf[0] == pytest.approx(1.0)
    assert pmf.sum() == pytest.approx(1.0)


def test_build_batters_faced_distribution_always_returns_full_length_array():
    """Regardless of how sparse a bin's own residual support is, the output
    pmf must always be the full max_slots+1 length — matches
    poisson_binomial_mixture_pmf's fixed-length contract."""

    residual_bins = pd.DataFrame([
        {'weight_bin': 0, 'weight_bin_lower': -np.inf, 'weight_bin_upper': np.inf, 'residual': 1, 'probability': 1.0},
    ])
    expected_pa = pd.DataFrame([{'expected_batters_faced': 5.0, 'expected_batters_faced_weight': 0.05}])

    result = build_batters_faced_distribution(expected_pa, residual_bins, max_slots=20)
    pmf = result['batters_faced_pmf'].iloc[0]

    assert len(pmf) == 21


def _anomaly_row(personId='10', gamepk='g1', game_season=2024, game_date='2024-04-01',
                  realized_batters_faced=20, expected_batters_faced=20.0,
                  expected_batters_faced_weight=0.5):
    return {
        'personId': personId, 'gamepk': gamepk, 'game_season': game_season,
        'game_date': pd.Timestamp(game_date), 'realized_batters_faced': realized_batters_faced,
        'expected_batters_faced': expected_batters_faced,
        'expected_batters_faced_weight': expected_batters_faced_weight,
    }


def test_build_pitcher_anomaly_count_flags_prior_anomaly_and_excludes_current_start():
    """Start 2 is anomalous (6 < 0.6*24=14.4, weight>=0.3) but its OWN count
    must still be 0 — only start 3 should reflect it, via the prior start."""

    rows = [
        _anomaly_row(gamepk='g1', game_date='2024-04-01', realized_batters_faced=25,
                     expected_batters_faced=24, expected_batters_faced_weight=0.5),
        _anomaly_row(gamepk='g2', game_date='2024-04-06', realized_batters_faced=6,
                     expected_batters_faced=24, expected_batters_faced_weight=0.5),
        _anomaly_row(gamepk='g3', game_date='2024-04-11', realized_batters_faced=23,
                     expected_batters_faced=23, expected_batters_faced_weight=0.7),
    ]

    result = build_pitcher_anomaly_count_this_season(pd.DataFrame(rows))

    counts = result.set_index('gamepk')['pitcher_anomaly_count_this_season']
    assert counts['g1'] == 0
    assert counts['g2'] == 0
    assert counts['g3'] == 1


def test_build_pitcher_anomaly_count_ignores_low_weight_starts():
    """A short start (4 < 0.6*20=12) with weight below min_weight=0.3 (a
    cold-start pitcher whose cascade estimate isn't reliable yet) must NOT
    count as an anomaly for later starts — avoids conflating this with the
    already-closed cold-start problem."""

    rows = [
        _anomaly_row(gamepk='g1', game_date='2024-04-01', realized_batters_faced=4,
                     expected_batters_faced=20, expected_batters_faced_weight=0.1),
        _anomaly_row(gamepk='g2', game_date='2024-04-06', realized_batters_faced=22,
                     expected_batters_faced=21, expected_batters_faced_weight=0.4),
    ]

    result = build_pitcher_anomaly_count_this_season(pd.DataFrame(rows))

    row_g2 = result[result['gamepk'] == 'g2'].iloc[0]
    assert row_g2['pitcher_anomaly_count_this_season'] == 0


def test_build_pitcher_anomaly_count_resets_each_season():
    """An anomaly in 2023 must not carry into the pitcher's 2024 count."""

    rows = [
        _anomaly_row(gamepk='g1', game_season=2023, game_date='2023-04-01',
                     realized_batters_faced=6, expected_batters_faced=24,
                     expected_batters_faced_weight=0.5),
        _anomaly_row(gamepk='g2', game_season=2024, game_date='2024-04-01',
                     realized_batters_faced=22, expected_batters_faced=21,
                     expected_batters_faced_weight=0.4),
    ]

    result = build_pitcher_anomaly_count_this_season(pd.DataFrame(rows))

    row_g2 = result[result['gamepk'] == 'g2'].iloc[0]
    assert row_g2['pitcher_anomaly_count_this_season'] == 0
