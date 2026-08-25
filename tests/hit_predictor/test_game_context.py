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
    build_pitcher_role_by_inning,
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
