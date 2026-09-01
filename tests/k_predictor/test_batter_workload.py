import pandas as pd
import pytest

from models.k_predictor.processing.features.batter_workload import (
    build_batter_shrunk_k_rate,
    build_batter_shrunk_obp_slg,
    build_opposing_lineup_extremum,
)


def _rolling_row(**overrides):
    row = {
        'batter_id': 'B1',
        'game_season': 2023,
        'batter_roll_season_pa_strikeout_rate': 0.15,
        'batter_roll_season_pa_total': 50,
    }
    row.update(overrides)
    return row


def _season_row(**overrides):
    row = {
        'batter_id': 'B1',
        'game_season': 2023,
        'batter_last_season_pa_strikeout_rate': 0.25,
    }
    row.update(overrides)
    return row


def test_build_batter_shrunk_k_rate_blends_toward_rolling_as_sample_grows():
    """shrinkage_weight = pa_total / (pa_total + k). k=50, pa_total=50 ->
    weight=0.5. Hand-computed: 0.5*0.25 + 0.5*0.15 = 0.20."""

    rolling = pd.DataFrame([_rolling_row(batter_roll_season_pa_strikeout_rate=0.15, batter_roll_season_pa_total=50)])
    season = pd.DataFrame([_season_row(batter_last_season_pa_strikeout_rate=0.25)])

    result = build_batter_shrunk_k_rate(rolling, season, window='season', k=50.0)

    assert result['batter_shrunk_k_rate'].iloc[0] == pytest.approx(0.5 * 0.25 + 0.5 * 0.15)
    assert result['batter_shrunk_k_rate_weight'].iloc[0] == pytest.approx(0.5)


def test_build_batter_shrunk_k_rate_trusts_last_season_at_zero_pa():
    """pa_total=0 -> weight=0 -> shrunk value equals last season's K rate
    exactly, same season-opener behavior as build_pitcher_shrunk_whip."""

    rolling = pd.DataFrame([_rolling_row(batter_roll_season_pa_strikeout_rate=None, batter_roll_season_pa_total=0)])
    season = pd.DataFrame([_season_row(batter_last_season_pa_strikeout_rate=0.25)])

    result = build_batter_shrunk_k_rate(rolling, season, window='season', k=50.0)

    assert result['batter_shrunk_k_rate'].iloc[0] == pytest.approx(0.25)
    assert result['batter_shrunk_k_rate_weight'].iloc[0] == 0.0


def test_build_batter_shrunk_k_rate_falls_back_to_rolling_when_no_last_season_stat():
    """A rookie/unseen batter has no last-season row — baseline falls back
    to this season's own rolling K rate rather than producing NaN."""

    rolling = pd.DataFrame([_rolling_row(batter_roll_season_pa_strikeout_rate=0.18, batter_roll_season_pa_total=30)])
    season = pd.DataFrame([_season_row(batter_last_season_pa_strikeout_rate=None)])

    result = build_batter_shrunk_k_rate(rolling, season, window='season', k=50.0)

    assert result['batter_shrunk_k_rate'].iloc[0] == pytest.approx(0.18)


def _obp_slg_rolling_row(**overrides):
    row = {
        'personId': 'B1',
        'game_season': 2023,
        'batter_roll_season_obp': 0.30,
        'batter_roll_season_slg': 0.40,
        'batter_roll_season_plate_appearances': 25,
    }
    row.update(overrides)
    return row


def _obp_slg_season_row(**overrides):
    row = {
        'personId': 'B1',
        'game_season': 2023,
        'batter_last_season_obp': 0.34,
        'batter_last_season_slg': 0.50,
    }
    row.update(overrides)
    return row


def test_build_batter_shrunk_obp_slg_blends_both_metrics():
    """Same shrinkage recipe as build_batter_shrunk_k_rate, applied to OBP and
    SLG independently, PA-weighted via plate_appearances (the box-score-rolling
    sample-size column, since this feeds off build_batter_rolling_stats rather
    than the pbp-based pa_total). weight = 25/(25+50) = 1/3."""

    rolling = pd.DataFrame([_obp_slg_rolling_row()])
    season = pd.DataFrame([_obp_slg_season_row()])

    result = build_batter_shrunk_obp_slg(rolling, season, window='season', k=50.0)

    weight = 25 / 75
    assert result['batter_shrunk_obp_weight'].iloc[0] == pytest.approx(weight)
    assert result['batter_shrunk_obp'].iloc[0] == pytest.approx((1 - weight) * 0.34 + weight * 0.30)
    assert result['batter_shrunk_slg'].iloc[0] == pytest.approx((1 - weight) * 0.50 + weight * 0.40)


def test_build_batter_shrunk_obp_slg_trusts_last_season_at_zero_pa():
    rolling = pd.DataFrame([_obp_slg_rolling_row(
        batter_roll_season_obp=None, batter_roll_season_slg=None, batter_roll_season_plate_appearances=0,
    )])
    season = pd.DataFrame([_obp_slg_season_row(batter_last_season_obp=0.34, batter_last_season_slg=0.50)])

    result = build_batter_shrunk_obp_slg(rolling, season, window='season', k=50.0)

    assert result['batter_shrunk_obp'].iloc[0] == pytest.approx(0.34)
    assert result['batter_shrunk_slg'].iloc[0] == pytest.approx(0.50)
    assert result['batter_shrunk_obp_weight'].iloc[0] == 0.0


def test_build_batter_shrunk_obp_slg_falls_back_to_rolling_when_no_last_season_stat():
    rolling = pd.DataFrame([_obp_slg_rolling_row(
        batter_roll_season_obp=0.28, batter_roll_season_slg=0.38, batter_roll_season_plate_appearances=40,
    )])
    season = pd.DataFrame([_obp_slg_season_row(batter_last_season_obp=None, batter_last_season_slg=None)])

    result = build_batter_shrunk_obp_slg(rolling, season, window='season', k=50.0)

    assert result['batter_shrunk_obp'].iloc[0] == pytest.approx(0.28)
    assert result['batter_shrunk_slg'].iloc[0] == pytest.approx(0.38)


def _lineup_slot_row(personId, gamepk='g1', batting_order=1, **overrides):
    row = {'personId': personId, 'gamepk': gamepk, 'batting_order': batting_order}
    row.update(overrides)
    return row


def _team_pbp_row(batter_id, gamepk='g1', batter_team_id='T1', **overrides):
    row = {'batter_id': batter_id, 'gamepk': gamepk, 'batter_team_id': batter_team_id}
    row.update(overrides)
    return row


def test_build_opposing_lineup_extremum_takes_min_across_starting_lineup():
    """3 starters for team T1 in g1, shrunk K rates {B1: 0.30, B2: 0.10,
    B3: 0.22} -- the toughest out (B2, lowest K rate) must win the MIN."""

    batter_boxscore = pd.DataFrame([
        _lineup_slot_row('B1', gamepk='g1', batting_order=1),
        _lineup_slot_row('B2', gamepk='g1', batting_order=2),
        _lineup_slot_row('B3', gamepk='g1', batting_order=3),
    ])
    pbp = pd.DataFrame([
        _team_pbp_row('B1', gamepk='g1'),
        _team_pbp_row('B2', gamepk='g1'),
        _team_pbp_row('B3', gamepk='g1'),
    ])
    batter_shrunk = pd.DataFrame([
        {'batter_id': 'B1', 'gamepk': 'g1', 'batter_shrunk_k_rate': 0.30},
        {'batter_id': 'B2', 'gamepk': 'g1', 'batter_shrunk_k_rate': 0.10},
        {'batter_id': 'B3', 'gamepk': 'g1', 'batter_shrunk_k_rate': 0.22},
    ])

    result = build_opposing_lineup_extremum(
        batter_boxscore, pbp, batter_shrunk,
        metric_col='batter_shrunk_k_rate', out_col='opp_team_toughest_out_shrunk_k_rate',
        agg='min',
    )

    row = result.loc[(result['batter_team_id'] == 'T1') & (result['gamepk'] == 'g1')]
    assert row['opp_team_toughest_out_shrunk_k_rate'].iloc[0] == pytest.approx(0.10)


def test_build_opposing_lineup_extremum_takes_max_when_agg_is_max():
    """Same fixture, agg='max' -- the highest value (B1, 0.30) wins instead."""

    batter_boxscore = pd.DataFrame([
        _lineup_slot_row('B1', gamepk='g1', batting_order=1),
        _lineup_slot_row('B2', gamepk='g1', batting_order=2),
        _lineup_slot_row('B3', gamepk='g1', batting_order=3),
    ])
    pbp = pd.DataFrame([
        _team_pbp_row('B1', gamepk='g1'),
        _team_pbp_row('B2', gamepk='g1'),
        _team_pbp_row('B3', gamepk='g1'),
    ])
    batter_shrunk = pd.DataFrame([
        {'batter_id': 'B1', 'gamepk': 'g1', 'batter_shrunk_obp': 0.30},
        {'batter_id': 'B2', 'gamepk': 'g1', 'batter_shrunk_obp': 0.10},
        {'batter_id': 'B3', 'gamepk': 'g1', 'batter_shrunk_obp': 0.22},
    ])

    result = build_opposing_lineup_extremum(
        batter_boxscore, pbp, batter_shrunk,
        metric_col='batter_shrunk_obp', out_col='opp_team_best_batter_shrunk_obp',
        agg='max',
    )

    row = result.loc[(result['batter_team_id'] == 'T1') & (result['gamepk'] == 'g1')]
    assert row['opp_team_best_batter_shrunk_obp'].iloc[0] == pytest.approx(0.30)


def test_build_opposing_lineup_extremum_uses_this_games_shrunk_value_not_other_games():
    """batter_shrunk_df is a PER-GAME rolling table (one row per batter per
    game across the whole multi-season dataset), not a static per-batter
    value -- the join must be scoped to (batter_id, gamepk), or a lineup
    slot fans out against every game that batter ever played and the
    extremum collapses toward whatever thin-sample blip exists anywhere in
    his history, not tonight's real value. B1 has a near-zero shrunk_k_rate
    in an UNRELATED earlier game (g0) and a normal 0.20 in TODAY's game
    (g1) -- the MIN across g1's lineup must reflect g1's value only."""

    batter_boxscore = pd.DataFrame([
        _lineup_slot_row('B1', gamepk='g1', batting_order=1),
        _lineup_slot_row('B2', gamepk='g1', batting_order=2),
    ])
    pbp = pd.DataFrame([
        _team_pbp_row('B1', gamepk='g1'),
        _team_pbp_row('B2', gamepk='g1'),
    ])
    batter_shrunk = pd.DataFrame([
        {'batter_id': 'B1', 'gamepk': 'g0', 'batter_shrunk_k_rate': 0.001},  # unrelated earlier game
        {'batter_id': 'B1', 'gamepk': 'g1', 'batter_shrunk_k_rate': 0.20},   # today's game
        {'batter_id': 'B2', 'gamepk': 'g1', 'batter_shrunk_k_rate': 0.25},
    ])

    result = build_opposing_lineup_extremum(
        batter_boxscore, pbp, batter_shrunk,
        metric_col='batter_shrunk_k_rate', out_col='opp_team_toughest_out_shrunk_k_rate',
        agg='min',
    )

    row = result.loc[(result['batter_team_id'] == 'T1') & (result['gamepk'] == 'g1')]
    assert row['opp_team_toughest_out_shrunk_k_rate'].iloc[0] == pytest.approx(0.20)


def test_build_opposing_lineup_extremum_nan_when_no_lineup_for_game():
    """gamepk g2 has no batting_order rows at all -- must not raise, and must
    not appear in the result with a fabricated value."""

    batter_boxscore = pd.DataFrame([
        _lineup_slot_row('B1', gamepk='g1', batting_order=1),
        # no row at all for gamepk g2
    ])
    pbp = pd.DataFrame([
        _team_pbp_row('B1', gamepk='g1'),
        _team_pbp_row('B1', gamepk='g2'),
    ])
    batter_shrunk = pd.DataFrame([
        {'batter_id': 'B1', 'gamepk': 'g1', 'batter_shrunk_k_rate': 0.20},
        {'batter_id': 'B1', 'gamepk': 'g2', 'batter_shrunk_k_rate': 0.20},
    ])

    result = build_opposing_lineup_extremum(
        batter_boxscore, pbp, batter_shrunk,
        metric_col='batter_shrunk_k_rate', out_col='opp_team_toughest_out_shrunk_k_rate',
        agg='min',
    )

    assert (result['gamepk'] == 'g2').sum() == 0


def test_build_opposing_lineup_extremum_supports_person_id_keyed_shrunk_df():
    """build_batter_shrunk_obp_slg's output is keyed on personId (box-score
    based), not batter_id (pbp based) like build_batter_shrunk_k_rate's --
    shrunk_id_col lets the join handle both without renaming upstream."""

    batter_boxscore = pd.DataFrame([
        _lineup_slot_row('B1', gamepk='g1', batting_order=1),
        _lineup_slot_row('B2', gamepk='g1', batting_order=2),
    ])
    pbp = pd.DataFrame([
        _team_pbp_row('B1', gamepk='g1'),
        _team_pbp_row('B2', gamepk='g1'),
    ])
    batter_shrunk = pd.DataFrame([
        {'personId': 'B1', 'gamepk': 'g1', 'batter_shrunk_slg': 0.40},
        {'personId': 'B2', 'gamepk': 'g1', 'batter_shrunk_slg': 0.55},
    ])

    result = build_opposing_lineup_extremum(
        batter_boxscore, pbp, batter_shrunk,
        metric_col='batter_shrunk_slg', out_col='opp_team_best_batter_shrunk_slg',
        agg='max', shrunk_id_col='personId',
    )

    row = result.loc[(result['batter_team_id'] == 'T1') & (result['gamepk'] == 'g1')]
    assert row['opp_team_best_batter_shrunk_slg'].iloc[0] == pytest.approx(0.55)
