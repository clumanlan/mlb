import numpy as np
import pandas as pd

from models.hit_predictor.processing.features.expected_role import (
    assign_expected_pitcher_role,
    assign_expected_pitcher_role_by_inning,
)


def _make_pa_outcome(**overrides):
    row = {
        'starting_pitcher_id': '10',
        'pitcher_team_id': 'T1',
        'game_season': 2024,
        'estimated_team_pa_position': 10,
        'batter_pa_number': 2,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _make_pitcher_start_depth_stats(**overrides):
    row = {
        'pitcher_id': '10',
        'game_season': 2024,
        'pitcher_last_season_start_avg_batters_faced_per_start': 20.0,
        'pitcher_last_season_start_n_starts': 25,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _empty_pitcher_start_depth_stats():
    return pd.DataFrame({
        'pitcher_id': pd.Series(dtype='object'),
        'game_season': pd.Series(dtype='int64'),
        'pitcher_last_season_start_avg_batters_faced_per_start': pd.Series(dtype='float64'),
        'pitcher_last_season_start_n_starts': pd.Series(dtype='float64'),
    })


def _make_league_avg_start_depth(**overrides):
    row = {'game_season': 2024, 'league_last_season_avg_batters_faced_per_start': 18.0}
    row.update(overrides)
    return pd.DataFrame([row])


def _empty_league_avg_start_depth():
    return pd.DataFrame({
        'game_season': pd.Series(dtype='int64'),
        'league_last_season_avg_batters_faced_per_start': pd.Series(dtype='float64'),
    })


def test_assign_expected_pitcher_role_labels_sp_when_position_within_starters_avg_depth():
    """The starter's own historical avg depth is 20; this batter's estimated
    team PA position is 10 — well within range, so expect 'sp'."""

    pa_outcome = _make_pa_outcome(estimated_team_pa_position=10)
    result = assign_expected_pitcher_role(
        pa_outcome, _make_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'sp'


def test_assign_expected_pitcher_role_labels_bullpen_when_position_exceeds_starters_avg_depth():
    """Same starter (avg depth 20), but this batter's estimated position
    (25) is past it — expect 'bullpen'."""

    pa_outcome = _make_pa_outcome(estimated_team_pa_position=25)
    result = assign_expected_pitcher_role(
        pa_outcome, _make_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'bullpen'


def test_assign_expected_pitcher_role_boundary_position_equal_to_avg_depth_is_sp():
    """Locks in <= (not <): a position exactly at the starter's average
    depth is still expected to be the starter, not the first bullpen PA."""

    pa_outcome = _make_pa_outcome(estimated_team_pa_position=20)
    result = assign_expected_pitcher_role(
        pa_outcome, _make_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'sp'


def test_assign_expected_pitcher_role_falls_back_to_league_average_when_pitcher_has_no_prior_season_stats():
    """A pitcher with no row in pitcher_start_depth_stats (rookie call-up,
    or no starts logged last season) falls back to the league-wide average
    (18 here). Position 20 is within the starter's own (nonexistent) depth
    but past the league average, so if the fallback is genuinely engaged —
    not just defaulted to 'sp' — this must resolve to 'bullpen'."""

    pa_outcome = _make_pa_outcome(estimated_team_pa_position=20)
    result = assign_expected_pitcher_role(
        pa_outcome, _empty_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'bullpen'


def test_assign_expected_pitcher_role_flags_league_fallback_usage_with_boolean_column():
    """expected_role_used_league_fallback distinguishes a coarse-fallback row
    from a row backed by the pitcher's own track record — True only when the
    pitcher has no individual depth stat."""

    fallback_row = assign_expected_pitcher_role(
        _make_pa_outcome(), _empty_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )
    own_data_row = assign_expected_pitcher_role(
        _make_pa_outcome(), _make_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert fallback_row.loc[0, 'expected_role_used_league_fallback'] == True
    assert own_data_row.loc[0, 'expected_role_used_league_fallback'] == False


def test_assign_expected_pitcher_role_defaults_to_sp_when_no_depth_data_exists_at_all():
    """With neither an individual nor a league depth stat available (e.g.
    the very first ingested season), the ultimate fallback is 'always sp' —
    never gate a PA to bullpen purely for lack of information."""

    pa_outcome = _make_pa_outcome(estimated_team_pa_position=999)
    result = assign_expected_pitcher_role(
        pa_outcome, _empty_pitcher_start_depth_stats(), _empty_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'sp'


def test_assign_expected_pitcher_role_times_through_order_is_capped_at_three_for_expected_sp():
    pa_outcome = _make_pa_outcome(estimated_team_pa_position=10, batter_pa_number=5)
    result = assign_expected_pitcher_role(
        pa_outcome, _make_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'sp'
    assert result.loc[0, 'expected_times_through_order'] == 3


def test_assign_expected_pitcher_role_times_through_order_is_nan_for_expected_bullpen():
    pa_outcome = _make_pa_outcome(estimated_team_pa_position=25, batter_pa_number=5)
    result = assign_expected_pitcher_role(
        pa_outcome, _make_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'bullpen'
    assert np.isnan(result.loc[0, 'expected_times_through_order'])


def test_assign_expected_pitcher_role_key_id_is_starting_pitcher_id_for_expected_sp():
    pa_outcome = _make_pa_outcome(estimated_team_pa_position=10, starting_pitcher_id='10', pitcher_team_id='T1')
    result = assign_expected_pitcher_role(
        pa_outcome, _make_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_key_id'] == '10'


def test_assign_expected_pitcher_role_key_id_is_pitcher_team_id_for_expected_bullpen():
    pa_outcome = _make_pa_outcome(estimated_team_pa_position=25, starting_pitcher_id='10', pitcher_team_id='T1')
    result = assign_expected_pitcher_role(
        pa_outcome, _make_pitcher_start_depth_stats(), _make_league_avg_start_depth()
    )

    assert result.loc[0, 'expected_pitcher_key_id'] == 'T1'


# --------------------------- INNING-BASED REPLACEMENT (Epic E / v4) --------------------------- #
# Same job as assign_expected_pitcher_role (pre-game-knowable sp/bullpen gate for a PA),
# but comparing an estimated INNING (from estimated_team_pa_position via a fixed
# avg-team-PAs-per-inning approximation) against the starter's own expected_start_innings
# (game_context.build_expected_start_innings — already a last-season/league/this-season
# shrinkage blend) instead of comparing raw PA position against a batters-faced average.

def _make_pa_outcome_inning(**overrides):
    row = {
        'starting_pitcher_id': '10',
        'pitcher_team_id': 'T1',
        'gamepk': 'g1',
        'estimated_team_pa_position': 10,
        'batter_pa_number': 2,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _make_expected_start_innings(**overrides):
    row = {'personId': '10', 'gamepk': 'g1', 'expected_start_innings': 5.0}
    row.update(overrides)
    return pd.DataFrame([row])


def _empty_expected_start_innings():
    return pd.DataFrame({
        'personId': pd.Series(dtype='object'),
        'gamepk': pd.Series(dtype='object'),
        'expected_start_innings': pd.Series(dtype='float64'),
    })


def test_assign_expected_pitcher_role_by_inning_labels_sp_when_estimated_inning_within_expected_depth():
    """PA position 10 -> estimated_inning = ceil(10/4.3) = 3, well within
    this starter's expected_start_innings of 5.0 -> 'sp'."""

    result = assign_expected_pitcher_role_by_inning(
        _make_pa_outcome_inning(estimated_team_pa_position=10), _make_expected_start_innings()
    )

    assert result.loc[0, 'estimated_inning'] == 3
    assert result.loc[0, 'expected_pitcher_role'] == 'sp'


def test_assign_expected_pitcher_role_by_inning_labels_bullpen_when_estimated_inning_exceeds_expected_depth():
    """PA position 25 -> estimated_inning = ceil(25/4.3) = 6, past this
    starter's expected_start_innings of 5.0 -> 'bullpen'."""

    result = assign_expected_pitcher_role_by_inning(
        _make_pa_outcome_inning(estimated_team_pa_position=25), _make_expected_start_innings()
    )

    assert result.loc[0, 'estimated_inning'] == 6
    assert result.loc[0, 'expected_pitcher_role'] == 'bullpen'


def test_assign_expected_pitcher_role_by_inning_boundary_inning_equal_to_expected_depth_is_sp():
    """Locks in <= (not <): PA position 19 -> estimated_inning =
    ceil(19/4.3) = 5, exactly at expected_start_innings (5.0) -> still 'sp',
    not the first bullpen inning."""

    result = assign_expected_pitcher_role_by_inning(
        _make_pa_outcome_inning(estimated_team_pa_position=19), _make_expected_start_innings()
    )

    assert result.loc[0, 'estimated_inning'] == 5
    assert result.loc[0, 'expected_pitcher_role'] == 'sp'


def test_assign_expected_pitcher_role_by_inning_defaults_to_sp_when_no_estimate_exists():
    """No matching row in expected_start_innings at all (shouldn't happen
    given build_expected_start_innings' own league-wide fallback, but not
    assumed here) -> never gate to bullpen purely for lack of information."""

    result = assign_expected_pitcher_role_by_inning(
        _make_pa_outcome_inning(estimated_team_pa_position=999), _empty_expected_start_innings()
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'sp'


def test_assign_expected_pitcher_role_by_inning_times_through_order_is_capped_at_three_for_expected_sp():
    result = assign_expected_pitcher_role_by_inning(
        _make_pa_outcome_inning(estimated_team_pa_position=10, batter_pa_number=5),
        _make_expected_start_innings(),
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'sp'
    assert result.loc[0, 'expected_times_through_order'] == 3


def test_assign_expected_pitcher_role_by_inning_times_through_order_is_nan_for_expected_bullpen():
    result = assign_expected_pitcher_role_by_inning(
        _make_pa_outcome_inning(estimated_team_pa_position=25, batter_pa_number=5),
        _make_expected_start_innings(),
    )

    assert result.loc[0, 'expected_pitcher_role'] == 'bullpen'
    assert np.isnan(result.loc[0, 'expected_times_through_order'])


def test_assign_expected_pitcher_role_by_inning_key_id_is_starting_pitcher_id_for_expected_sp():
    result = assign_expected_pitcher_role_by_inning(
        _make_pa_outcome_inning(estimated_team_pa_position=10, starting_pitcher_id='10', pitcher_team_id='T1'),
        _make_expected_start_innings(),
    )

    assert result.loc[0, 'expected_pitcher_key_id'] == '10'


def test_assign_expected_pitcher_role_by_inning_key_id_is_pitcher_team_id_for_expected_bullpen():
    result = assign_expected_pitcher_role_by_inning(
        _make_pa_outcome_inning(estimated_team_pa_position=25, starting_pitcher_id='10', pitcher_team_id='T1'),
        _make_expected_start_innings(),
    )

    assert result.loc[0, 'expected_pitcher_key_id'] == 'T1'
