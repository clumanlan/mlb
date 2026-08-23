import numpy as np
import pandas as pd

from models.hit_predictor.baseline.statistical.shrinkage import add_shrinkage_component


def _pa_df(rows):
    """Build a minimal PA-level frame: batter_id, game_season, game_date,
    gamepk, play_id, is_hit, last_season_ba.
    """
    return pd.DataFrame(rows)


def test_uses_last_season_ba_as_shrink_target_when_present():
    # Two batters, both with zero PAs so far this season (cold start), one
    # game each. Batter A has a last_season_ba of 0.400, batter B of 0.100.
    # With no observed PAs yet, the shrinkage prediction should sit at each
    # batter's own last_season_ba, not converge to a shared value.
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 0, "last_season_ba": 0.400},
        {"batter_id": "B", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "2", "play_id": 1, "is_hit": 1, "last_season_ba": 0.100},
    ])

    out = add_shrinkage_component(df, k=100.0)

    assert np.isclose(out.loc[out["batter_id"] == "A", "shrinkage_pred"].iloc[0], 0.400)
    assert np.isclose(out.loc[out["batter_id"] == "B", "shrinkage_pred"].iloc[0], 0.100)


def test_falls_back_to_league_avg_when_last_season_ba_missing():
    # Batter A has a last_season_ba; batter B (rookie) has none (NaN). Both
    # have zero PAs so far this season. Batter B's cold-start prediction
    # should equal the season's league-average hit rate computed from the
    # rest of the frame, not a hardcoded constant.
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.400},
        {"batter_id": "B", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "2", "play_id": 1, "is_hit": 0, "last_season_ba": np.nan},
    ])

    out = add_shrinkage_component(df, k=100.0)

    league_avg = df["is_hit"].mean()
    assert np.isclose(
        out.loc[out["batter_id"] == "B", "shrinkage_pred"].iloc[0], league_avg
    )


def test_first_pa_of_season_equals_shrink_target_exactly():
    # A batter's very first PA of the season has zero observed PAs behind
    # it, so the cascading formula (hits_before + k*target) / (pa_before + k)
    # collapses to exactly `target` (last_season_ba here) regardless of k.
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.275},
    ])

    out = add_shrinkage_component(df, k=50.0)

    assert np.isclose(out["shrinkage_pred"].iloc[0], 0.275)


def test_excludes_current_row_outcome_from_cumulative_stats():
    # Same batter, three PAs across three games in season order, all hits.
    # The prediction for PA #2 must only reflect PA #1's outcome (not its
    # own), and PA #3's prediction must only reflect PAs #1-#2 -- i.e. no
    # lookahead leakage from a PA's own realized outcome into its own
    # predicted probability.
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.300},
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-02",
         "gamepk": "2", "play_id": 1, "is_hit": 1, "last_season_ba": 0.300},
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-03",
         "gamepk": "3", "play_id": 1, "is_hit": 1, "last_season_ba": 0.300},
    ])

    out = add_shrinkage_component(df, k=100.0).sort_values("game_date")
    preds = out["shrinkage_pred"].to_numpy()

    # PA1: 0 hits/0 PA before -> exactly last_season_ba.
    assert np.isclose(preds[0], 0.300)
    # PA2: 1 hit/1 PA before -> (1 + 100*0.300) / (1 + 100), strictly above
    # last_season_ba since the one observed PA so far was a hit.
    expected_pa2 = (1 + 100.0 * 0.300) / (1 + 100.0)
    assert np.isclose(preds[1], expected_pa2)
    assert preds[1] > 0.300
    # PA3 must differ from PA2 -- if the current row's own outcome leaked
    # into its own prediction, PA2 and PA3 would be computed identically
    # from PA1's single observed hit.
    expected_pa3 = (2 + 100.0 * 0.300) / (2 + 100.0)
    assert np.isclose(preds[2], expected_pa3)
    assert preds[2] != preds[1]


def test_output_is_probability_between_0_and_1():
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.400},
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-02",
         "gamepk": "2", "play_id": 1, "is_hit": 0, "last_season_ba": 0.400},
        {"batter_id": "B", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "3", "play_id": 1, "is_hit": 0, "last_season_ba": np.nan},
    ])

    out = add_shrinkage_component(df, k=100.0)

    assert out["shrinkage_pred"].between(0, 1).all()
    assert out["shrinkage_pred"].isna().sum() == 0
