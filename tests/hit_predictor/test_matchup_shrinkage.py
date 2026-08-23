import numpy as np
import pandas as pd

from models.hit_predictor.baseline.statistical.shrinkage import (
    add_matchup_shrinkage_component,
    add_shrinkage_component,
)


def _pa_df(rows):
    return pd.DataFrame(rows)


def test_matches_batter_only_shrinkage_when_pitcher_data_missing():
    # No pitcher_last_season_pa_hit_rate at all -> log5 target collapses to
    # last_season_ba alone (b*L/L = b), same as add_shrinkage_component.
    rows = [
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.300,
         "pitcher_last_season_pa_hit_rate": np.nan},
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-02",
         "gamepk": "2", "play_id": 1, "is_hit": 0, "last_season_ba": 0.300,
         "pitcher_last_season_pa_hit_rate": np.nan},
    ]
    batter_only = add_shrinkage_component(_pa_df(rows), k=100.0)
    matchup = add_matchup_shrinkage_component(_pa_df(rows), k=100.0)

    batter_only = batter_only.sort_values("game_date")
    matchup = matchup.sort_values("game_date")

    np.testing.assert_allclose(
        matchup["matchup_shrinkage_pred"].to_numpy(),
        batter_only["shrinkage_pred"].to_numpy(),
    )


def test_matchup_target_uses_log5_combination_when_both_present():
    # First PA of the season -> matchup_shrinkage_pred equals shrink_target
    # exactly, same cold-start property as add_shrinkage_component. League
    # avg here is derived from the single row itself (is_hit=1 -> mean=1.0),
    # so hand-compute with L=1.0: target = b*p/L = 0.300*0.150/1.0 = 0.045.
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.300,
         "pitcher_last_season_pa_hit_rate": 0.150},
    ])

    out = add_matchup_shrinkage_component(df, k=100.0)

    assert np.isclose(out["matchup_shrinkage_pred"].iloc[0], 0.300 * 0.150 / 1.0)


def test_dominant_pitcher_lowers_prediction_relative_to_weak_pitcher():
    # Same batter, same league avg, cold start (no PAs yet) -- only the
    # pitcher rate differs. Dominant (low hit rate allowed) pitcher must
    # produce a strictly lower prediction than a weak (high hit rate
    # allowed) pitcher.
    base_row = {
        # is_hit=1 on both rows (not 0) so the season league average used as
        # the log5 denominator isn't degenerate zero.
        "batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
        "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.280,
    }
    df = _pa_df([
        {**base_row, "batter_id": "A", "gamepk": "1", "pitcher_last_season_pa_hit_rate": 0.150},
        {**base_row, "batter_id": "B", "gamepk": "2", "pitcher_last_season_pa_hit_rate": 0.320},
    ])

    out = add_matchup_shrinkage_component(df, k=100.0)

    dominant_pred = out.loc[out["batter_id"] == "A", "matchup_shrinkage_pred"].iloc[0]
    weak_pred = out.loc[out["batter_id"] == "B", "matchup_shrinkage_pred"].iloc[0]
    assert dominant_pred < weak_pred


def test_matchup_target_clipped_to_valid_probability_range():
    # b=0.9, p=0.9 with a low league avg (0.1 from the second row) would
    # give an unclipped log5 target of 0.9*0.9/0.1 = 8.1 -- must clip to <=1.
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "B", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "2", "play_id": 1, "is_hit": 0, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "C", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "3", "play_id": 1, "is_hit": 0, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "D", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "4", "play_id": 1, "is_hit": 0, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "E", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "5", "play_id": 1, "is_hit": 0, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "F", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "6", "play_id": 1, "is_hit": 0, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "G", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "7", "play_id": 1, "is_hit": 0, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "H", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "8", "play_id": 1, "is_hit": 0, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "I", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "9", "play_id": 1, "is_hit": 0, "last_season_ba": 0.900,
         "pitcher_last_season_pa_hit_rate": 0.900},
        {"batter_id": "J", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "10", "play_id": 1, "is_hit": 1, "last_season_ba": 0.100,
         "pitcher_last_season_pa_hit_rate": 0.100},
    ])

    out = add_matchup_shrinkage_component(df, k=100.0)

    assert out["matchup_shrinkage_pred"].between(0, 1).all()


def test_falls_back_symmetrically_when_only_batter_missing():
    # last_season_ba missing (rookie), pitcher rate present -> target should
    # equal the pitcher rate alone (b filled with L, so b*p/L = L*p/L = p).
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 0, "last_season_ba": np.nan,
         "pitcher_last_season_pa_hit_rate": 0.200},
        {"batter_id": "B", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "2", "play_id": 1, "is_hit": 1, "last_season_ba": 0.250,
         "pitcher_last_season_pa_hit_rate": np.nan},
    ])

    out = add_matchup_shrinkage_component(df, k=100.0)

    row_a = out[out["batter_id"] == "A"].iloc[0]
    assert np.isclose(row_a["matchup_shrinkage_pred"], row_a["pitcher_last_season_pa_hit_rate"])


def test_output_is_probability_between_0_and_1():
    df = _pa_df([
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "1", "play_id": 1, "is_hit": 1, "last_season_ba": 0.400,
         "pitcher_last_season_pa_hit_rate": 0.150},
        {"batter_id": "A", "game_season": 2024, "game_date": "2024-04-02",
         "gamepk": "2", "play_id": 1, "is_hit": 0, "last_season_ba": 0.400,
         "pitcher_last_season_pa_hit_rate": 0.320},
        {"batter_id": "B", "game_season": 2024, "game_date": "2024-04-01",
         "gamepk": "3", "play_id": 1, "is_hit": 0, "last_season_ba": np.nan,
         "pitcher_last_season_pa_hit_rate": np.nan},
    ])

    out = add_matchup_shrinkage_component(df, k=100.0)

    assert out["matchup_shrinkage_pred"].between(0, 1).all()
    assert out["matchup_shrinkage_pred"].isna().sum() == 0
