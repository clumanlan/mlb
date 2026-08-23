import numpy as np
import pandas as pd

from models.hit_predictor.baseline.statistical.shrinkage import (
    add_matchup_shrinkage_component,
    add_matchup_shrinkage_component_blended,
)


def _pa_df(rows):
    return pd.DataFrame(rows)


def _pitcher_row(pitcher_id, role, gamepk, game_date, batter_id, is_hit,
                  last_season_ba=0.250, pitcher_last_season_pa_hit_rate=0.157):
    return {
        "batter_id": batter_id, "game_season": 2024, "game_date": game_date,
        "gamepk": gamepk, "play_id": 1, "is_hit": is_hit,
        "last_season_ba": last_season_ba,
        "realized_pitcher_key_id": pitcher_id, "pitcher_role": role,
        "pitcher_last_season_pa_hit_rate": pitcher_last_season_pa_hit_rate,
    }


def test_matches_static_matchup_version_on_a_pitchers_very_first_pa():
    # cum_pa_before == 0 for the pitcher -> pitcher_blended == his last-
    # season prior exactly -> identical to the static (non-blended) version.
    # is_hit=1 (not 0) so the season league average isn't degenerate zero.
    row = _pitcher_row("SNELL", "sp", "1", "2024-04-08", "A", 1)
    df = _pa_df([row])

    static = add_matchup_shrinkage_component(df.copy(), k=100.0)
    blended = add_matchup_shrinkage_component_blended(df.copy(), k=100.0)

    assert np.isclose(
        blended["matchup_shrinkage_blended_pred"].iloc[0],
        static["matchup_shrinkage_pred"].iloc[0],
    )


def test_pitchers_own_poor_in_season_results_raise_later_predictions():
    # Snell's last-season prior is "dominant" (0.157), but he allows hits in
    # his first two starts this season. A later start against a fresh
    # batter (same last_season_ba, no history of their own) must predict
    # HIGHER under the blended version than the static version, because the
    # blended pitcher rate has moved up toward his actual in-season form.
    rows = [
        _pitcher_row("SNELL", "sp", "1", "2024-04-08", "early_batter_1", 1),
        _pitcher_row("SNELL", "sp", "1", "2024-04-08", "early_batter_2", 1),
        _pitcher_row("SNELL", "sp", "2", "2024-04-14", "early_batter_3", 1),
        _pitcher_row("SNELL", "sp", "2", "2024-04-14", "early_batter_4", 1),
        _pitcher_row("SNELL", "sp", "3", "2024-04-20", "fresh_batter", 0),
    ]
    df = _pa_df(rows)

    static = add_matchup_shrinkage_component(df.copy(), k=100.0)
    blended = add_matchup_shrinkage_component_blended(df.copy(), k=100.0)

    static_pred = static.loc[df["batter_id"] == "fresh_batter", "matchup_shrinkage_pred"].iloc[0]
    blended_pred = blended.loc[df["batter_id"] == "fresh_batter", "matchup_shrinkage_blended_pred"].iloc[0]

    assert blended_pred > static_pred


def test_does_not_leak_across_different_pitchers():
    # Pitcher X allows nothing but hits early; pitcher Y (unrelated) faces a
    # fresh batter later -- Y's blended rate must be untouched by X's
    # results (still equal to Y's own last-season prior, cold start).
    rows = [
        _pitcher_row("X", "sp", "1", "2024-04-08", "b1", 1),
        _pitcher_row("X", "sp", "1", "2024-04-08", "b2", 1),
        _pitcher_row("Y", "sp", "2", "2024-04-14", "b3", 0, pitcher_last_season_pa_hit_rate=0.200),
    ]
    df = _pa_df(rows)

    blended = add_matchup_shrinkage_component_blended(df, k=100.0)
    static = add_matchup_shrinkage_component(df, k=100.0)

    y_blended = blended.loc[df["batter_id"] == "b3", "matchup_shrinkage_blended_pred"].iloc[0]
    y_static = static.loc[df["batter_id"] == "b3", "matchup_shrinkage_pred"].iloc[0]
    assert np.isclose(y_blended, y_static)


def test_output_is_probability_between_0_and_1():
    rows = [
        _pitcher_row("SNELL", "sp", "1", "2024-04-08", "A", 1),
        _pitcher_row("SNELL", "sp", "2", "2024-04-14", "A", 0),
        _pitcher_row("SNELL", "sp", "3", "2024-04-20", "B", 0,
                      last_season_ba=np.nan, pitcher_last_season_pa_hit_rate=np.nan),
    ]
    df = _pa_df(rows)

    out = add_matchup_shrinkage_component_blended(df, k=100.0)

    assert out["matchup_shrinkage_blended_pred"].between(0, 1).all()
    assert out["matchup_shrinkage_blended_pred"].isna().sum() == 0
