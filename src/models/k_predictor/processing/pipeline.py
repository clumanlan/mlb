from models.hit_predictor.processing.pipeline import (
    build_pbp_features,
    _create_batting_order,
    _add_estimated_team_pa_position,
)
from models.k_predictor.processing.schema import STRIKEOUTS


def build_pbp_features_strikeout(pbp, schedule, player_info):
    """hit_predictor's build_pbp_features computes every row/PA-level
    feature (pitch state, starting pitcher, PA number, times_through_order,
    game_date, handedness) target-agnostically, except its own is_hit
    column. Reuse it unmodified and add is_strikeout alongside it, rather
    than duplicating the pipeline."""

    pbp = build_pbp_features(pbp, schedule, player_info)
    return pbp.assign(is_strikeout=lambda x: x['play_result'].isin(STRIKEOUTS).astype(int))


def create_pa_outcome_strikeout(pbp, batter_boxscore, game_info, schedule):
    """Mirrors hit_predictor's create_pa_outcome, swapping is_hit for
    is_strikeout as the PA-grain label. pbp must already carry is_strikeout
    (via build_pbp_features_strikeout)."""

    batting_order = _create_batting_order(batter_boxscore)
    game_info = game_info[["gamepk", "game_season", "weather_condition", "weather_temp"]].drop_duplicates("gamepk")
    schedule = schedule[["gamepk", "game_date", "venue_id"]].drop_duplicates("gamepk")

    pa_outcome = pbp[[
        'gamepk', 'batter_team_name', 'batter_team_id', 'play_id', 'pitcher_id', 'pitcher_name',
        'batter_id', 'batter_name', 'is_strikeout', 'pitcher_throw_hand', 'batter_bat_side',
        'pitcher_role', 'pitcher_team_id', 'starting_pitcher_id', 'batter_pa_number',
    ]].drop_duplicates().reset_index(drop=True)

    pa_outcome = pa_outcome.merge(schedule, on="gamepk", how="left")
    pa_outcome = pa_outcome.merge(game_info, on="gamepk", how="left")
    pa_outcome = pa_outcome.merge(batting_order, on=["gamepk", "batter_id"], how="inner")
    pa_outcome = _add_estimated_team_pa_position(pa_outcome)

    return pa_outcome
