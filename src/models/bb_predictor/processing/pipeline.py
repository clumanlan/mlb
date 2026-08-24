from models.hit_predictor.processing.pipeline import (
    build_pbp_features,
    _create_batting_order,
    _add_estimated_team_pa_position,
)
from models.bb_predictor.processing.schema import WALKS


def build_pbp_features_walk(pbp, schedule, player_info):
    """hit_predictor's build_pbp_features computes every row/PA-level
    feature (pitch state, starting pitcher, PA number, times_through_order,
    game_date, handedness) target-agnostically, except its own is_hit
    column. Reuse it unmodified and add is_walk alongside it, rather than
    duplicating the pipeline — same pattern as k_predictor's
    build_pbp_features_strikeout."""

    pbp = build_pbp_features(pbp, schedule, player_info)
    return pbp.assign(is_walk=lambda x: x['play_result'].isin(WALKS).astype(int))


def create_pa_outcome_walk(pbp, batter_boxscore, game_info, schedule):
    """Mirrors hit_predictor's create_pa_outcome / k_predictor's
    create_pa_outcome_strikeout, swapping is_hit for is_walk as the PA-grain
    label. pbp must already carry is_walk (via build_pbp_features_walk).

    Scoped to REALIZED pitcher_role=='sp' only — same divergence from
    hit_predictor's own create_pa_outcome (which pools both sp and bullpen
    PAs) that k_predictor already made, and for the same reason: BB-prop's
    real-world target is a named starting pitcher's walk total, and a
    bullpen PA's actual pitcher isn't identifiable pre-game (pooled by
    team, see expected_role.py) — it isn't the subject of that prop at
    all."""

    batting_order = _create_batting_order(batter_boxscore)
    game_info = game_info[["gamepk", "game_season", "weather_condition", "weather_temp"]].drop_duplicates("gamepk")
    schedule = schedule[["gamepk", "game_date", "venue_id"]].drop_duplicates("gamepk")

    pbp = pbp[pbp['pitcher_role'] == 'sp']

    pa_outcome = pbp[[
        'gamepk', 'batter_team_name', 'batter_team_id', 'play_id', 'pitcher_id', 'pitcher_name',
        'batter_id', 'batter_name', 'is_walk', 'pitcher_throw_hand', 'batter_bat_side',
        'pitcher_role', 'pitcher_team_id', 'starting_pitcher_id', 'batter_pa_number',
    ]].drop_duplicates().reset_index(drop=True)

    pa_outcome = pa_outcome.merge(schedule, on="gamepk", how="left")
    pa_outcome = pa_outcome.merge(game_info, on="gamepk", how="left")
    pa_outcome = pa_outcome.merge(batting_order, on=["gamepk", "batter_id"], how="inner")
    pa_outcome = _add_estimated_team_pa_position(pa_outcome)

    return pa_outcome
