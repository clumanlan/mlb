import numpy as np
import pandas as pd

from models.hit_predictor.processing.pipeline import (
    build_pbp_features,
    _create_batting_order,
    _add_estimated_team_pa_position,
)
from models.k_predictor.processing.schema import STRIKEOUTS


def _add_platoon_matchup(df):
    """same_hand/opposite_hand/switch_hitter, mirroring hit_predictor's
    slice_diagnostic.py derivation -- makes the batter_bat_side ==
    pitcher_throw_hand equality an explicit categorical feature instead of
    leaving it for a shallow tree to rediscover across two separately
    ordinal/one-hot-encoded columns."""
    is_switch = (df['batter_bat_side'] == 'S').fillna(False)
    is_same_hand = (df['batter_bat_side'] == df['pitcher_throw_hand']).fillna(False)

    platoon_matchup = pd.Series('opposite_hand', index=df.index)
    platoon_matchup[is_same_hand] = 'same_hand'
    platoon_matchup[is_switch] = 'switch_hitter'
    platoon_matchup[df['batter_bat_side'].isna() | df['pitcher_throw_hand'].isna()] = np.nan

    return df.assign(platoon_matchup=platoon_matchup)


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
    (via build_pbp_features_strikeout).

    Scoped to REALIZED pitcher_role=='sp' only — a deliberate divergence
    from hit_predictor's own create_pa_outcome, which pools both sp and
    bullpen PAs. K-prop's real-world target is a named starting pitcher's
    strikeout total; a bullpen PA's actual pitcher isn't identifiable
    pre-game (pooled by team, see expected_role.py) and isn't the subject of
    that prop at all. Same convention as season_stats.py's
    _create_pitcher_start_depth_stats sp_pbp filter — REALIZED role, not
    expected_pitcher_role, since this defines "was the true pitcher a
    starter," not a pre-game-knowable gate (that's what expected_pitcher_role,
    assigned downstream, is for)."""

    batting_order = _create_batting_order(batter_boxscore)
    game_info = game_info[["gamepk", "game_season", "weather_condition", "weather_temp"]].drop_duplicates("gamepk")
    schedule = schedule[["gamepk", "game_date", "venue_id"]].drop_duplicates("gamepk")

    pbp = pbp[pbp['pitcher_role'] == 'sp']

    pa_outcome = pbp[[
        'gamepk', 'batter_team_name', 'batter_team_id', 'play_id', 'pitcher_id', 'pitcher_name',
        'batter_id', 'batter_name', 'is_strikeout', 'pitcher_throw_hand', 'batter_bat_side',
        'pitcher_role', 'pitcher_team_id', 'starting_pitcher_id', 'batter_pa_number',
    ]].drop_duplicates().reset_index(drop=True)

    pa_outcome = _add_platoon_matchup(pa_outcome)

    pa_outcome = pa_outcome.merge(schedule, on="gamepk", how="left")
    pa_outcome = pa_outcome.merge(game_info, on="gamepk", how="left")
    pa_outcome = pa_outcome.merge(batting_order, on=["gamepk", "batter_id"], how="inner")
    pa_outcome = _add_estimated_team_pa_position(pa_outcome)

    return pa_outcome
