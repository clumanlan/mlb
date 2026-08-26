import pandas as pd

from models.hit_predictor.processing.features.rolling_stats import _pitcher_pa_outcome_per_game


def create_start_pa_outcome(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (personId, gamepk) starting-pitcher start:
    realized_batters_faced label — the regression target for this model.

    Reuses hit_predictor's _pitcher_pa_outcome_per_game (already the source
    k_predictor's count-distribution-check scripts use for the same number)
    rather than recomputing the PA count. Scoped to REALIZED
    pitcher_role == 'sp' only, by construction — same reasoning as
    short_outing_predictor's create_start_outcome: a bullpen pbp row isn't
    part of a "start" at all.
    """
    sp_pbp = pbp[pbp['pitcher_role'] == 'sp']

    per_start = _pitcher_pa_outcome_per_game(sp_pbp, entity_col='pitcher_id')
    per_start = per_start.rename(columns={
        'pitcher_id': 'personId', 'pa_total': 'realized_batters_faced',
    })

    return per_start[
        ['personId', 'gamepk', 'game_date', 'game_season', 'realized_batters_faced']
    ].reset_index(drop=True)
