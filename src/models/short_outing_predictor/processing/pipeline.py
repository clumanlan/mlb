import pandas as pd

from models.hit_predictor.processing.features.season_stats import _pitcher_role_lookup
from models.short_outing_predictor.processing.schema import SHORT_OUTING_IP_THRESHOLD


def create_start_outcome(pitcher_boxscore: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (personId, gamepk) starting-pitcher start: is_short_outing
    label (realized ip <= SHORT_OUTING_IP_THRESHOLD) — the grain this model
    predicts at, unlike hit_predictor/k_predictor/bb_predictor/n_pa_predictor's
    PA or batter-game grain. pitcher_boxscore must already be decimal-IP
    (hit_predictor's process_pitcher_boxscore) and pbp must already carry
    pitcher_role (hit_predictor's build_pbp_features) — same role-tagging
    join as season_stats.py's _create_pitcher_start_ip_stats /
    game_context.py's build_pitcher_start_ip_this_season, duplicated here
    rather than shared, same convention those two already use.

    Scoped to REALIZED pitcher_role == 'sp' only, by construction rather
    than choice: a bullpen boxscore row isn't a "start" at all, so it's out
    of scope for a short-outing label regardless of any population-scoping
    decision (contrast k_predictor's/bb_predictor's own sp-only scoping,
    a deliberate choice about a PA-grain population that could have gone
    either way).
    """
    role_lookup = _pitcher_role_lookup(pbp)[['gamepk', 'pitcher_id', 'pitcher_role']].rename(
        columns={'pitcher_id': 'personId'}
    )
    tagged = pitcher_boxscore.assign(
        personId=lambda x: x['personId'].astype(str), gamepk=lambda x: x['gamepk'].astype(str),
    ).merge(
        role_lookup.assign(
            personId=lambda x: x['personId'].astype(str), gamepk=lambda x: x['gamepk'].astype(str),
        ),
        on=['gamepk', 'personId'], how='left',
    )

    starts = tagged[tagged['pitcher_role'] == 'sp'].copy()
    starts['is_short_outing'] = (starts['ip'] <= SHORT_OUTING_IP_THRESHOLD).astype(int)

    return starts[
        ['personId', 'gamepk', 'game_date', 'game_season', 'ip', 'is_short_outing']
    ].reset_index(drop=True)
