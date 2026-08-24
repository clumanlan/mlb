import numpy as np
import pandas as pd

from models.hit_predictor.processing.features.rolling_stats import _rolling_sum, _rolling_prefix
from models.hit_predictor.processing.features.season_stats import _prefix_stat_cols

# The one genuinely new feature this model needs: a batter's own rolling
# average plate-appearance count per game, built on the corrected
# PA-outcome-based n_pa label (pipeline.build_n_pa_label), not
# batter_boxscore's flawed plate_appearances=ab+bb column (undercounts —
# misses HBP/sac fly/sac bunt). Same shift(1) point-in-time-safe engine as
# rolling_stats.py — see that module's docstring for window semantics.

KEY_COLS = ['batter_id', 'gamepk', 'game_date', 'game_season']


def build_batter_pa_rolling_stats(batter_game: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """One row per (batter_id, gamepk): that batter's rolling games_n and
    avg_n_pa_per_game, UP TO (not including) this game.

    batter_game must have one row per (batter_id, gamepk) with an n_pa
    column (e.g. pipeline.build_batter_game_frame's output).
    """

    df = batter_game[KEY_COLS + ['n_pa']].copy()
    df['games_n'] = 1

    rolled = _rolling_sum(df, entity_col='batter_id', cols=['n_pa', 'games_n'], window=window)

    rolled['avg_n_pa_per_game'] = rolled['n_pa'] / rolled['games_n'].replace(0, np.nan)
    # games_n itself is a real, meaningful 0 for a season-opening game (not
    # NaN) — only its ratio (avg_n_pa_per_game) should be undefined with no
    # prior games, same convention as game_context.build_pitcher_start_ip_this_season.
    rolled['games_n'] = rolled['games_n'].fillna(0)

    rolled = rolled[KEY_COLS + ['games_n', 'avg_n_pa_per_game']]

    return _prefix_stat_cols(rolled, prefix=_rolling_prefix('batter_n_pa', window), key_cols=KEY_COLS)
