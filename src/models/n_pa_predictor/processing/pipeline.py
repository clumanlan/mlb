import pandas as pd

from models.hit_predictor.processing.pipeline import _create_batting_order
from models.n_pa_predictor.processing.schema import PA_OUTCOMES


def build_n_pa_label(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (gamepk, batter_id): realized plate-appearance count for
    that batter-game — the regression target for this model.

    Built independently from hit_predictor's own batter_pa_number counter
    (which runs on unfiltered pbp) by filtering to PA_OUTCOMES first, then
    counting distinct play_ids — so a non-PA-outcome row (e.g. a
    runner-movement event carrying its own play_id) can't inflate the count.
    Uses n_pa_predictor's own (locally extended) PA_OUTCOMES — see
    schema.py — not hit_predictor's PBP.PA_OUTCOMES directly, which is
    missing 'Field Error'/'Sac Fly Double Play'. This is a realized/
    post-game label, not a feature, so using it as-is (no point-in-time
    shift) is correct here.
    """

    pa_rows = pbp[pbp['play_result'].isin(PA_OUTCOMES)]
    pa_rows = pa_rows[['gamepk', 'play_id', 'batter_id']].drop_duplicates()

    return (
        pa_rows
        .groupby(['gamepk', 'batter_id'])
        .size()
        .rename('n_pa')
        .reset_index()
    )


def build_batter_game_frame(
    pbp: pd.DataFrame, batter_boxscore: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    """One row per starter per game: realized n_pa (label) + batting_order
    and game_date (join keys for feature assembly). Scoped to starters only
    (has a batting_order) — same inner-join filter hit_predictor's
    create_pa_outcome applies, since a sub/pinch-hitter's PA count is driven
    by substitution timing, a different and much noisier problem than a
    locked-in starter's.
    """

    label = build_n_pa_label(pbp)
    batting_order = _create_batting_order(batter_boxscore)

    frame = label.merge(batting_order, on=['gamepk', 'batter_id'], how='inner')
    frame = frame.merge(
        schedule[['gamepk', 'game_date']].drop_duplicates('gamepk'),
        on='gamepk', how='left',
    )

    return frame
