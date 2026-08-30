from models.hit_predictor.processing.pipeline import _create_batting_order


def build_opposing_lineup_extremum(
    batter_boxscore, pbp, batter_shrunk_df, metric_col: str, out_col: str,
    agg: str = 'min', shrunk_id_col: str = 'batter_id',
):
    """One row per (batter_team_id, gamepk): the min/max of a per-batter
    shrunk metric across that game's STARTING lineup only (same
    starters-only pooling as build_team_batter_strikeout_rolling_feats) --
    but a CROSS-SECTIONAL extremum over the current lineup, not a rolling
    extremum over a team's own past games (contrast
    rolling_stats.build_team_strikeout_volatility's MAX, which rolls a
    team's own history forward).

    Used both directions: agg='min' surfaces the toughest out in the lineup
    (a single elite-contact/low-K batter caps a pitcher's total-K ceiling
    regardless of how weak the rest of the lineup is), agg='max' surfaces
    the best individual OBP/SLG batter.

    A game with no identifiable starting lineup (no batting_order rows)
    produces no row at all for that game -- same "no lineup -> no output"
    behavior as build_team_batter_strikeout_rolling_feats, not a NaN-filled
    row or a crash.

    shrunk_id_col: the entity-id column name in batter_shrunk_df.
    build_batter_shrunk_k_rate's output is keyed on batter_id (pbp-based);
    build_batter_shrunk_obp_slg's is keyed on personId (box-score based) --
    _create_batting_order always renames to batter_id, so this lets either
    shrunk table join without the caller renaming first.

    batter_shrunk_df is a PER-GAME rolling table (one row per batter per
    game), not a static per-batter value -- the join is scoped to
    (batter_id, gamepk) so each lineup slot only ever sees THAT game's
    shrunk value, never fanning out against every other game the same
    batter appears in elsewhere in a multi-season dataset.
    """
    starters = _create_batting_order(batter_boxscore)[['gamepk', 'batter_id']]
    team_lookup = pbp[['batter_id', 'gamepk', 'batter_team_id']].drop_duplicates()

    lineup = starters.merge(team_lookup, on=['batter_id', 'gamepk'], how='left')

    shrunk = batter_shrunk_df[[shrunk_id_col, 'gamepk', metric_col]].rename(columns={shrunk_id_col: 'batter_id'})
    lineup = lineup.merge(shrunk, on=['batter_id', 'gamepk'], how='inner')

    return (
        lineup.groupby(['batter_team_id', 'gamepk'])[metric_col]
        .agg(agg)
        .reset_index()
        .rename(columns={metric_col: out_col})
    )


def build_batter_shrunk_k_rate(
    batter_rolling, batter_season, window: str | int = 'season', k: float = 50.0,
):
    """Blend last season's K rate toward this season's own emerging rolling K
    rate as in-season PA accumulate — shrinkage_weight = pa_total /
    (pa_total + k), 0 at a season opener (trust last season fully), rising
    toward 1 as this season's own PA sample grows. Same recipe as
    pitcher_workload.build_pitcher_shrunk_whip, PA-weighted instead of
    games_n-weighted (a batter's own PA count is the natural sample-size
    denominator here, matching the k=50 precedent already used for batter
    shrinkage in experiments/count_distribution_check/run_naive_batter_uncertainty.py).

    Baseline falls back to this season's own rolling K rate when there's no
    last-season row (rookie/unseen batter) — best available signal rather
    than NaN.

    batter_rolling: rolling_stats.build_pbp_batter_rolling_feats output.
    batter_season: season_stats.build_pbp_batter_feats output (already
        shifted to last season).
    """
    prefix = 'batter_roll_season_' if window == 'season' else f'batter_roll_last{window}g_'
    rolling_col = f'{prefix}pa_strikeout_rate'
    pa_col = f'{prefix}pa_total'

    df = batter_rolling.merge(
        batter_season[['batter_id', 'game_season', 'batter_last_season_pa_strikeout_rate']],
        on=['batter_id', 'game_season'], how='left',
    )

    pa_total = df[pa_col].fillna(0)
    weight = pa_total / (pa_total + k)
    baseline = df['batter_last_season_pa_strikeout_rate'].fillna(df[rolling_col])
    rolling_safe = df[rolling_col].fillna(baseline)

    df['batter_shrunk_k_rate_weight'] = weight
    df['batter_shrunk_k_rate'] = (1 - weight) * baseline + weight * rolling_safe

    return df


def build_batter_shrunk_obp_slg(
    batter_rolling, batter_season, window: str | int = 'season', k: float = 50.0,
):
    """Same shrinkage recipe as build_batter_shrunk_k_rate, applied to OBP and
    SLG independently (a batter's shrunk OBP and shrunk SLG are computed and
    weighted separately, not combined into one metric). PA-weighted via
    plate_appearances — the box-score-rolling sample-size column, since this
    feeds off rolling_stats.build_batter_rolling_stats/season_stats.build_batter_stats
    (box-score based) rather than the pbp-based pa_total
    build_batter_shrunk_k_rate uses. Both metrics share one weight, computed
    from the same plate_appearances denominator.

    batter_rolling: rolling_stats.build_batter_rolling_stats output (keyed on
        personId — box-score based, unlike build_pbp_batter_rolling_feats's
        batter_id).
    batter_season: season_stats.build_batter_stats output (already shifted to
        last season, also keyed on personId).
    """
    prefix = 'batter_roll_season_' if window == 'season' else f'batter_roll_last{window}g_'
    pa_col = f'{prefix}plate_appearances'

    df = batter_rolling.merge(
        batter_season[['personId', 'game_season', 'batter_last_season_obp', 'batter_last_season_slg']],
        on=['personId', 'game_season'], how='left',
    )

    plate_appearances = df[pa_col].fillna(0)
    weight = plate_appearances / (plate_appearances + k)
    df['batter_shrunk_obp_weight'] = weight

    for metric in ('obp', 'slg'):
        rolling_col = f'{prefix}{metric}'
        last_season_col = f'batter_last_season_{metric}'
        baseline = df[last_season_col].fillna(df[rolling_col])
        rolling_safe = df[rolling_col].fillna(baseline)
        df[f'batter_shrunk_{metric}'] = (1 - weight) * baseline + weight * rolling_safe

    return df
