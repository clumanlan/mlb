"""
Statistical shrinkage baseline — cascading empirical-Bayes estimator.

For each PA, predict:
    (cum_hits_before + k * shrink_target) / (cum_pa_before + k)

where cum_hits_before/cum_pa_before are the batter's hits/PAs strictly
earlier in the same season (no lookahead), and shrink_target is
last_season_ba when available, falling back to that season's
league-average hit rate for batters with no prior-season data (rookies).

Distinct from baseline/rules/run_baseline.py's fixed-weight 3-component
blend (prev_season/rolling/order_slot) -- this is a single cascading
formula, evaluated at both PA and game grain via
run_baseline.py::run_pa_vs_game_grain_check.
"""
import numpy as np
import pandas as pd


def _cascading_shrinkage(
    df: pd.DataFrame, shrink_target: pd.Series, k: float,
    entity_cols: tuple = ('batter_id',),
) -> tuple:
    """Point-in-time-safe empirical-Bayes cascade shared by every shrinkage
    variant in this module: (cum_hits_before + k*shrink_target) /
    (cum_pa_before + k). Returns (sorted_df, prediction_series) so callers
    attach the prediction under whichever column name is theirs.

    entity_cols identifies whose running count this is -- defaults to the
    batter (('batter_id',)), but the pitcher-side blend below reuses this
    exact cascade grouped by pitcher identity instead
    (('realized_pitcher_key_id', 'pitcher_role')), always combined with
    game_season.
    """
    entity_cols = list(entity_cols)
    sort_cols = entity_cols + ['game_season', 'game_date', 'gamepk', 'play_id']
    df = df.sort_values(sort_cols)
    shrink_target = shrink_target.reindex(df.index)

    grp = df.groupby(entity_cols + ['game_season'])
    cum_hits_before = grp['is_hit'].cumsum() - df['is_hit']
    cum_pa_before = grp.cumcount()

    pred = (cum_hits_before + k * shrink_target) / (cum_pa_before + k)
    return df, pred


def add_shrinkage_component(df: pd.DataFrame, k: float = 100.0) -> pd.DataFrame:
    df = df.copy()
    season_league_avg = df.groupby('game_season')['is_hit'].transform('mean')
    shrink_target = df['last_season_ba'].fillna(season_league_avg)

    df, pred = _cascading_shrinkage(df, shrink_target, k)
    df['shrinkage_pred'] = pred

    return df


def add_matchup_shrinkage_component(df: pd.DataFrame, k: float = 100.0) -> pd.DataFrame:
    """
    Same cascading empirical-Bayes formula as add_shrinkage_component, but
    the prior (shrink_target) is a log5-combined batter x pitcher matchup
    rate instead of batter-only last_season_ba -- same simple multiplicative
    log5 convention already used elsewhere in this repo
    (tier1_feats.build_log5_matchup_features: batter_rate * pitcher_rate /
    league_rate), not the full odds-normalized log5 formula, for consistency.

    Requires df to carry last_season_ba, pitcher_last_season_pa_hit_rate,
    plus the same keys add_shrinkage_component needs (batter_id, game_season,
    game_date, gamepk, play_id, is_hit).

    Fallback chain, achieved by filling either missing side with that
    season's league average before multiplying:
      - pitcher rate missing (rookie/rare pitcher): b*L/L = b -- collapses
        to the batter-only target, same as add_shrinkage_component.
      - last_season_ba missing (rookie batter): L*p/L = p -- collapses to
        the pitcher-only rate.
      - both missing: L*L/L = L -- season league average, same floor as
        add_shrinkage_component.
    matchup_target is clipped to [0, 1] -- it becomes this row's empirical-
    Bayes prior, which must be a valid probability.
    """
    df = df.copy()
    season_league_avg = df.groupby('game_season')['is_hit'].transform('mean')

    batter_rate = df['last_season_ba'].fillna(season_league_avg)
    pitcher_rate = df['pitcher_last_season_pa_hit_rate'].fillna(season_league_avg)
    matchup_target = np.clip(
        batter_rate * pitcher_rate / season_league_avg.replace(0, np.nan), 0, 1
    )

    df, pred = _cascading_shrinkage(df, matchup_target, k)
    df['matchup_shrinkage_pred'] = pred

    return df


def add_matchup_shrinkage_component_blended(df: pd.DataFrame, k: float = 100.0) -> pd.DataFrame:
    """
    Symmetric version of add_matchup_shrinkage_component: the pitcher side
    of the log5 matchup target is now ALSO its own point-in-time-safe
    empirical-Bayes cascade (this-season cumulative hit-rate-allowed shrunk
    toward last-season rate) instead of a frozen last-season snapshot --
    exactly the same treatment the batter side already gets. Diagnosed as
    the likely reason add_matchup_shrinkage_component came back flat
    (calibration_only, not real_improvement): a pitcher's actual in-season
    form (e.g. a dominant arm pitching shakily in April) was invisible to
    the static version all season.

    Requires the same columns as add_matchup_shrinkage_component, plus
    realized_pitcher_key_id and pitcher_role (the grouping key for the
    pitcher-side cascade -- individual pitcher_id for 'sp' rows, pooled
    pitcher_team_id for 'bullpen' rows, same convention used throughout
    this repo's pitcher season-stats tables).

    The batter side of the log5 combination stays last_season_ba (static)
    -- the batter's own in-season effect is already applied by the outer
    cascade, so blending it into the log5 input too would double-count it.
    """
    df = df.copy()
    season_league_avg = df.groupby('game_season')['is_hit'].transform('mean')

    pitcher_prior = df['pitcher_last_season_pa_hit_rate'].fillna(season_league_avg)
    _, pitcher_blended = _cascading_shrinkage(
        df, pitcher_prior, k, entity_cols=('realized_pitcher_key_id', 'pitcher_role')
    )
    pitcher_blended = pitcher_blended.reindex(df.index)

    batter_rate = df['last_season_ba'].fillna(season_league_avg)
    pitcher_rate = pitcher_blended.fillna(season_league_avg)
    matchup_target = np.clip(
        batter_rate * pitcher_rate / season_league_avg.replace(0, np.nan), 0, 1
    )

    df, pred = _cascading_shrinkage(df, matchup_target, k)
    df['matchup_shrinkage_blended_pred'] = pred

    return df
