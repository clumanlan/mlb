"""
processing/features/tier1_feats.py

Tier 1 candidate features from research/feature_glossary_gap_analysis.md
("build now — no new ingestion" section): everything derivable from
columns already pulled by this pipeline. Two kinds of function live here:

- Pure formula/combination functions (build_log5_matchup_features,
  build_velocity_decline_trend_features) that operate on an already-
  assembled model_df — same category as interaction_feats.py, which this
  file follows as a precedent for "combine already-built features" logic
  kept separate from the season/rolling aggregation builders.
- Small aggregation helpers (pitch_type_entropy) consumed by
  season_stats.py's pitcher aggregation functions.

Stat definitions/rationale: see FEATURE_GLOSSARY.md.
"""

import numpy as np
import pandas as pd

from .interaction_feats import build_trend_features, find_rolling_trend_pairs


def pitch_type_entropy(pitch_types: pd.Series) -> float:
    """Shannon entropy (base 2, in bits) of a pitch-type distribution.

    0 = maximally predictable (always the same pitch), higher = more varied
    arsenal usage. A low-entropy pitcher is measurably easier to sit on —
    see FEATURE_GLOSSARY.md / research/feature_glossary_gap_analysis.md.
    """
    counts = pitch_types.value_counts()
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


DEFAULT_LOG5_OUTCOMES = ['single', 'xbh', 'hr', 'walk', 'hbp', 'strikeout']


def build_log5_matchup_features(
    df: pd.DataFrame,
    outcomes: list[str] = DEFAULT_LOG5_OUTCOMES,
    batter_prefix: str = 'batter_season_pa_',
    pitcher_prefix: str = 'pitcher_season_pa_',
    league_prefix: str = 'league_season_pa_',
) -> pd.DataFrame:
    """Extended log5 matchup: log5_matchup_{outcome} = batter_rate *
    pitcher_rate / league_rate, for each outcome in `outcomes`.

    Classic log5 (SABR) generalized from a single win-probability event to
    the PA-outcome multinomial — six of the seven outcomes the audit named
    (1B/HR/BB/HBP/K, with 2B+3B folded into xbh since this pipeline doesn't
    track them as separate rate columns; see FEATURE_GLOSSARY.md). Operates
    on an already-assembled model_df carrying the batter/pitcher/league
    season-rate columns those existing builders produce — a join + formula
    on data already merged, not a new source. Zero league_rate guards to
    NaN rather than raising or producing inf.
    """
    new_cols = {}
    for outcome in outcomes:
        batter_col = f'{batter_prefix}{outcome}_rate'
        pitcher_col = f'{pitcher_prefix}{outcome}_rate'
        league_col = f'{league_prefix}{outcome}_rate'
        new_cols[f'log5_matchup_{outcome}'] = (
            df[batter_col] * df[pitcher_col] / df[league_col].replace(0, np.nan)
        )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


DEFAULT_VELOCITY_DECLINE_STATS = ('stuff_start_speed_mean', 'stuff_spin_rate_mean')


def build_velocity_decline_trend_features(
    df: pd.DataFrame, stats: tuple[str, ...] = DEFAULT_VELOCITY_DECLINE_STATS
) -> pd.DataFrame:
    """Fatigue/decline signal: trend_ratio/trend_direction (short-window vs
    season-long, via interaction_feats.build_trend_features) restricted to
    velocity/spin stats only.

    Deliberately narrower than v3_interaction_feats/train.py's blanket
    find_rolling_trend_pairs(all columns) approach, which measurably hurt
    val metrics when applied to every rolling stat pair (see
    interaction_feats.build_trend_features's docstring) — a velocity dip
    specifically is a documented leading fatigue/injury indicator (see
    research/feature_glossary_gap_analysis.md), a narrower and better-
    motivated hypothesis than the blanket one, not a re-run of it.
    """
    all_pairs = find_rolling_trend_pairs(list(df.columns))
    target_pairs = [
        (short_col, season_col) for short_col, season_col in all_pairs
        if any(season_col.endswith(f'_roll_season_{stat}') for stat in stats)
    ]
    return build_trend_features(df, target_pairs)
