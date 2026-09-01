import re

import numpy as np
import pandas as pd

# Helpers for v3_interaction_feats. find_rolling_trend_pairs/find_sample_size_col
# locate raw column pairs — first used to check via 2-way partial dependence
# (see experiments/v3_interaction_feats/train.py) whether a tree model fit on
# raw features already learns a given interaction; once a PDP check shows it
# doesn't, build_trend_features/build_shrinkage_weight_features turn the
# same pairs into real engineered columns.

_SEASON_RE = re.compile(r'^(?P<prefix>.+)_roll_season_(?P<stat>.+)$')
_SHORT_RE = re.compile(r'^(?P<prefix>.+)_roll_last(?P<window>\d+)g_(?P<stat>.+)$')

# Checked in priority order: a rate's own true PA denominator (plate_appearances,
# or pa_total for pbp-derived rates) is the most direct sample-size signal;
# ab/ip/n_pitches are coarser fallbacks for stat families with no PA column.
_SAMPLE_SIZE_SUFFIXES = ['plate_appearances', 'pa_total', 'ab', 'ip', 'n_pitches']


def find_rolling_trend_pairs(columns: list[str]) -> list[tuple[str, str]]:
    """Pair every *_roll_last{N}g_{stat} column with its *_roll_season_{stat}
    counterpart (same entity prefix, same stat name), when both exist in
    `columns`. The short window's game count (N) is not hardcoded — any
    *_roll_last{N}g_ prefix matches.

    Returns (short_col, season_col) tuples, sorted for determinism.
    """
    season_lookup = {}
    for col in columns:
        m = _SEASON_RE.match(col)
        if m:
            season_lookup[(m.group('prefix'), m.group('stat'))] = col

    pairs = []
    for col in columns:
        m = _SHORT_RE.match(col)
        if not m:
            continue
        season_col = season_lookup.get((m.group('prefix'), m.group('stat')))
        if season_col:
            pairs.append((col, season_col))

    return sorted(pairs)


def _entity_and_stat(col: str) -> tuple[str, str] | None:
    """Split a rolling column into (entity_prefix, stat_name), matching
    either the short-window or season-window naming pattern. Returns None
    for columns that aren't rolling stats at all (e.g. 'batting_order')."""
    m = _SHORT_RE.match(col)
    if m:
        return m.group('prefix'), m.group('stat')
    m = _SEASON_RE.match(col)
    if m:
        return m.group('prefix'), m.group('stat')
    return None


def _rolling_prefix(col: str) -> str | None:
    m = _SHORT_RE.match(col)
    if m:
        return f"{m.group('prefix')}_roll_last{m.group('window')}g_"
    m = _SEASON_RE.match(col)
    if m:
        return f"{m.group('prefix')}_roll_season_"
    return None


def find_sample_size_col(columns: list[str], rate_col: str) -> str | None:
    """Find the sample-size denominator column sharing rate_col's rolling
    prefix (e.g. 'batter_roll_last10g_') — the count feature indicating how
    much history a short-window rate is built on. Returns None if no
    candidate from _SAMPLE_SIZE_SUFFIXES is present in `columns` (or the only
    candidate present is rate_col itself — e.g. asking for
    plate_appearances's own sample-size column, a degenerate self-pair).
    """
    prefix = _rolling_prefix(rate_col)
    if prefix is None:
        return None

    for suffix in _SAMPLE_SIZE_SUFFIXES:
        candidate = f'{prefix}{suffix}'
        if candidate in columns and candidate != rate_col:
            return candidate

    return None


def build_trend_features(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Add two columns for every (short_col, season_col) pair (e.g. from
    find_rolling_trend_pairs):
      - {prefix}_trend_ratio_{stat} = short / season: how many times a
        player's recent-window rate is running relative to their own season
        baseline (>1 hot, <1 cold, NaN-guarded against a zero season value).
      - {prefix}_trend_direction_{stat} = sign(short - season): a coarse
        +1/-1/0 hot/cold/flat indicator, throwing away magnitude.

    An earlier version of this function (build_trend_diff_features, plain
    short - season) got used heavily by the Random Forest — it dominated the
    feature-importance ranking — but measurably hurt held-out val metrics
    (ROC-AUC, PR-AUC, log loss, Brier, decile spread all worse than the
    raw-features baseline). A deterministic linear combination of two
    already-present raw columns adds no information a tree can't already
    reconstruct with an extra split, while diluting RandomForestClassifier's
    max_features='sqrt' random split sampling with a near-duplicate column.
    Ratio and direction are non-linear/coarsened transforms instead, tested
    as a different hypothesis, not a guaranteed fix — see
    experiments/v3_interaction_feats/train.py for the val-metric comparison.
    """
    # Collect all new columns in a dict and concat once — assigning columns
    # one at a time onto a copy (result[col] = ...) fragments pandas' internal
    # block manager on a wide/long frame like model_df (hundreds of columns,
    # 600k+ rows), which is slow and can spike memory enough to OOM-kill the
    # process (observed in practice during v3_interaction_feats/train.py).
    new_cols = {}
    for short_col, season_col in pairs:
        parsed = _entity_and_stat(short_col)
        if parsed is None:
            continue
        prefix, stat = parsed
        new_cols[f'{prefix}_trend_ratio_{stat}'] = df[short_col] / df[season_col].replace(0, np.nan)
        new_cols[f'{prefix}_trend_direction_{stat}'] = np.sign(df[short_col] - df[season_col])
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _parse_rolling_col(col: str) -> tuple[str, str, str] | None:
    """Like _entity_and_stat, but keeps the window bucket ('season' or
    'last{N}g') instead of discarding it -- player-vs-league pairing needs
    an EXACT window match, unlike trend-pairing above which spans windows
    on purpose (short window vs. season baseline)."""
    m = _SHORT_RE.match(col)
    if m:
        return m.group('prefix'), f"last{m.group('window')}g", m.group('stat')
    m = _SEASON_RE.match(col)
    if m:
        return m.group('prefix'), 'season', m.group('stat')
    return None


def find_player_vs_league_pairs(columns: list[str]) -> list[tuple[str, str]]:
    """Pair every non-league rolling column with the league_roll_{window}_{stat}
    column sharing its exact window bucket and stat name (e.g.
    batter_roll_season_pa_strikeout_rate <-> league_roll_season_pa_strikeout_rate),
    regardless of the player-side entity prefix (batter/pitcher/opp_team/
    pitching_team/anything). A league_roll_* column never appears as the
    player side of a pair.

    Returns (player_col, league_col) tuples, sorted for determinism. Sample-
    size denominator columns (_SAMPLE_SIZE_SUFFIXES: plate_appearances,
    pa_total, ab, ip, n_pitches) are excluded on both sides -- these are raw
    COUNTS, not rates, so a player's own count divided by the league's
    cumulative count is not a meaningful ratio (caught via a real degenerate
    run: pitcher_vs_league_ratio_season_pa_total came back with mean 0.0037
    and a constant direction of -1.0).
    """
    league_lookup = {}
    for col in columns:
        parsed = _parse_rolling_col(col)
        if parsed is None:
            continue
        entity, window, stat = parsed
        if entity == 'league' and stat not in _SAMPLE_SIZE_SUFFIXES:
            league_lookup[(window, stat)] = col

    pairs = []
    for col in columns:
        parsed = _parse_rolling_col(col)
        if parsed is None:
            continue
        entity, window, stat = parsed
        if entity == 'league' or stat in _SAMPLE_SIZE_SUFFIXES:
            continue
        league_col = league_lookup.get((window, stat))
        if league_col:
            pairs.append((col, league_col))

    return sorted(pairs)


def build_player_vs_league_features(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Add two columns for every (player_col, league_col) pair (e.g. from
    find_player_vs_league_pairs):
      - {entity}_vs_league_ratio_{window}_{stat} = player / league: is this
        player's rate above or below what the league is doing RIGHT NOW
        (>1 above, <1 below, NaN-guarded against a zero league value) --
        the sabermetric "+"-stat pattern (wRC+/ERA+ are literally
        player-rate-over-league-rate, era-adjusted).
      - {entity}_vs_league_direction_{window}_{stat} = sign(player - league):
        a coarse +1/-1/0 above/below/equal indicator, magnitude discarded --
        same rationale as build_trend_features's own direction column
        (a plain difference is a linear combination of two already-present
        raw columns a tree can reconstruct with one extra split; ratio and
        direction are the non-linear/coarsened transforms that add real
        information).
    """
    # Same fragmentation/OOM-risk reasoning as build_trend_features above —
    # build every new column in a dict, concat once.
    new_cols = {}
    for player_col, league_col in pairs:
        parsed = _parse_rolling_col(player_col)
        if parsed is None:
            continue
        entity, window, stat = parsed
        new_cols[f'{entity}_vs_league_ratio_{window}_{stat}'] = df[player_col] / df[league_col].replace(0, np.nan)
        new_cols[f'{entity}_vs_league_direction_{window}_{stat}'] = np.sign(df[player_col] - df[league_col])
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_shrinkage_weight_features(
    df: pd.DataFrame, rate_to_sample_col: dict[str, str], k: float = 10.0
) -> pd.DataFrame:
    """For each {rate_col: sample_size_col}, add:
      - {prefix}_shrinkage_weight_{stat} = sample / (sample + k): a smooth
        0->1 confidence weight, 0.5 at sample==k.
      - {prefix}_shrunk_{stat} = rate * weight: an explicit product between
        a rolling rate and its own sample size, added because the PDP
        diagnostic in experiments/v3_interaction_feats/train.py showed the
        raw (rate, sample_size) pair isn't already combined this way —
        the rate's PDP sensitivity didn't visibly change with sample size.
    """
    # Same fragmentation/OOM-risk reasoning as build_trend_features above —
    # build every new column in a dict, concat once.
    new_cols = {}
    for rate_col, sample_col in rate_to_sample_col.items():
        parsed = _entity_and_stat(rate_col)
        if parsed is None:
            continue
        prefix, stat = parsed
        weight = df[sample_col] / (df[sample_col] + k)
        new_cols[f'{prefix}_shrinkage_weight_{stat}'] = weight
        new_cols[f'{prefix}_shrunk_{stat}'] = df[rate_col] * weight
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
