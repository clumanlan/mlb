import numpy as np
import pandas as pd

from .season_stats import _shift_to_last_season

# Stat definitions/rationale: see FEATURE_GLOSSARY.md. v1 park factor — a
# single-season hit-rate index per venue (trailing 1 season, same
# _shift_to_last_season convention as every other season-level feature in
# this repo). A 3-year rolling window is the sabermetric standard for
# reducing single-season noise, but is deferred — see the "Next steps" note
# this feature was added under.


def _create_venue_hit_factor(schedule: pd.DataFrame, batter_boxscore: pd.DataFrame) -> pd.DataFrame:
    """One row per (venue_id, game_season): that venue's hit rate that
    season divided by the league-wide hit rate that season. > 1 = hitter-
    friendly, < 1 = pitcher-friendly, 1 = neutral."""

    game_venue = (
        schedule[['gamepk', 'venue_id', 'game_date']]
        .drop_duplicates('gamepk')
        .assign(game_season=lambda x: x['game_date'].dt.year)
        [['gamepk', 'venue_id', 'game_season']]
    )

    game_hitting = (
        batter_boxscore
        .groupby('gamepk')[['h', 'ab']].sum()
        .reset_index()
        .merge(game_venue, on='gamepk', how='inner')
    )

    venue_stats = (
        game_hitting
        .groupby(['venue_id', 'game_season'])[['h', 'ab']].sum()
        .reset_index()
        .rename(columns={'h': 'park_season_h', 'ab': 'park_season_ab'})
    )

    league_stats = (
        game_hitting
        .groupby('game_season')[['h', 'ab']].sum()
        .reset_index()
        .assign(league_season_hit_rate=lambda x: x['h'] / x['ab'].replace(0, np.nan))
        [['game_season', 'league_season_hit_rate']]
    )

    df = venue_stats.merge(league_stats, on='game_season', how='left')
    df['park_season_hit_factor'] = (
        (df['park_season_h'] / df['park_season_ab'].replace(0, np.nan))
        / df['league_season_hit_rate']
    )

    return df[['venue_id', 'game_season', 'park_season_hit_factor', 'park_season_ab']]


def build_park_factors(schedule: pd.DataFrame, batter_boxscore: pd.DataFrame) -> pd.DataFrame:
    df = _create_venue_hit_factor(schedule, batter_boxscore)

    return _shift_to_last_season(df)


def _create_venue_strikeout_factor(schedule: pd.DataFrame, pitcher_boxscore: pd.DataFrame) -> pd.DataFrame:
    """One row per (venue_id, game_season): that venue's K rate (k/ip) that
    season divided by the league-wide K rate that season — same shape as
    _create_venue_hit_factor, k/ip instead of h/ab (matching
    season_stats.build_pitcher_stats' own k_rate = k/ip convention). > 1 =
    strikeout-friendly, < 1 = strikeout-suppressing, 1 = neutral."""

    game_venue = (
        schedule[['gamepk', 'venue_id', 'game_date']]
        .drop_duplicates('gamepk')
        .assign(game_season=lambda x: x['game_date'].dt.year)
        [['gamepk', 'venue_id', 'game_season']]
    )

    game_pitching = (
        pitcher_boxscore
        .groupby('gamepk')[['k', 'ip']].sum()
        .reset_index()
        .merge(game_venue, on='gamepk', how='inner')
    )

    venue_stats = (
        game_pitching
        .groupby(['venue_id', 'game_season'])[['k', 'ip']].sum()
        .reset_index()
        .rename(columns={'k': 'park_season_k', 'ip': 'park_season_ip'})
    )

    league_stats = (
        game_pitching
        .groupby('game_season')[['k', 'ip']].sum()
        .reset_index()
        .assign(league_season_k_rate=lambda x: x['k'] / x['ip'].replace(0, np.nan))
        [['game_season', 'league_season_k_rate']]
    )

    df = venue_stats.merge(league_stats, on='game_season', how='left')
    df['park_season_strikeout_factor'] = (
        (df['park_season_k'] / df['park_season_ip'].replace(0, np.nan))
        / df['league_season_k_rate']
    )

    return df[['venue_id', 'game_season', 'park_season_strikeout_factor', 'park_season_ip']]


def build_park_strikeout_factor(schedule: pd.DataFrame, pitcher_boxscore: pd.DataFrame) -> pd.DataFrame:
    df = _create_venue_strikeout_factor(schedule, pitcher_boxscore)

    return _shift_to_last_season(df)
