import numpy as np


def assign_expected_pitcher_role(pa_outcome, pitcher_start_depth_stats, league_avg_start_depth):
    """Replaces realized pitcher_role/pitcher_key_id as the pitcher-side
    merge/gating key with a pre-game-knowable estimate: expected_pitcher_role
    is 'sp' while the batter's estimated team PA position is still within the
    starter's own historical average depth, 'bullpen' after. Deliberately
    applied to historical (already-completed) rows too, not just future
    predictions — using the realized outcome for training rows and only the
    estimate at serving time would teach the model precision it will never
    have live (train/serve skew).

    Fallback chain when the starter has no individual depth stat (rookie
    call-up, or no starts logged last season): league-wide average depth for
    that season, then 'always sp' if even that doesn't exist (e.g. the very
    first ingested season) — never gate a PA to bullpen purely for lack of
    information.
    """
    df = pa_outcome.merge(
        pitcher_start_depth_stats[
            ['pitcher_id', 'game_season', 'pitcher_last_season_start_avg_batters_faced_per_start']
        ].rename(columns={'pitcher_id': 'starting_pitcher_id'}),
        on=['starting_pitcher_id', 'game_season'], how='left',
    )
    df = df.merge(league_avg_start_depth, on='game_season', how='left')

    own_depth = df['pitcher_last_season_start_avg_batters_faced_per_start']
    df['expected_role_used_league_fallback'] = own_depth.isna()

    avg_depth = own_depth.fillna(df['league_last_season_avg_batters_faced_per_start'])
    avg_depth = avg_depth.fillna(np.inf)

    df['expected_pitcher_role'] = np.where(
        df['estimated_team_pa_position'] <= avg_depth, 'sp', 'bullpen'
    )
    df['expected_times_through_order'] = np.where(
        df['expected_pitcher_role'] == 'sp',
        df['batter_pa_number'].clip(upper=3),
        np.nan,
    )
    df['expected_pitcher_key_id'] = np.where(
        df['expected_pitcher_role'] == 'sp',
        df['starting_pitcher_id'],
        df['pitcher_team_id'],
    )

    return df.drop(columns=[
        'pitcher_last_season_start_avg_batters_faced_per_start',
        'league_last_season_avg_batters_faced_per_start',
    ])
