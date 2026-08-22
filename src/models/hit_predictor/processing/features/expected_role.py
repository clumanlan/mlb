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


# Fixed approximation for converting a PA-position estimate into an inning estimate —
# league-average team plate appearances per (half-)inning. Same spirit as
# estimated_team_pa_position's own 9-batter-cycle assumption (pipeline.py): not exact
# for any single inning (a high-scoring inning sends more than ~4.3 batters to the
# plate), an accepted approximation rather than a bug.
AVG_TEAM_PA_PER_INNING = 4.3


def assign_expected_pitcher_role_by_inning(pa_outcome, expected_start_innings):
    """Inning-based replacement for assign_expected_pitcher_role: estimates
    which inning a PA falls in from the same pre-game-knowable
    estimated_team_pa_position used by the batters-faced version (converted
    via AVG_TEAM_PA_PER_INNING), then compares it against the starter's own
    pre-game expected_start_innings (game_context.build_expected_start_innings
    — a shrinkage blend of his last-season baseline and this-season rolling
    average, itself already falling back to league-wide when he has no track
    record) to decide sp vs bullpen for that PA.

    Unlike assign_expected_pitcher_role, the upstream fallback chain
    (individual -> league -> this-season-alone) already happened inside
    build_expected_start_innings, so this function only needs one further
    fallback: a PA with no matching expected_start_innings row at all
    (shouldn't happen given that chain, but not assumed) defaults to 'sp' —
    same "never gate to bullpen purely for lack of information" philosophy.

    estimated_inning is kept as its own output column (not just consumed
    internally) — same 'expose the intermediate, not just the final
    decision' principle used throughout this project.
    """
    df = pa_outcome.merge(
        expected_start_innings[['personId', 'gamepk', 'expected_start_innings']].rename(
            columns={'personId': 'starting_pitcher_id'}
        ),
        on=['starting_pitcher_id', 'gamepk'], how='left',
    )

    df['estimated_inning'] = np.ceil(df['estimated_team_pa_position'] / AVG_TEAM_PA_PER_INNING)
    avg_depth = df['expected_start_innings'].fillna(np.inf)

    df['expected_pitcher_role'] = np.where(
        df['estimated_inning'] <= avg_depth, 'sp', 'bullpen'
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

    return df.drop(columns=['expected_start_innings'])
