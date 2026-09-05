def build_pitcher_shrunk_whip(
    pitcher_rolling_all_roles, pitcher_season_all_roles, window: str | int = 'season', k: float = 20.0,
):
    """Blend last season's WHIP toward this season's own emerging rolling
    WHIP as in-season games accumulate — shrinkage_weight = games_n /
    (games_n + k), 0 at a season opener (trust last season fully), rising
    toward 1 as this season's own sample grows. Same pattern as
    hit_predictor's game_context.build_expected_start_innings, generalized
    to WHIP and to both pitcher roles (not SP-only) via pitcher_key_id/
    pitcher_role instead of personId.

    Baseline falls back to this season's own rolling WHIP when there's no
    last-season row (rookie/unseen pitcher) — best available signal rather
    than NaN.

    pitcher_rolling_all_roles: rolling_stats.build_pitcher_rolling_stats_all_roles output.
    pitcher_season_all_roles: season_stats.build_pitcher_stats_all_roles output (already
        shifted to last season).
    """
    prefix = 'pitcher_roll_season_' if window == 'season' else f'pitcher_roll_last{window}g_'
    rolling_col = f'{prefix}whip'
    games_col = f'{prefix}games_n'

    df = pitcher_rolling_all_roles.merge(
        pitcher_season_all_roles[['pitcher_key_id', 'pitcher_role', 'game_season', 'pitcher_last_season_whip']],
        on=['pitcher_key_id', 'pitcher_role', 'game_season'], how='left',
    )

    games_n = df[games_col].fillna(0)
    weight = games_n / (games_n + k)
    baseline = df['pitcher_last_season_whip'].fillna(df[rolling_col])
    rolling_safe = df[rolling_col].fillna(baseline)

    df['pitcher_shrunk_whip_weight'] = weight
    df['pitcher_shrunk_whip'] = (1 - weight) * baseline + weight * rolling_safe

    return df


def build_pitcher_projected_workload(
    df, position_col: str = 'estimated_team_pa_position',
    pace_col: str = 'pitcher_roll_last3g_pa_pitch_count_mean',
    out_col: str = 'pitcher_projected_pitches_before_pa',
):
    """Projects how many pitches the pitcher has thrown BEFORE this PA:
    (batters faced before this PA) * (pitches per PA, trailing-3-game pace).

    estimated_team_pa_position is this PA's 1-indexed slot in the game (see
    pipeline.py's _add_estimated_team_pa_position) -- position - 1 batters
    have been faced already. Pre-game-knowable for the same reason
    expected_role.py already treats estimated_team_pa_position/batter_pa_number
    as safe to use directly: both are a deterministic function of
    batting_order (known before first pitch), not a leaked realized outcome.

    Sharper fatigue signal than PA-position or times_through_order alone --
    two batters at the same slot can represent very different actual
    workload depending on how many pitches the pitcher's been burning per
    batter (deep counts, foul balls, working around traffic).
    """
    df = df.copy()
    df[out_col] = (df[position_col] - 1) * df[pace_col]
    return df
