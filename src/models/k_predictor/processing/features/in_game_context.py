import numpy as np


def build_pitcher_in_game_running_stats(df):
    """Realized within-game running state as of STRICTLY BEFORE each PA --
    the genuinely new information a bettor/analyst has mid-game that no
    pre-game feature in this pipeline carries (contrast
    pitcher_workload.build_pitcher_projected_workload, which projects
    pre-game pace forward rather than using realized in-game outcomes).

    Ordered by real chronological play_id within (gamepk, pitcher_id), same
    pattern already established in
    experiments/count_distribution_check/within_game_correlation_check.py
    (sort_values(["gamepk", "pitcher_id", "play_id"]) + groupby cumcount).
    Resets at the start of every new (gamepk, pitcher_id) group -- a
    pitcher's prior start never leaks into the next one, and two pitchers
    in the same gamepk (home SP / away SP) never see each other's counts.

    df must have: gamepk, pitcher_id, play_id, is_strikeout.

    Adds:
      pitcher_pa_faced_this_game_so_far -- count of this pitcher's PAs in
        this game strictly before the current one (0 at the first PA)
      pitcher_k_this_game_so_far -- strikeouts among those prior PAs
      pitcher_k_this_game_so_far_rate -- k / pa_faced, NaN at the first PA
        (no prior PAs to compute a rate from -- 0/0, not a real 0)
    """
    df = df.sort_values(['gamepk', 'pitcher_id', 'play_id']).copy()
    grp = df.groupby(['gamepk', 'pitcher_id'])

    df['pitcher_pa_faced_this_game_so_far'] = grp.cumcount()
    df['pitcher_k_this_game_so_far'] = grp['is_strikeout'].cumsum() - df['is_strikeout']
    df['pitcher_k_this_game_so_far_rate'] = (
        df['pitcher_k_this_game_so_far'] / df['pitcher_pa_faced_this_game_so_far'].replace(0, np.nan)
    )

    return df


def build_pitcher_in_game_hot_cold_gap(df, pregame_rate_col='pitcher_roll_season_pa_strikeout_rate'):
    """Is this pitcher over- or under-performing his own established
    pre-game rate TONIGHT, so far? Deliberately built from an
    already-point-in-time-safe PRE-GAME rate column rather than a model's
    own predicted probabilities -- using a model's own predictions as a
    feature into itself (when fit on the same rows the model was fit on)
    would be circular/leaky; a pre-game rate constant carries no such risk.

    Must run after build_pitcher_in_game_running_stats (needs its
    pitcher_pa_faced_this_game_so_far / pitcher_k_this_game_so_far output).

    Adds:
      pitcher_expected_k_this_game_so_far -- pregame_rate_col * PAs faced
        so far (a real 0 at the first PA, not NaN -- 0 PAs faced means 0
        expected strikeouts is a fact, not an undefined rate)
      pitcher_hot_cold_gap_this_game_so_far -- actual minus expected;
        positive means running hotter than his own established rate
    """
    df = df.copy()
    df['pitcher_expected_k_this_game_so_far'] = (
        df[pregame_rate_col] * df['pitcher_pa_faced_this_game_so_far']
    )
    df['pitcher_hot_cold_gap_this_game_so_far'] = (
        df['pitcher_k_this_game_so_far'] - df['pitcher_expected_k_this_game_so_far']
    )

    return df
