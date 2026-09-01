import numpy as np
import pandas as pd

from .rolling_stats import _rolling_sum, _rolling_max, _rolling_pooled_std, _rolling_prefix
from .season_stats import _prefix_stat_cols, _pitcher_role_lookup

# Pre-game-knowable game-level context (venue, calendar, team form/rest,
# probable starters — never in-game state like inning/count/score, which
# isn't knowable before first pitch). See FEATURE_GLOSSARY.md.


def build_datetime_features(df: pd.DataFrame, datetime_col: str = 'game_datetime') -> pd.DataFrame:
    """Decompose datetime_col into calendar/time-of-day parts, fastai
    add_datepart-style: expose the raw calendar components and let the model
    find any day-of-week/month/hour pattern itself, rather than hand-picking
    a single derived flag (e.g. day/night, which would need a venue-timezone
    lookup we don't have — game_datetime is UTC, so hour/minute reflect UTC
    start time, not local first-pitch time).

    No 'Elapsed'/raw-epoch column — see test_game_context.py's
    test_build_datetime_features_does_not_include_raw_epoch for why.
    """
    dt = df[datetime_col]

    new_cols = {
        'game_dt_year': dt.dt.year,
        'game_dt_month': dt.dt.month,
        'game_dt_week': dt.dt.isocalendar().week.astype('int64'),
        'game_dt_day': dt.dt.day,
        'game_dt_dayofweek': dt.dt.dayofweek,
        'game_dt_dayofyear': dt.dt.dayofyear,
        'game_dt_is_month_end': dt.dt.is_month_end,
        'game_dt_is_month_start': dt.dt.is_month_start,
        'game_dt_is_quarter_end': dt.dt.is_quarter_end,
        'game_dt_is_quarter_start': dt.dt.is_quarter_start,
        'game_dt_is_year_end': dt.dt.is_year_end,
        'game_dt_is_year_start': dt.dt.is_year_start,
        'game_dt_hour': dt.dt.hour,
        'game_dt_minute': dt.dt.minute,
    }

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_doubleheader_flag(df: pd.DataFrame, code_col: str = 'doubleheader') -> pd.DataFrame:
    """Adds is_doubleheader (bool): True when the MLB API's doubleheader
    code is 'Y' (traditional doubleheader) or 'S' (split doubleheader —
    different times/tickets, same two teams same day), False for 'N'. Both
    games of a doubleheader carry the same code, so both rows are flagged —
    both are affected by the day's compressed rest, not just game 2.

    Needs `code_col` (raw schedule's 'doubleheader' column, present in
    raw_data/games/schedule but not yet in processed_data/games/schedule's
    SCHEDULE_COLUMNS — existing processed files need reprocessing from raw
    to pick this up; a separate backfill task, not this function's concern).
    """
    result = df.copy()
    result['is_doubleheader'] = df[code_col].isin(['Y', 'S'])
    return result


# --------------------------- TEAM WIN/LOSS RECORD & REST --------------------------- #
# Team-centric rolling stats, reusing rolling_stats.py's _rolling_sum engine (same
# "roll counts, not rates" rule — a rate is only ever divided once, from already-
# rolled sums) with sort_col='game_datetime': schedule's game_date has no time
# component and can't reliably order two games sharing the same date (a
# doubleheader), so game 1's pre-game record must not leak game 2's same-day result.

TEAM_GAME_KEY_COLS = ['team_id', 'gamepk', 'game_date', 'game_datetime', 'game_season']


def _team_game_long(schedule: pd.DataFrame) -> pd.DataFrame:
    """Reshape schedule (one row per game: home_id/away_id/home_score/away_score)
    into one row per (team_id, gamepk) — the team-centric view every team-level
    stat in this module is built from. Each game contributes exactly 2 rows, one
    per side, with team_id/opp_id/team_score/opp_score/win_n from that team's
    own perspective.
    """
    base_cols = ['gamepk', 'game_date', 'game_datetime', 'home_id', 'away_id', 'home_score', 'away_score']

    home = schedule[base_cols].rename(columns={
        'home_id': 'team_id', 'away_id': 'opp_id',
        'home_score': 'team_score', 'away_score': 'opp_score',
    })
    away = schedule[base_cols].rename(columns={
        'away_id': 'team_id', 'home_id': 'opp_id',
        'away_score': 'team_score', 'home_score': 'opp_score',
    })

    df = pd.concat([home, away], ignore_index=True)
    df['game_season'] = df['game_date'].dt.year
    df['win_n'] = (df['team_score'] > df['opp_score']).astype(int)
    df['games_n'] = 1

    return df


def build_team_win_loss_record(schedule: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """One row per (team_id, gamepk): that team's win/loss record and run
    differential rolled forward — shift(1) so this game's own result never
    leaks into its own pre-game record.

    window='season': expanding season-to-date record (resets each year).
    window=<int>: trailing N-game form indicator (carries across season
        boundaries, same convention as rolling_stats.py's short windows).
    """
    df = _team_game_long(schedule)

    stat_cols = ['win_n', 'games_n', 'team_score', 'opp_score']
    rolled = _rolling_sum(df, entity_col='team_id', cols=stat_cols, window=window, sort_col='game_datetime')

    rolled = rolled.assign(
        win_pct=lambda x: x['win_n'] / x['games_n'].replace(0, np.nan),
        run_diff=lambda x: x['team_score'] - x['opp_score'],
    )
    rolled['run_diff_avg'] = rolled['run_diff'] / rolled['games_n'].replace(0, np.nan)
    rolled = rolled.rename(columns={'team_score': 'runs_scored', 'opp_score': 'runs_allowed'})

    final_cols = [
        'win_n', 'games_n', 'win_pct', 'runs_scored', 'runs_allowed', 'run_diff', 'run_diff_avg',
    ]
    rolled = rolled[TEAM_GAME_KEY_COLS + final_cols]

    return _prefix_stat_cols(rolled, prefix=_rolling_prefix('team', window), key_cols=TEAM_GAME_KEY_COLS)


def build_team_rest_days(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (team_id, gamepk): calendar days since that team's
    previous game (any opponent, home or away). NaN for a team's first game
    in the data (nothing prior to compare against). 0 for the second game of
    a same-day doubleheader — sorted by game_datetime (not game_date alone)
    so the two games order correctly despite sharing a date.
    """
    df = _team_game_long(schedule)
    df = df.sort_values(['team_id', 'game_datetime']).reset_index(drop=True)
    df['team_days_since_last_game'] = df.groupby('team_id')['game_date'].diff().dt.days

    return df[TEAM_GAME_KEY_COLS + ['team_days_since_last_game']]


def build_team_scoring_volatility(schedule: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """One row per (team_id, gamepk): mean/std/max of that team's per-game
    runs scored, rolled forward — shift(1) so this game's own score never
    leaks into its own pre-game read. Same window convention as
    build_team_win_loss_record (window='season': expanding, resets each
    year; window=<int>: trailing N-game).

    Captures scoring EXPLOSIVENESS/volatility, not just level (see
    build_team_win_loss_record's runs_scored, a rolling sum/level stat) —
    a team that occasionally puts up a blowout can put a starter in
    trouble even if its average is unremarkable.

    Std is NaN with fewer than 2 prior games (sample std undefined at
    n<=1). NOTE: _rolling_pooled_std orders by game_date, not
    game_datetime, unlike build_team_win_loss_record/build_team_rest_days
    — a pre-existing limitation shared by every other _rolling_pooled_std
    caller in this file, not something this function fixes.
    """
    per_game = _team_game_long(schedule)
    per_game['team_score_sumsq'] = per_game['team_score'] ** 2

    std_df = _rolling_pooled_std(
        per_game, entity_col='team_id', n_col='games_n', sum_col='team_score',
        sumsq_col='team_score_sumsq', window=window, out_col='runs_scored_std',
    )
    std_df['runs_scored_mean'] = std_df['team_score'] / std_df['games_n'].replace(0, np.nan)

    max_df = _rolling_max(
        per_game, entity_col='team_id', cols=['team_score'], window=window, sort_col='game_datetime',
    ).rename(columns={'team_score': 'runs_scored_max'})

    rolled = std_df[TEAM_GAME_KEY_COLS + ['runs_scored_mean', 'runs_scored_std']].merge(
        max_df[TEAM_GAME_KEY_COLS + ['runs_scored_max']], on=TEAM_GAME_KEY_COLS,
    )

    return _prefix_stat_cols(rolled, prefix=_rolling_prefix('team', window), key_cols=TEAM_GAME_KEY_COLS)


# --------------------------- PROBABLE STARTER JOIN --------------------------- #

def build_probable_starters(schedule: pd.DataFrame, game_info: pd.DataFrame) -> pd.DataFrame:
    """One row per (team_id, gamepk): that team's probable starting pitcher
    ID for this game. schedule carries team identity (home_id/away_id);
    game_info carries the pitcher IDs (schedule only has probable-pitcher
    NAMES, not IDs — see GAME_INFO_COLUMNS vs SCHEDULE_COLUMNS). Left-joined
    on gamepk so a game with no matching game_info row still gets a row per
    team, with a null probable_starter_id, rather than being dropped.
    """
    merged = schedule[['gamepk', 'home_id', 'away_id']].merge(
        game_info[['gamepk', 'probable_pitcher_home_id', 'probable_pitcher_away_id']],
        on='gamepk', how='left',
    )

    home = merged[['gamepk', 'home_id', 'probable_pitcher_home_id']].rename(
        columns={'home_id': 'team_id', 'probable_pitcher_home_id': 'probable_starter_id'}
    )
    away = merged[['gamepk', 'away_id', 'probable_pitcher_away_id']].rename(
        columns={'away_id': 'team_id', 'probable_pitcher_away_id': 'probable_starter_id'}
    )

    return pd.concat([home, away], ignore_index=True)


# --------------------------- STARTER INNINGS ESTIMATE (Epic E) --------------------------- #
# season_stats.py's build_pitcher_start_ip_stats gives a fixed last-season baseline.
# This complements it with the SAME pitcher's own in-season, rolling average — thin/empty
# early in a new season (few starts so far), which is exactly why build_expected_start_innings
# blends the two, shrinking toward this season's own emerging average as starts accumulate
# rather than trusting last season's number forever.

def build_pitcher_start_ip_this_season(
    pitcher_boxscore: pd.DataFrame, pbp: pd.DataFrame, window: str | int = 'season'
) -> pd.DataFrame:
    """One row per (personId, gamepk) SP start: rolling avg IP per start and
    starts-so-far count, UP TO (not including) this start, shift(1), same
    _rolling_sum engine as rolling_stats.py (sort_col='game_datetime' for the
    same doubleheader-ordering reason as the team stats above).

    window='season' (default, preserves every existing caller's column names
        and values): expanding within season — a season-to-date average.
    window=<int>: trailing N-start average, carrying across season
        boundaries — same 'recent form, not season-to-date' distinction
        _rolling_sum's own window=<int> branch documents for team stats.
        Distinct column prefix ('pitcher_last{N}_start_ip_') so a caller can
        merge both alongside each other without a collision.
    """
    role_lookup = _pitcher_role_lookup(pbp)[['gamepk', 'pitcher_id', 'pitcher_role']].rename(
        columns={'pitcher_id': 'personId'}
    )
    tagged = pitcher_boxscore.assign(
        personId=lambda x: x['personId'].astype(str), gamepk=lambda x: x['gamepk'].astype(str)
    ).merge(
        role_lookup.assign(
            personId=lambda x: x['personId'].astype(str), gamepk=lambda x: x['gamepk'].astype(str)
        ),
        on=['gamepk', 'personId'],
        how='left',
    )

    sp_box = tagged[tagged['pitcher_role'] == 'sp'].copy()
    sp_box['starts_n'] = 1

    rolled = _rolling_sum(
        sp_box, entity_col='personId', cols=['ip', 'starts_n'], window=window, sort_col='game_datetime'
    )
    rolled['avg_ip_per_start'] = rolled['ip'] / rolled['starts_n'].replace(0, np.nan)
    # starts_n itself is a real, meaningful 0 for a season-opening start (not NaN) —
    # only its ratio (avg_ip_per_start) should be undefined with no prior starts.
    rolled['starts_n'] = rolled['starts_n'].fillna(0)

    key_cols = ['personId', 'gamepk', 'game_date', 'game_datetime', 'game_season']
    rolled = rolled[key_cols + ['starts_n', 'avg_ip_per_start']]

    prefix = 'pitcher_this_season_start_ip_' if window == 'season' else f'pitcher_last{window}_start_ip_'
    return _prefix_stat_cols(rolled, prefix=prefix, key_cols=key_cols)


def build_expected_start_innings(
    pitcher_start_ip_last_season: pd.DataFrame,
    pitcher_start_ip_this_season: pd.DataFrame,
    league_avg_start_ip: pd.DataFrame,
    k: float = 5.0,
) -> pd.DataFrame:
    """Blend last season's fixed baseline IP/start toward this season's own
    emerging average as starts accumulate — shrinkage_weight = starts_n /
    (starts_n + k), 0 at a season opener (trust last season fully), rising
    toward 1 as this season's own sample grows. Baseline itself falls back
    last-season -> league-wide -> this-season-alone, for a rookie or
    otherwise-unseen pitcher with no last-season row.

    NOTE — this is a 2-level cascade (pitcher -> league), not 3-level. See
    build_expected_batters_faced below for the pitcher -> team -> league
    version built for k_predictor: this function deliberately wasn't
    extended with a team-level rung to match, since n_pa_predictor's locked
    production threshold and short_outing_predictor's baseline+v1 both
    already depend on its exact current fallback behavior — adding a team
    rung here would change their feature values and require re-verifying
    both. If a team-level fallback for IP-per-start is ever wanted, build a
    third function rather than changing this one's behavior in place.

    Every input to the formula (not just the final blend) survives as its
    own column — same 'expose raw denominators, let the model see the
    building blocks' principle as rolling_stats.py's sample-size columns.
    Unlike that case (where an explicit engineered feature got tested
    against a real model and didn't help — see interaction_feats.py), this
    blended number drives real algorithmic logic (how many rows the
    long-format sp/bullpen expansion emits), not a redundant model input.
    """
    df = pitcher_start_ip_this_season.merge(
        pitcher_start_ip_last_season[[
            'personId', 'game_season',
            'pitcher_last_season_start_ip_avg_ip_per_start',
            'pitcher_last_season_start_ip_n_starts',
        ]],
        on=['personId', 'game_season'], how='left',
    )
    df = df.merge(league_avg_start_ip, on='game_season', how='left')

    baseline = (
        df['pitcher_last_season_start_ip_avg_ip_per_start']
        .fillna(df['league_last_season_avg_ip_per_start'])
        .fillna(df['pitcher_this_season_start_ip_avg_ip_per_start'])
    )

    starts_n = df['pitcher_this_season_start_ip_starts_n']
    weight = starts_n / (starts_n + k)
    this_season_avg_safe = df['pitcher_this_season_start_ip_avg_ip_per_start'].fillna(0)

    df['expected_start_innings_weight'] = weight
    df['expected_start_innings'] = (1 - weight) * baseline + weight * this_season_avg_safe

    return df


# --------------------------- PITCHER START PA (batters-faced estimate, k_predictor) --------------------------- #
# k_predictor's own 3-level pitcher -> team -> league shrinkage cascade for "how many
# batters will this starter face" — same overall shape as the IP cascade above, but
# pbp-derived (no pitcher_boxscore input needed: batters faced is a PA count, and pbp
# already carries pitcher_role directly) and with a team-level middle rung this file's
# IP cascade deliberately doesn't have. See season_stats.py's PITCHER START PA section
# for why this is a separate cascade, not a retrofit of build_expected_start_innings.

def build_pitcher_start_pa_this_season(
    pbp: pd.DataFrame, window: str | int = 'season'
) -> pd.DataFrame:
    """One row per (personId, gamepk) SP start: rolling avg batters faced
    per start and starts-so-far count, UP TO (not including) this start,
    shift(1) — same _rolling_sum engine and shape as
    build_pitcher_start_ip_this_season. Also carries pitcher_team_id for
    THIS start (not last season's team), so build_expected_batters_faced
    can look up the team fallback for wherever this pitcher currently
    plays, correctly following him through a mid-season trade."""

    sp_pbp = pbp[pbp['pitcher_role'] == 'sp']
    last_pitch = sp_pbp[
        sp_pbp['pitch_number'] == sp_pbp.groupby(['gamepk', 'play_id'])['pitch_number'].transform('max')
    ]
    per_start = (
        last_pitch
        .groupby(['pitcher_id', 'pitcher_team_id', 'gamepk', 'game_date', 'game_datetime', 'game_season'])
        .agg(pa_total=('play_result', 'count'))
        .reset_index()
        .rename(columns={'pitcher_id': 'personId'})
    )
    per_start['starts_n'] = 1

    rolled = _rolling_sum(
        per_start, entity_col='personId', cols=['pa_total', 'starts_n'], window=window, sort_col='game_datetime'
    )
    rolled['avg_pa_per_start'] = rolled['pa_total'] / rolled['starts_n'].replace(0, np.nan)
    rolled['starts_n'] = rolled['starts_n'].fillna(0)

    key_cols = ['personId', 'pitcher_team_id', 'gamepk', 'game_date', 'game_datetime', 'game_season']
    rolled = rolled[key_cols + ['starts_n', 'avg_pa_per_start']]

    prefix = 'pitcher_this_season_start_pa_' if window == 'season' else f'pitcher_last{window}_start_pa_'
    return _prefix_stat_cols(rolled, prefix=prefix, key_cols=key_cols)


def build_pitcher_rest_days(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (personId, gamepk) SP start: calendar days since that
    pitcher's own previous SP start (any team, following him through a
    trade), NaN for his first SP start in the data. Same shape as
    build_team_rest_days (sort by game_datetime, .diff() per entity) but
    keyed on the pitcher instead of the team, and scoped to
    pitcher_role == 'sp' — a relief appearance in between two starts is not
    "his last start."
    """
    sp_pbp = pbp[pbp['pitcher_role'] == 'sp']
    per_start = (
        sp_pbp
        .groupby(['pitcher_id', 'gamepk', 'game_date', 'game_datetime', 'game_season'])
        .size()
        .reset_index(name='_n')
        .drop(columns='_n')
        .rename(columns={'pitcher_id': 'personId'})
    )

    per_start = per_start.sort_values(['personId', 'game_datetime']).reset_index(drop=True)
    per_start['pitcher_days_since_last_start'] = (
        per_start.groupby('personId')['game_date'].diff().dt.days
    )

    return per_start[['personId', 'gamepk', 'game_date', 'game_datetime', 'game_season', 'pitcher_days_since_last_start']]


def build_pitcher_workload_density(
    pitcher_boxscore: pd.DataFrame, pbp: pd.DataFrame, rest_days: pd.DataFrame
) -> pd.DataFrame:
    """One row per (personId, gamepk) SP start: pitcher_last_start_pitches
    (that pitcher's own previous SP start's pitch count, shift(1) — NaN for
    his first SP start) and pitcher_workload_density = pitches / rest days
    since that start (from build_pitcher_rest_days's output, passed in —
    composition, not recomputed here), NaN-guarded (not inf) when rest days
    is 0 or NaN.

    Pitches thrown, not batters faced: pitch count is the actual lever
    managers pull on for a pull decision, and batters-faced recency is
    already covered by the separate trailing-3-start PA trend feature.
    """
    role_lookup = pbp[pbp['pitcher_role'] == 'sp'][['pitcher_id', 'gamepk']].drop_duplicates().rename(
        columns={'pitcher_id': 'personId'}
    )
    sp_box = pitcher_boxscore.merge(role_lookup, on=['personId', 'gamepk'], how='inner')
    sp_box = sp_box.sort_values(['personId', 'game_datetime']).reset_index(drop=True)
    sp_box['pitcher_last_start_pitches'] = sp_box.groupby('personId')['p'].shift(1)

    result = sp_box[['personId', 'gamepk', 'pitcher_last_start_pitches']].merge(
        rest_days[['personId', 'gamepk', 'pitcher_days_since_last_start']], on=['personId', 'gamepk'], how='left',
    )
    result['pitcher_workload_density'] = (
        result['pitcher_last_start_pitches'] / result['pitcher_days_since_last_start'].replace(0, np.nan)
    )

    return result


def build_expected_batters_faced(
    pitcher_start_pa_last_season: pd.DataFrame,
    pitcher_start_pa_this_season: pd.DataFrame,
    team_avg_start_pa: pd.DataFrame,
    league_avg_start_pa: pd.DataFrame,
    k: float = 5.0,
) -> pd.DataFrame:
    """3-level shrinkage cascade for pre-game-knowable expected batters
    faced this start: baseline = pitcher's own last season -> his CURRENT
    team's last-season average -> full league last-season average (first
    non-null wins), then blended toward this season's own emerging average
    as starts accumulate — shrinkage_weight = starts_n / (starts_n + k),
    identical formula to build_expected_start_innings. See that function's
    own docstring and season_stats.py's PITCHER START PA section for why
    this is a deliberately separate function rather than a retrofit."""

    df = pitcher_start_pa_this_season.merge(
        pitcher_start_pa_last_season[[
            'personId', 'game_season',
            'pitcher_last_season_start_pa_avg_pa_per_start',
            'pitcher_last_season_start_pa_n_starts',
        ]],
        on=['personId', 'game_season'], how='left',
    )
    df = df.merge(team_avg_start_pa, on=['pitcher_team_id', 'game_season'], how='left')
    df = df.merge(league_avg_start_pa, on='game_season', how='left')

    baseline = (
        df['pitcher_last_season_start_pa_avg_pa_per_start']
        .fillna(df['team_last_season_avg_pa_per_start'])
        .fillna(df['league_last_season_avg_pa_per_start'])
        .fillna(df['pitcher_this_season_start_pa_avg_pa_per_start'])
    )

    starts_n = df['pitcher_this_season_start_pa_starts_n']
    weight = starts_n / (starts_n + k)
    this_season_avg_safe = df['pitcher_this_season_start_pa_avg_pa_per_start'].fillna(0)

    df['expected_batters_faced_weight'] = weight
    df['expected_batters_faced'] = (1 - weight) * baseline + weight * this_season_avg_safe

    return df


def build_pitcher_anomaly_count_this_season(
    start_df: pd.DataFrame, anomaly_threshold: float = 0.6, min_weight: float = 0.3,
) -> pd.DataFrame:
    """One row per (personId, gamepk) SP start: pitcher_anomaly_count_this_season
    — a point-in-time-safe cumulative count of this pitcher's STRICTLY PRIOR
    starts this season where realized_batters_faced came in well under the
    pre-game expected_batters_faced (batters_faced_predictor's residual
    analysis found established starters getting pulled after 0-2 clean
    innings, with no shared pre-game-knowable trigger — rest days, weather,
    day-of-week, and opponent quality were all checked and ruled out — so
    this tracks THAT a pitcher has already shown the pattern this season,
    not WHY, as a base-rate/recency signal).

    A start is "anomalous" if realized_batters_faced < anomaly_threshold *
    expected_batters_faced AND expected_batters_faced_weight >= min_weight —
    the weight gate keeps a cold-start pitcher's still-unreliable cascade
    estimate (already a separate, closed problem — see
    batters_faced_predictor's v3) from being spuriously flagged as anomalous.

    Requires personId, gamepk, game_season, game_date, realized_batters_faced,
    expected_batters_faced, expected_batters_faced_weight already merged onto
    start_df (batters_faced_predictor's own create_start_pa_outcome + the
    expected_pa frame from build_expected_batters_faced, composed by the
    caller — not recomputed here).
    """
    df = start_df.sort_values(['personId', 'game_season', 'game_date']).copy()

    is_anomalous = (
        (df['realized_batters_faced'] < anomaly_threshold * df['expected_batters_faced'])
        & (df['expected_batters_faced_weight'] >= min_weight)
    ).astype(float)

    df['pitcher_anomaly_count_this_season'] = (
        is_anomalous.groupby([df['personId'], df['game_season']]).transform(lambda s: s.shift(1).fillna(0).cumsum())
    )

    return df[['personId', 'gamepk', 'pitcher_anomaly_count_this_season']]


# --------------------------- LONG EXPANSION (Epic E, replaces expected_role.py) --------------------------- #

def build_pitcher_role_by_inning(team_game: pd.DataFrame, max_innings: int = 9) -> pd.DataFrame:
    """Expand one row per (team_id, gamepk) — carrying an expected_start_innings
    estimate — into one row per inning (1..max_innings): 'sp' for the starter's
    expected innings (rounded UP — a partial inning pitched still counts as that
    inning belonging to the starter), 'bullpen' for the remainder. Clipped to
    [0, max_innings] so an implausibly high estimate can't create a 10th inning
    or leave every inning as 'sp' with no bullpen rows.

    A missing estimate (NaN — shouldn't happen given build_expected_start_innings'
    league-wide fallback, but not asserted here) degrades to every inning being
    'bullpen': NaN comparisons are always False, so no inning ever qualifies as 'sp'.

    This is the direct, inning-based replacement for expected_role.py's PA-
    position-based sp/bullpen gating — see FEATURE_GLOSSARY.md. Swapping the
    v2/v3 training pipeline over to it is a separate migration, not bundled here.
    """
    df = team_game.copy()
    df['sp_innings'] = np.ceil(df['expected_start_innings']).clip(lower=0, upper=max_innings)

    innings = pd.DataFrame({'inning': range(1, max_innings + 1)})
    expanded = df.merge(innings, how='cross')

    expanded['pitcher_role'] = np.where(expanded['inning'] <= expanded['sp_innings'], 'sp', 'bullpen')

    return expanded.drop(columns=['sp_innings'])


# --------------------------- BATTER SLOT EXPANSION (count-distribution check) --------------------------- #
# Same expansion shape as build_pitcher_role_by_inning above (cross join a pregame
# estimate against a fixed range, then tag/filter), but on the batter-lineup axis
# instead of innings — feeds k_predictor's total-strikeout count-distribution check.

def build_batter_slot_expansion(
    pitcher_starts: pd.DataFrame, batting_order: pd.DataFrame, max_slots: int = 45
) -> pd.DataFrame:
    """Expand one row per pitcher start (carrying expected_batters_faced) into
    one row per synthetic batter-slot 1..round(expected_batters_faced), cycling
    through the real 9-batter lineup order (slot 10 = lineup position 1 again,
    2nd time through). Each slot carries the specific real batter_id in that
    position (from batting_order — a realized historical lineup standing in
    for real pregame lineup data, which this pipeline doesn't ingest yet) and
    expected_times_through_order, capped at 3 — the same cap
    _add_pbp_times_through_order and expected_role.py already use everywhere
    else this concept appears, so the trained PA classifier sees the same
    feature shape it was trained on rather than an unseen value like 4 or 5.

    batting_order is deduplicated to one row per (gamepk, batting_order)
    before joining — real MLB data isn't always a clean 1:1 slot mapping (a
    substitution is sometimes logged sharing the SAME slot number as the
    starter it replaced, confirmed against real 2024 data: roughly half of
    all (gamepk, batting_order) pairs had 2 batter rows). Without this, the
    join below is on the slot NUMBER, not a real key — a collision fans out
    every downstream slot silently rather than erroring."""

    batting_order = batting_order.drop_duplicates(subset=['gamepk', 'batting_order'], keep='first')

    df = pitcher_starts.copy()
    df['n_slots'] = df['expected_batters_faced'].round().clip(lower=0, upper=max_slots)

    slots = pd.DataFrame({'slot': range(1, max_slots + 1)})
    expanded = df.merge(slots, how='cross')
    expanded = expanded[expanded['slot'] <= expanded['n_slots']].copy()

    expanded['lineup_position'] = ((expanded['slot'] - 1) % 9) + 1
    cycle = ((expanded['slot'] - 1) // 9) + 1
    expanded['expected_times_through_order'] = cycle.clip(upper=3)

    expanded = expanded.merge(
        batting_order.rename(columns={'batting_order': 'lineup_position'}),
        on=['gamepk', 'lineup_position'], how='left',
    )

    return expanded.drop(columns=['n_slots'])


# --------------------------- BATTERS-FACED RESIDUAL DISTRIBUTION --------------------------- #
# build_expected_batters_faced above is a fixed POINT estimate. k_predictor's
# count-distribution-check diagnostic showed the estimate's error scales with
# expected_batters_faced_weight (thin this-season sample = bigger miss) — these two
# functions turn that point estimate plus its known error-correlate into a real
# distribution: fit an empirical residual histogram per weight bin from historical
# starts (build_batters_faced_residual_bins), then shift/scatter that bin's
# histogram around a new start's own point estimate (build_batters_faced_distribution).
# Chosen over a parametric count distribution (no precedent in this codebase, and
# batters-faced is plausibly bimodal — most starts end near a normal pull point, a
# minority end early from injury/blowout/quick-hook) or full simulation (would need
# manager pull-decision and pitch-count/score-trajectory features this pipeline
# doesn't have). See ROADMAP.md's batters-faced-distribution plan.

def build_batters_faced_residual_bins(fit_starts: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Fit an empirical residual histogram per expected_batters_faced_weight
    bin from historical starts (realized_batters_faced known). Bin edges are
    extended to -inf/+inf at the outer boundaries so any future weight value
    (0..1 by construction, but not asserted here) always resolves to exactly
    one bin — same 'always resolves, never leaves a gap' contract
    build_expected_batters_faced's own fallback cascade guarantees.

    Returns long-format: weight_bin, weight_bin_lower, weight_bin_upper,
    residual (realized - round(expected), int), probability (sums to 1.0
    within each bin).

    Rows missing expected_batters_faced (build_expected_batters_faced's
    3-level cascade still returns NaN for a true first-ever start with no
    league fallback for that season yet — rare, but present in real data)
    are dropped before computing residuals: a NaN residual can't be
    assigned to any bin.
    """
    df = fit_starts.dropna(subset=['expected_batters_faced', 'realized_batters_faced']).copy()
    df['residual'] = (df['realized_batters_faced'] - df['expected_batters_faced'].round()).astype(int)

    bin_labels, edges = pd.qcut(
        df['expected_batters_faced_weight'], n_bins, labels=False, duplicates='drop', retbins=True
    )
    df['weight_bin'] = bin_labels
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf

    counts = df.groupby(['weight_bin', 'residual']).size().rename('n').reset_index()
    counts['probability'] = counts['n'] / counts.groupby('weight_bin')['n'].transform('sum')

    bin_edges = pd.DataFrame({
        'weight_bin': range(len(edges) - 1),
        'weight_bin_lower': edges[:-1],
        'weight_bin_upper': edges[1:],
    })

    return counts.merge(bin_edges, on='weight_bin')[
        ['weight_bin', 'weight_bin_lower', 'weight_bin_upper', 'residual', 'probability']
    ]


def build_batters_faced_distribution(
    expected_pa: pd.DataFrame, residual_bins: pd.DataFrame, max_slots: int = 45
) -> pd.DataFrame:
    """Apply build_batters_faced_residual_bins' fitted histograms to new
    starts: resolve each start's own weight_bin, shift that bin's residual
    pmf by the start's rounded expected_batters_faced, clip to [0, max_slots]
    (accumulating mass at the boundary, not overwriting it — a start can
    have multiple residuals collapse to the same clipped n), and scatter
    into a fixed-length pmf. Always returns a length-(max_slots+1) array
    per row, matching poisson_binomial_mixture_pmf's own fixed-length
    contract (the combinator this distribution feeds).
    """
    bins = residual_bins[['weight_bin', 'weight_bin_lower', 'weight_bin_upper']].drop_duplicates()

    def _pmf_for_row(weight, base):
        bin_row = bins[(bins['weight_bin_lower'] <= weight) & (weight < bins['weight_bin_upper'])].iloc[0]
        own = residual_bins[residual_bins['weight_bin'] == bin_row['weight_bin']]
        n_vals = (base + own['residual']).clip(0, max_slots)
        pmf = np.zeros(max_slots + 1)
        np.add.at(pmf, n_vals.to_numpy(), own['probability'].to_numpy())
        return pmf

    df = expected_pa.copy()
    df['batters_faced_pmf'] = [
        _pmf_for_row(w, round(e))
        for w, e in zip(df['expected_batters_faced_weight'], df['expected_batters_faced'])
    ]
    return df
