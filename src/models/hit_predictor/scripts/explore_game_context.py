"""
Exploration script for processing/features/game_context.py.

Pulls a slice of real schedule data from S3 for one team and prints the
resulting datetime/doubleheader features, so you can eyeball real values
while building out the rest of Epic A/E. Not part of the test suite or any
experiment train.py — run directly:

    PYTHONPATH=src python src/models/hit_predictor/scripts/explore_game_context.py
"""

import awswrangler as wr
import boto3
import pandas as pd

from models.hit_predictor.processing.features.game_context import (
    build_datetime_features,
    build_doubleheader_flag,
    build_team_win_loss_record,
    build_team_rest_days,
    build_pitcher_start_ip_this_season,
    build_expected_start_innings,
    build_pitcher_role_by_inning,
)
from models.hit_predictor.processing.features.season_stats import (
    build_pitcher_start_ip_stats,
    build_league_avg_start_ip,
)
from models.hit_predictor.processing.pipeline import _add_pbp_starting_pitcher

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)

BUCKET = 'mlbdk'
REGION = 'us-east-2'
SAMPLE_TEAM = 'New York Yankees'
SAMPLE_SEASON = 2024
# 4/13/2024 is a real split doubleheader (Yankees @ Guardians) — picked so
# is_doubleheader actually fires True in the printed output, not just an
# all-False example.
SAMPLE_START = '2024-04-10'
SAMPLE_END = '2024-04-16'

DATETIME_DOUBLEHEADER_COLS = [
    'gamepk', 'game_date', 'home_name', 'away_name', 'game_num',
    'doubleheader', 'is_doubleheader',
    'game_dt_year', 'game_dt_month', 'game_dt_day', 'game_dt_dayofweek',
    'game_dt_hour', 'game_dt_minute',
    'game_dt_is_month_end', 'game_dt_is_month_start',
    'game_dt_is_quarter_end', 'game_dt_is_quarter_start',
]

TEAM_RECORD_COLS = [
    'gamepk', 'game_date', 'team_days_since_last_game',
    'team_roll_season_win_n', 'team_roll_season_games_n', 'team_roll_season_win_pct',
    'team_roll_season_run_diff_avg',
    'team_roll_last10g_win_n', 'team_roll_last10g_games_n', 'team_roll_last10g_win_pct',
    'team_roll_last10g_run_diff_avg',
]


def load_raw_schedule(season: int) -> pd.DataFrame:
    """Raw (not processed_data) schedule — the 'doubleheader' code column
    isn't in processed_data/games/schedule's SCHEDULE_COLUMNS yet (see
    game_context.py's build_doubleheader_flag docstring). Read chunked and
    cast venue_id to str first: raw daily files have an inconsistent
    string/int64 venue_id type across dates that otherwise breaks the concat.
    """
    session = boto3.Session(region_name=REGION)
    frames = []
    for chunk in wr.s3.read_parquet(
        path=f's3://{BUCKET}/raw_data/games/schedule/{season}/',
        boto3_session=session, chunked=True,
    ):
        chunk['venue_id'] = chunk['venue_id'].astype(str)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def load_pbp_role_tags(season: int) -> pd.DataFrame:
    """Only the 4 columns needed to derive pitcher_role (sp/bullpen) — full
    play-by-play is pitch-level and much larger; this column-pruned read is
    ~120MB/season instead of the multi-GB full-table pulls that caused OOM
    issues earlier this session. Role itself is derived the same way
    pipeline.py does it (first pitcher to throw for a team in the game).
    """
    session = boto3.Session(region_name=REGION)
    pbp = wr.s3.read_parquet(
        path=f's3://{BUCKET}/processed_data/prepared/playbyplay/{season}/',
        boto3_session=session,
        columns=['gamepk', 'play_id', 'pitcher_id', 'pitcher_team_id'],
    )
    return _add_pbp_starting_pitcher(pbp)[['gamepk', 'pitcher_id', 'pitcher_team_id', 'pitcher_role']].drop_duplicates()


def load_pitcher_boxscore(season: int) -> pd.DataFrame:
    session = boto3.Session(region_name=REGION)
    return wr.s3.read_parquet(
        path=f's3://{BUCKET}/processed_data/prepared/pitcher_boxscore/{season}/', boto3_session=session,
    )


def load_processed_schedule(season: int) -> pd.DataFrame:
    """processed_data/games/schedule — clean, deduplicated (process_schedule
    resolves the same rescheduled-game duplicate-gamepk issue the raw-schedule
    doubleheader example surfaced), used for team record/rest since those only
    need columns SCHEDULE_COLUMNS already has (no 'doubleheader' code needed).
    """
    session = boto3.Session(region_name=REGION)
    raw = wr.s3.read_parquet(
        path=f's3://{BUCKET}/processed_data/games/schedule/{season}/', boto3_session=session,
    )
    return raw


def main() -> None:
    print(f"Loading raw schedule for {SAMPLE_SEASON} (datetime + doubleheader demo) ...")
    raw_schedule = load_raw_schedule(SAMPLE_SEASON)
    raw_schedule = raw_schedule.rename(columns={'game_id': 'gamepk'})
    raw_schedule['game_date'] = pd.to_datetime(raw_schedule['game_date'])
    # Raw schedule's game_datetime is a string (unlike processed_data/games/schedule,
    # where process_schedule() already converts it) — build_datetime_features expects
    # a real datetime column, so convert here rather than loosening its contract.
    raw_schedule['game_datetime'] = pd.to_datetime(raw_schedule['game_datetime'])

    team_week = raw_schedule[
        ((raw_schedule['home_name'] == SAMPLE_TEAM) | (raw_schedule['away_name'] == SAMPLE_TEAM))
        & raw_schedule['game_date'].between(SAMPLE_START, SAMPLE_END)
    ].copy()
    print(f"{len(team_week)} {SAMPLE_TEAM} games between {SAMPLE_START} and {SAMPLE_END}")

    dt_dh_result = build_datetime_features(team_week)
    dt_dh_result = build_doubleheader_flag(dt_dh_result)

    print("\n=== datetime + doubleheader features ===")
    print(dt_dh_result[DATETIME_DOUBLEHEADER_COLS].sort_values('game_date').to_string(index=False))

    print(f"\nLoading processed schedule for {SAMPLE_SEASON} (team record/rest demo) ...")
    processed_schedule = load_processed_schedule(SAMPLE_SEASON)

    team_id = processed_schedule.loc[processed_schedule['home_name'] == SAMPLE_TEAM, 'home_id'].iloc[0]
    team_full_season = processed_schedule[
        (processed_schedule['home_id'] == team_id) | (processed_schedule['away_id'] == team_id)
    ].copy()
    print(f"{len(team_full_season)} {SAMPLE_TEAM} games in {SAMPLE_SEASON} (full season, for accurate rolling record)")

    record = build_team_win_loss_record(team_full_season, window='season')
    record = record.merge(
        build_team_win_loss_record(team_full_season, window=10)[
            ['team_id', 'gamepk', 'team_roll_last10g_win_n', 'team_roll_last10g_games_n',
             'team_roll_last10g_win_pct', 'team_roll_last10g_run_diff_avg']
        ],
        on=['team_id', 'gamepk'], how='left',
    )
    record = record.merge(
        build_team_rest_days(team_full_season)[['team_id', 'gamepk', 'team_days_since_last_game']],
        on=['team_id', 'gamepk'], how='left',
    )

    team_record = record[
        (record['team_id'] == team_id) & record['game_date'].between(SAMPLE_START, SAMPLE_END)
    ]
    print("\n=== team win/loss record + rest days (season-to-date AND trailing 10g) ===")
    print(team_record[TEAM_RECORD_COLS].sort_values('game_date').to_string(index=False))

    demo_starter_innings_estimate(team_id, processed_schedule)


def demo_starter_innings_estimate(team_id, processed_schedule: pd.DataFrame) -> None:
    """Epic E2/E3: pick a real Yankees starter with multiple 2024 starts,
    show his expected_start_innings blend (last-season baseline x this-season
    rolling avg, shrinkage-weighted) for each of his starts, then expand one
    of those starts into the long sp/bullpen-by-inning table.
    """
    print(f"\nLoading 2023+2024 pitcher_boxscore and play-by-play role tags (for starter-innings demo) ...")
    boxscore_2023 = load_pitcher_boxscore(2023)
    boxscore_2024 = load_pitcher_boxscore(2024)
    pbp_role_2023 = load_pbp_role_tags(2023)
    pbp_role_2024 = load_pbp_role_tags(2024)

    team_starters_2024 = boxscore_2024[boxscore_2024['team_id'] == str(team_id)].merge(
        pbp_role_2024.rename(columns={'pitcher_id': 'personId'})[['gamepk', 'personId', 'pitcher_role']],
        on=['gamepk', 'personId'], how='left',
    )
    sp_start_counts = (
        team_starters_2024[team_starters_2024['pitcher_role'] == 'sp']
        .groupby(['personId', 'player_name']).size().sort_values(ascending=False)
    )
    sample_pitcher_id, sample_pitcher_name = sp_start_counts.index[0]
    print(f"Sample starter: {sample_pitcher_name} (personId={sample_pitcher_id}), "
          f"{sp_start_counts.iloc[0]} starts in 2024")

    last_season_ip = build_pitcher_start_ip_stats(boxscore_2023, pbp_role_2023.rename(columns={'pitcher_id': 'pitcher_id'}))
    league_avg_ip = build_league_avg_start_ip(last_season_ip)
    this_season_ip = build_pitcher_start_ip_this_season(boxscore_2024, pbp_role_2024)

    expected = build_expected_start_innings(last_season_ip, this_season_ip, league_avg_ip)
    pitcher_starts = expected[expected['personId'] == sample_pitcher_id].sort_values('game_date')

    print("\n=== expected_start_innings across the season (this pitcher's own starts) ===")
    print(pitcher_starts[[
        'gamepk', 'game_date',
        'pitcher_last_season_start_ip_avg_ip_per_start', 'pitcher_last_season_start_ip_n_starts',
        'pitcher_this_season_start_ip_avg_ip_per_start', 'pitcher_this_season_start_ip_starts_n',
        'league_last_season_avg_ip_per_start',
        'expected_start_innings_weight', 'expected_start_innings',
    ]].to_string(index=False))

    # Long expansion for a start in the back half of the season, once this-season
    # sample has accumulated (more illustrative than a season-opener, where
    # expected_start_innings trivially equals the last-season baseline).
    later_start = pitcher_starts[pitcher_starts['pitcher_this_season_start_ip_starts_n'] >= 5].iloc[0]
    team_game = pd.DataFrame([{
        'team_id': team_id, 'gamepk': later_start['gamepk'],
        'expected_start_innings': later_start['expected_start_innings'],
    }])
    long_table = build_pitcher_role_by_inning(team_game)

    print(f"\n=== long (inning-by-inning) expansion for gamepk={later_start['gamepk']} "
          f"(expected_start_innings={later_start['expected_start_innings']:.2f}) ===")
    print(long_table.to_string(index=False))


if __name__ == '__main__':
    main()
