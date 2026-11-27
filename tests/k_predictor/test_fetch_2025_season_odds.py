import pandas as pd

from models.k_predictor.backtest.fetch_2025_season_odds import (
    dates_to_fetch,
    day_cost,
    expected_games_by_date,
    has_budget_for,
    listing_matches_schedule,
    PROP_COLUMNS,
    s3_key_for,
    with_prop_schema,
    saved_dates_from_keys,
)


def _sched(rows):
    return pd.DataFrame(rows, columns=["game_id", "game_date", "game_type", "status"])


class TestExpectedGamesByDate:
    def test_counts_only_completed_regular_season_games(self):
        sched = _sched([
            (1, "2025-04-01", "R", "Final"),
            (2, "2025-04-01", "R", "Completed Early: Rain"),
            (3, "2025-04-01", "R", "Postponed"),
            (4, "2025-04-01", "S", "Final"),  # spring training
            (5, "2025-04-02", "R", "Final"),
        ])
        assert expected_games_by_date(sched) == {"2025-04-01": 2, "2025-04-02": 1}

    def test_dedupes_a_game_seen_in_multiple_daily_files(self):
        sched = _sched([(1, "2025-04-01", "R", "Final"), (1, "2025-04-01", "R", "Final")])
        assert expected_games_by_date(sched) == {"2025-04-01": 1}


class TestDatesToFetch:
    def test_chronological_and_skips_already_saved(self):
        expected = {"2025-04-03": 15, "2025-04-01": 15, "2025-04-02": 15}
        assert dates_to_fetch(expected, already_saved={"2025-04-02"}) == ["2025-04-01", "2025-04-03"]


class TestBudget:
    def test_day_cost_is_listing_plus_one_call_per_game(self):
        assert day_cost(15) == 160

    def test_stops_when_day_would_dip_into_reserve(self):
        assert has_budget_for(n_games=15, used=19_300, limit=20_000, reserve=500) is True   # 700 left, needs 160+500
        assert has_budget_for(n_games=15, used=19_400, limit=20_000, reserve=500) is False  # 600 left


class TestListingMatchesSchedule:
    def test_exact_match(self):
        assert listing_matches_schedule(n_listed=15, n_expected=15) is True

    def test_one_short_tolerated(self):
        # a single game can legitimately be missing from the book's listing
        assert listing_matches_schedule(n_listed=14, n_expected=15) is True

    def test_bigger_gap_flags_the_day(self):
        # the 2026-08-31 tell was 2/15; the UTC-window bug produced 8/15
        assert listing_matches_schedule(n_listed=8, n_expected=15) is False


class TestS3Layout:
    def test_key_follows_repo_date_partition_convention(self):
        assert s3_key_for("2025-04-09") == "raw_data/odds/historical/pitcher_strikeouts/2025/2025-04-09.parquet"

    def test_saved_dates_parsed_from_listed_keys(self):
        # S3 (not a local folder) is the record of what's already been paid
        # for, so a rerun from any machine never re-buys a saved date.
        keys = [
            "s3://mlbdk/raw_data/odds/historical/pitcher_strikeouts/2025/2025-04-09.parquet",
            "s3://mlbdk/raw_data/odds/historical/pitcher_strikeouts/2025/2025-04-10.parquet",
        ]
        assert saved_dates_from_keys(keys) == {"2025-04-09", "2025-04-10"}


class TestWithPropSchema:
    def test_zero_prop_day_still_has_columns(self):
        # A day with no K lines must still be writable (awswrangler rejects a
        # column-less frame) so it's recorded as paid-for and not re-bought.
        out = with_prop_schema(pd.DataFrame([]))
        assert list(out.columns) == PROP_COLUMNS and len(out) == 0

    def test_nonempty_frame_unchanged(self):
        df = pd.DataFrame([{c: 1 for c in PROP_COLUMNS}])
        assert with_prop_schema(df).equals(df)
