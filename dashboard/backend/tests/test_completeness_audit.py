import pandas as pd
import pytest

import completeness_audit


# ---------------------------------------------------------------------------
# diff_game_coverage — pure set-diff, no I/O
# ---------------------------------------------------------------------------

class TestDiffGameCoverage:
    def test_flags_missing_games_per_table(self):
        scheduled_pks = {1, 2, 3}
        table_pks = {
            "batter_boxscore": {1, 2},
            "pitcher_boxscore": {1, 2, 3},
            "playbyplay": {1, 2, 3},
        }

        result = completeness_audit.diff_game_coverage(scheduled_pks, table_pks)

        assert result["batter_boxscore"]["missing_count"] == 1
        assert result["batter_boxscore"]["missing_gamepks"] == [3]
        assert result["pitcher_boxscore"]["missing_count"] == 0
        assert result["pitcher_boxscore"]["missing_gamepks"] == []
        assert result["playbyplay"]["missing_count"] == 0

    def test_reports_complete_when_every_table_covers_every_scheduled_game(self):
        scheduled_pks = {1, 2}
        table_pks = {
            "batter_boxscore": {1, 2},
            "pitcher_boxscore": {1, 2},
        }

        result = completeness_audit.diff_game_coverage(scheduled_pks, table_pks)

        assert result["is_complete"] is True

    def test_reports_incomplete_when_any_table_is_missing_games(self):
        scheduled_pks = {1, 2}
        table_pks = {
            "batter_boxscore": {1, 2},
            "pitcher_boxscore": {1},
        }

        result = completeness_audit.diff_game_coverage(scheduled_pks, table_pks)

        assert result["is_complete"] is False

    def test_missing_gamepks_are_sorted(self):
        scheduled_pks = {1, 2, 3, 4}
        table_pks = {"batter_boxscore": {2}}

        result = completeness_audit.diff_game_coverage(scheduled_pks, table_pks)

        assert result["batter_boxscore"]["missing_gamepks"] == [1, 3, 4]

    def test_extra_games_in_table_not_in_schedule_do_not_count_as_missing(self):
        scheduled_pks = {1, 2}
        table_pks = {"batter_boxscore": {1, 2, 999}}

        result = completeness_audit.diff_game_coverage(scheduled_pks, table_pks)

        assert result["batter_boxscore"]["missing_count"] == 0

    def test_empty_schedule_is_trivially_complete(self):
        result = completeness_audit.diff_game_coverage(set(), {"batter_boxscore": set()})

        assert result["is_complete"] is True
        assert result["batter_boxscore"]["missing_count"] == 0


# ---------------------------------------------------------------------------
# run_season_completeness_audit — orchestration, S3 reads mocked
# ---------------------------------------------------------------------------

class TestRunSeasonCompletenessAudit:
    def _schedule_df(self):
        return pd.DataFrame([
            {"gamepk": 1, "game_type": "R", "game_date": "2026-04-01"},
            {"gamepk": 2, "game_type": "R", "game_date": "2026-04-01"},
            {"gamepk": 3, "game_type": "R", "game_date": "2026-04-02"},
            {"gamepk": 99, "game_type": "S", "game_date": "2026-03-01"},  # spring training, excluded
        ])

    def test_excludes_non_regular_season_games_from_schedule(self, monkeypatch):
        calls = {}

        def fake_read_season(bucket, prefix, columns=None):
            calls[prefix] = True
            if "schedule" in prefix:
                return self._schedule_df()
            return pd.DataFrame({"gamepk": [1, 2, 3]})

        monkeypatch.setattr(completeness_audit.s3_client, "read_s3_parquet_season", fake_read_season)

        result = completeness_audit.run_season_completeness_audit("mlbdk", "2026")

        assert result["total_scheduled_games"] == 3

    def test_returns_missing_games_with_dates_for_incomplete_table(self, monkeypatch):
        def fake_read_season(bucket, prefix, columns=None):
            if "schedule" in prefix:
                return self._schedule_df()
            if "batter_boxscore" in prefix:
                return pd.DataFrame({"gamepk": [1, 2]})  # missing gamepk 3
            return pd.DataFrame({"gamepk": [1, 2, 3]})

        monkeypatch.setattr(completeness_audit.s3_client, "read_s3_parquet_season", fake_read_season)

        result = completeness_audit.run_season_completeness_audit("mlbdk", "2026")

        bb = result["tables"]["batter_boxscore"]
        assert bb["missing_count"] == 1
        assert bb["missing_games"] == [{"gamepk": 3, "game_date": "2026-04-02"}]
        assert result["is_complete"] is False

    def test_returns_checked_at_timestamp_and_year(self, monkeypatch):
        def fake_read_season(bucket, prefix, columns=None):
            if "schedule" in prefix:
                return self._schedule_df()
            return pd.DataFrame({"gamepk": [1, 2, 3]})

        monkeypatch.setattr(completeness_audit.s3_client, "read_s3_parquet_season", fake_read_season)

        result = completeness_audit.run_season_completeness_audit("mlbdk", "2026")

        assert result["year"] == "2026"
        assert "checked_at" in result and result["checked_at"] is not None

    def test_empty_schedule_returns_complete_with_zero_games(self, monkeypatch):
        def fake_read_season(bucket, prefix, columns=None):
            if "schedule" in prefix:
                return pd.DataFrame(columns=["gamepk", "game_type", "game_date"])
            return pd.DataFrame(columns=["gamepk"])

        monkeypatch.setattr(completeness_audit.s3_client, "read_s3_parquet_season", fake_read_season)

        result = completeness_audit.run_season_completeness_audit("mlbdk", "2026")

        assert result["total_scheduled_games"] == 0
        assert result["is_complete"] is True
