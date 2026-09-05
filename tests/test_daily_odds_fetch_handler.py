from unittest.mock import patch, MagicMock
import pytest


class TestQuotaLimitConfig:
    def test_quota_limit_matches_current_20k_plan(self):
        """
        The Odds API plan was upgraded from 500 to 20,000 credits/month on 2026-08-27
        (see CLAUDE.md's Known Issues / Deferred Work), but this constant was never
        updated — the Lambda has been hard-stopping on the old 500 ceiling since usage
        crossed it on 2026-09-02, blocking odds fetches for four days straight even
        though the account had ~18,600 credits of real headroom left.
        """
        from handler import QUOTA_LIMIT
        assert QUOTA_LIMIT == 20000


class TestFetchAndStore:
    def test_fetch_and_store_writes_team_odds_to_s3(self):
        fake_team_odds = [{"id": "g1", "home_team": "Red Sox", "away_team": "Yankees", "bookmakers": []}]

        with patch("handler.get_team_odds", return_value=fake_team_odds) as mock_team, \
             patch("handler.get_all_player_props", return_value=[]), \
             patch("handler.wr.s3.to_parquet") as mock_write, \
             patch("handler.get_last_usage", return_value=None):
            from handler import fetch_and_store
            fetch_and_store("2026-04-30", api_key="test-key")
            mock_team.assert_called_once_with("2026-04-30", api_key="test-key")
            write_call = mock_write.call_args_list[0]
            assert write_call.kwargs["path"] == "s3://mlbdk/raw_data/odds/team_odds/2026/2026-04-30.parquet"

    def test_fetch_and_store_writes_player_props_to_s3(self):
        fake_props = [{"game": {"id": "g1"}, "props": {"bookmakers": []}}]

        with patch("handler.get_team_odds", return_value=[]), \
             patch("handler.get_all_player_props", return_value=fake_props) as mock_props, \
             patch("handler.wr.s3.to_parquet") as mock_write, \
             patch("handler.get_last_usage", return_value=None):
            from handler import fetch_and_store
            fetch_and_store("2026-04-30", api_key="test-key")
            mock_props.assert_called_once_with("2026-04-30", api_key="test-key")
            write_call = mock_write.call_args_list[1]
            assert write_call.kwargs["path"] == "s3://mlbdk/raw_data/odds/player_props/2026/2026-04-30.parquet"

    def test_fetch_and_store_sets_quota_from_real_api_usage(self):
        with patch("handler.get_team_odds", return_value=[]), \
             patch("handler.get_all_player_props", return_value=[]), \
             patch("handler.wr.s3.to_parquet"), \
             patch("handler.get_last_usage", return_value=487), \
             patch("handler.set_monthly_usage") as mock_set:
            from handler import fetch_and_store
            fetch_and_store("2026-04-30", api_key="test-key")
            mock_set.assert_called_once_with(year="2026", month="04", used_count=487)

    def test_fetch_and_store_logs_quota_usage(self, caplog):
        import logging
        with patch("handler.get_team_odds", return_value=[]), \
             patch("handler.get_all_player_props", return_value=[]), \
             patch("handler.wr.s3.to_parquet"), \
             patch("handler.get_last_usage", return_value=487), \
             patch("handler.set_monthly_usage"):
            from handler import fetch_and_store
            with caplog.at_level(logging.INFO):
                fetch_and_store("2026-04-30", api_key="test-key")
        assert any("Requests used" in r.message for r in caplog.records)

    def test_fetch_and_store_skips_quota_update_when_usage_unknown(self, caplog):
        import logging
        with patch("handler.get_team_odds", return_value=[]), \
             patch("handler.get_all_player_props", return_value=[]), \
             patch("handler.wr.s3.to_parquet"), \
             patch("handler.get_last_usage", return_value=None), \
             patch("handler.set_monthly_usage") as mock_set:
            from handler import fetch_and_store
            with caplog.at_level(logging.WARNING):
                fetch_and_store("2026-04-30", api_key="test-key")
        mock_set.assert_not_called()
        assert any("quota tracker not updated" in r.message.lower() for r in caplog.records)


class TestHandlerQuotaGate:
    def test_run_stops_when_quota_exceeded(self):
        with patch("handler.check_quota", return_value={"current": 500, "warn": True, "stop": True}), \
             patch("handler.get_api_key", return_value="test-key"):
            from handler import run
            with pytest.raises(Exception, match="quota"):
                run("2026-04-30")

    def test_run_logs_warning_when_near_quota(self, caplog):
        import logging
        with patch("handler.check_quota", return_value={"current": 450, "warn": True, "stop": False}), \
             patch("handler.get_api_key", return_value="test-key"), \
             patch("handler.fetch_and_store", return_value={"team_odds": 0, "player_props": 0}):
            from handler import run
            with caplog.at_level(logging.WARNING):
                run("2026-04-30")
        assert any("approaching limit" in r.message for r in caplog.records)

    def test_run_fetches_api_key_from_ssm(self):
        with patch("handler.check_quota", return_value={"current": 0, "warn": False, "stop": False}), \
             patch("handler.get_api_key", return_value="real-key") as mock_key, \
             patch("handler.fetch_and_store", return_value={"team_odds": 0, "player_props": 0}):
            from handler import run
            run("2026-04-30")
            mock_key.assert_called_once()


class TestHandlerEventParsing:
    def test_handler_uses_date_from_event(self):
        with patch("handler.run") as mock_run, \
             patch("handler.write_status"):
            from handler import handler
            handler({"date": "2026-04-29"}, {})
            mock_run.assert_called_once_with("2026-04-29")

    def test_handler_defaults_to_today_when_no_date(self):
        with patch("handler.run") as mock_run, \
             patch("handler.datetime") as mock_dt, \
             patch("handler.write_status"):
            mock_dt.utcnow.return_value.strftime.return_value = "2026-04-30"
            mock_dt.utcnow.return_value.isoformat.return_value = "2026-04-30T00:00:00"
            from handler import handler
            handler({}, {})
            mock_run.assert_called_once_with("2026-04-30")

    def test_handler_returns_200_on_success(self):
        with patch("handler.run"), \
             patch("handler.write_status"):
            from handler import handler
            result = handler({"date": "2026-04-29"}, {})
            assert result["statusCode"] == 200

    def test_handler_reraises_on_exception(self):
        with patch("handler.run", side_effect=Exception("boom")), \
             patch("handler.write_status"):
            from handler import handler
            with pytest.raises(Exception, match="boom"):
                handler({"date": "2026-04-29"}, {})


class TestHandlerWriteStatus:
    def test_handler_calls_write_status_on_success(self):
        with patch("handler.run", return_value={"team_odds": 8, "player_props": 8}), \
             patch("handler.write_status") as mock_ws:
            from handler import handler
            handler({"date": "2026-04-29"}, {})
        mock_ws.assert_called_once()
        call_kwargs = mock_ws.call_args.kwargs
        assert call_kwargs["status"] == "success"
        assert call_kwargs["error"] is None
        assert call_kwargs["function_name"] == "daily_odds_fetch"
        assert call_kwargs["run_date"] == "2026-04-29"

    def test_handler_calls_write_status_on_failure(self):
        with patch("handler.run", side_effect=Exception("quota exceeded")), \
             patch("handler.write_status") as mock_ws:
            from handler import handler
            with pytest.raises(Exception, match="quota exceeded"):
                handler({"date": "2026-04-29"}, {})
        mock_ws.assert_called_once()
        call_kwargs = mock_ws.call_args.kwargs
        assert call_kwargs["status"] == "failed"
        assert "quota exceeded" in call_kwargs["error"]

    def test_handler_write_status_games_processed_shape(self):
        games = {"team_odds": 8, "player_props": 8}
        with patch("handler.run", return_value=games), \
             patch("handler.write_status") as mock_ws:
            from handler import handler
            handler({"date": "2026-04-29"}, {})
        gp = mock_ws.call_args.kwargs["games_processed"]
        assert "team_odds" in gp
        assert "player_props" in gp
