from unittest.mock import patch, MagicMock
import pytest


class TestCheckQuota:
    def test_check_quota_no_flags_below_warn_threshold(self):
        with patch("odds_quota.get_monthly_usage", return_value=400):
            from odds_quota import check_quota
            result = check_quota(year="2026", month="04", limit=500, warn_threshold=0.9)
            assert result["warn"] is False
            assert result["stop"] is False
            assert result["current"] == 400

    def test_check_quota_warns_at_90_percent(self):
        with patch("odds_quota.get_monthly_usage", return_value=450):
            from odds_quota import check_quota
            result = check_quota(year="2026", month="04", limit=500, warn_threshold=0.9)
            assert result["warn"] is True
            assert result["stop"] is False

    def test_check_quota_stops_at_limit(self):
        with patch("odds_quota.get_monthly_usage", return_value=500):
            from odds_quota import check_quota
            result = check_quota(year="2026", month="04", limit=500, warn_threshold=0.9)
            assert result["warn"] is True
            assert result["stop"] is True

    def test_check_quota_stops_when_over_limit(self):
        with patch("odds_quota.get_monthly_usage", return_value=501):
            from odds_quota import check_quota
            result = check_quota(year="2026", month="04", limit=500, warn_threshold=0.9)
            assert result["stop"] is True


class TestSetMonthlyUsage:
    def test_set_monthly_usage_writes_given_count_to_ssm(self):
        mock_ssm = MagicMock()

        with patch("odds_quota.boto3.client", return_value=mock_ssm):
            from odds_quota import set_monthly_usage
            set_monthly_usage(year="2026", month="04", used_count=487)
            mock_ssm.put_parameter.assert_called_once_with(
                Name="/mlb/odds-api/requests-used/2026/04",
                Value="487",
                Type="String",
                Overwrite=True,
            )

    def test_set_monthly_usage_does_not_add_one(self):
        mock_ssm = MagicMock()

        with patch("odds_quota.boto3.client", return_value=mock_ssm):
            from odds_quota import set_monthly_usage
            set_monthly_usage(year="2026", month="04", used_count=0)
            mock_ssm.put_parameter.assert_called_once_with(
                Name="/mlb/odds-api/requests-used/2026/04",
                Value="0",
                Type="String",
                Overwrite=True,
            )


class TestGetMonthlyUsage:
    def test_get_monthly_usage_returns_count_from_ssm(self):
        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "42"}}

        with patch("odds_quota.boto3.client", return_value=mock_ssm):
            from odds_quota import get_monthly_usage
            result = get_monthly_usage(year="2026", month="04")
            assert result == 42
            mock_ssm.get_parameter.assert_called_once_with(
                Name="/mlb/odds-api/requests-used/2026/04"
            )

    def test_get_monthly_usage_returns_zero_when_parameter_not_found(self):
        from botocore.exceptions import ClientError

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.side_effect = ClientError(
            {"Error": {"Code": "ParameterNotFound", "Message": "not found"}},
            "GetParameter",
        )

        with patch("odds_quota.boto3.client", return_value=mock_ssm):
            from odds_quota import get_monthly_usage
            result = get_monthly_usage(year="2026", month="04")
            assert result == 0
