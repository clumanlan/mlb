import boto3
import pytest
from moto import mock_aws
from unittest.mock import MagicMock, patch
import awswrangler as wr

BUCKET = "mlbdk"
DATE = "2026-04-30"
YEAR = "2026"
MONTH = "04"

FAKE_GAMES = [
    {
        "id": "game-abc",
        "away_team": "New York Yankees",
        "home_team": "Boston Red Sox",
        "commence_time": f"{DATE}T17:05:00Z",
    }
]
FAKE_TEAM_ODDS = [
    {
        "id": "game-abc",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "bookmakers": [{"key": "draftkings", "markets": []}],
    }
]
FAKE_PROPS = {
    "id": "game-abc",
    "bookmakers": [{"key": "draftkings", "markets": []}],
}


def _api_response(data):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = data
    mock.headers = {"x-requests-remaining": "490"}
    return mock


def _route_request(url, params=None):
    if "events/" in url:
        return _api_response(FAKE_PROPS)
    if params and "spreads" in params.get("markets", ""):
        return _api_response(FAKE_TEAM_ODDS)
    return _api_response(FAKE_GAMES)


@pytest.fixture()
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "test")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test")


@pytest.fixture()
def aws_infra(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
        )
        ssm = boto3.client("ssm", region_name="us-east-2")
        ssm.put_parameter(
            Name="/mlb/odds-api/api-key",
            Value="test-api-key",
            Type="SecureString",
        )
        yield s3, ssm


class TestHandlerEndToEnd:
    def test_writes_team_odds_parquet_to_s3(self, aws_infra):
        s3, _ = aws_infra
        with patch("odds_fetch.requests.get", side_effect=_route_request):
            from handler import handler
            handler({"date": DATE}, {})

        objects = s3.list_objects_v2(
            Bucket=BUCKET, Prefix=f"raw_data/odds/team_odds/{YEAR}/"
        )
        assert objects["KeyCount"] == 1
        assert objects["Contents"][0]["Key"] == f"raw_data/odds/team_odds/{YEAR}/{DATE}.parquet"

    def test_team_odds_parquet_contains_expected_rows(self, aws_infra):
        _, _ = aws_infra
        with patch("odds_fetch.requests.get", side_effect=_route_request):
            from handler import handler
            handler({"date": DATE}, {})

        df = wr.s3.read_parquet(f"s3://{BUCKET}/raw_data/odds/team_odds/{YEAR}/{DATE}.parquet")
        assert len(df) == 1
        assert df.iloc[0]["id"] == "game-abc"
        assert df.iloc[0]["home_team"] == "Boston Red Sox"

    def test_writes_player_props_parquet_to_s3(self, aws_infra):
        s3, _ = aws_infra
        with patch("odds_fetch.requests.get", side_effect=_route_request):
            from handler import handler
            handler({"date": DATE}, {})

        objects = s3.list_objects_v2(
            Bucket=BUCKET, Prefix=f"raw_data/odds/player_props/{YEAR}/"
        )
        assert objects["KeyCount"] == 1
        assert objects["Contents"][0]["Key"] == f"raw_data/odds/player_props/{YEAR}/{DATE}.parquet"

    def test_player_props_parquet_contains_one_row_per_game(self, aws_infra):
        _, _ = aws_infra
        with patch("odds_fetch.requests.get", side_effect=_route_request):
            from handler import handler
            handler({"date": DATE}, {})

        df = wr.s3.read_parquet(f"s3://{BUCKET}/raw_data/odds/player_props/{YEAR}/{DATE}.parquet")
        assert len(df) == len(FAKE_GAMES)

    def test_quota_counter_increments_from_zero(self, aws_infra):
        _, ssm = aws_infra
        with patch("odds_fetch.requests.get", side_effect=_route_request):
            from handler import handler
            handler({"date": DATE}, {})

        response = ssm.get_parameter(
            Name=f"/mlb/odds-api/requests-used/{YEAR}/{MONTH}"
        )
        assert response["Parameter"]["Value"] == "1"

    def test_quota_counter_increments_from_existing_value(self, aws_infra):
        _, ssm = aws_infra
        ssm.put_parameter(
            Name=f"/mlb/odds-api/requests-used/{YEAR}/{MONTH}",
            Value="41",
            Type="String",
        )
        with patch("odds_fetch.requests.get", side_effect=_route_request):
            from handler import handler
            handler({"date": DATE}, {})

        response = ssm.get_parameter(
            Name=f"/mlb/odds-api/requests-used/{YEAR}/{MONTH}"
        )
        assert response["Parameter"]["Value"] == "42"

    def test_handler_returns_200(self, aws_infra):
        with patch("odds_fetch.requests.get", side_effect=_route_request):
            from handler import handler
            result = handler({"date": DATE}, {})
        assert result["statusCode"] == 200

    def test_handler_returns_500_when_quota_exceeded(self, aws_infra):
        _, ssm = aws_infra
        ssm.put_parameter(
            Name=f"/mlb/odds-api/requests-used/{YEAR}/{MONTH}",
            Value="500",
            Type="String",
        )
        from handler import handler
        result = handler({"date": DATE}, {})
        assert result["statusCode"] == 500
        assert "quota" in result["body"].lower()
