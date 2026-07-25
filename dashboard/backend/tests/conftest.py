import sys
import os
from datetime import date, timedelta
import pytest

# Put dashboard/backend/ on sys.path so tests can import main, s3_client, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_addoption(parser):
    parser.addoption("--date", action="store", default=None, help="Game date YYYY-MM-DD")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: hits real AWS S3 — requires credentials in .env"
    )


@pytest.fixture
def run_date(request) -> str:
    """
    Date string (YYYY-MM-DD) used by integration tests.
    Pass --date YYYY-MM-DD on the command line to override; defaults to yesterday.
    """
    cli_date = request.config.getoption("--date", default=None)
    if cli_date:
        return cli_date
    return str(date.today() - timedelta(days=1))
