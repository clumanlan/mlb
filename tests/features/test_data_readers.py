from unittest.mock import patch

from data_readers import read_playbyplay


def test_read_playbyplay_passes_columns_through():
    """Real production incident (2026-09-18): reading the full prepared
    playbyplay table (70 columns, mostly wide Statcast float columns) for a
    whole season inside a Lambda OOM'd twice in a row (512MB then 2048MB,
    maxed out both times) before batter_pa_volume ever needed more than 4 of
    those columns. Column selection at read time cut one day's file from
    7.26MB to 0.79MB in-memory (measured directly against real S3 data) —
    callers that need only a few columns must be able to ask for them."""
    with patch("data_readers.wr") as mock_wr:
        read_playbyplay(2026, columns=["gamepk", "play_id", "batter_id", "play_result"])

    mock_wr.s3.read_parquet.assert_called_once_with(
        path="s3://mlbdk/processed_data/prepared/playbyplay/2026/",
        columns=["gamepk", "play_id", "batter_id", "play_result"],
    )


def test_read_playbyplay_defaults_to_all_columns():
    """Backward compatible: no columns argument reads everything, same as
    before this fix — some future caller may genuinely need the full
    Statcast width."""
    with patch("data_readers.wr") as mock_wr:
        read_playbyplay(2026)

    mock_wr.s3.read_parquet.assert_called_once_with(
        path="s3://mlbdk/processed_data/prepared/playbyplay/2026/",
        columns=None,
    )
