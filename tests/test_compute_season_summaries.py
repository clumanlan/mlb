import sys
import os
import logging
import pytest
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/features/transforms")))

from compute_season_summaries import (
    compute_batter_season_summary,
    compute_sp_season_summary,
    compute_team_season_summary,
)

# Minimal DataFrames matching the DuckDB column requirements in each function

_BATTER = pd.DataFrame({
    'personId': [1], 'player_name': ['A'], 'game_season': [2025], 'game_type': ['R'],
    'plate_appearances': [4], 'ab': [3], 'h': [1], 'singles': [1], 'doubles': [0],
    'triples': [0], 'hr': [0], 'total_bases': [1], 'rbi': [0], 'bb': [1],
    'k': [1], 'sb': [0], 'lob': [2],
})

_PITCHER = pd.DataFrame({
    'gamepk': [1], 'team_id': [1], 'personId': [100], 'ip': [6.0],
    'game_season': [2025], 'game_type': ['R'],
    'h': [3], 'er': [1], 'bb': [1], 'k': [5], 'hr': [0],
})

_PBP = pd.DataFrame({
    'gamepk': [1], 'pitcher_team_id': [1], 'inning': [1],
    'event_index': [1], 'pitcher_id': [100], 'pitcher_name': ['P'],
})

_TEAM_BATTER = pd.DataFrame({
    'team_id': [1], 'game_season': [2025], 'game_type': ['R'], 'gamepk': [1],
    'plate_appearances': [30], 'ab': [27], 'h': [8], 'singles': [5], 'doubles': [2],
    'triples': [0], 'hr': [1], 'total_bases': [12], 'r': [4], 'rbi': [4],
    'bb': [3], 'k': [6], 'sb': [1], 'lob': [5],
})

_TEAM_PITCHER = pd.DataFrame({
    'team_id': [1], 'game_season': [2025], 'game_type': ['R'],
    'ip': [9.0], 'h': [5], 'er': [2], 'bb': [2], 'k': [8], 'hr': [1],
})


def test_compute_batter_season_summary_logs_output_shape(caplog):
    with patch("compute_season_summaries.read_batter_boxscore", return_value=_BATTER):
        with caplog.at_level(logging.INFO):
            compute_batter_season_summary("2025")
    assert any("Batter" in r.message and "shape" in r.message for r in caplog.records)


def test_compute_sp_season_summary_logs_output_shape(caplog):
    with patch("compute_season_summaries.read_pitcher_boxscore", return_value=_PITCHER), \
         patch("compute_season_summaries.read_playbyplay", return_value=_PBP):
        with caplog.at_level(logging.INFO):
            compute_sp_season_summary("2025")
    assert any("SP" in r.message and "shape" in r.message for r in caplog.records)


def test_compute_team_season_summary_logs_output_shape(caplog):
    with patch("compute_season_summaries.read_batter_boxscore", return_value=_TEAM_BATTER), \
         patch("compute_season_summaries.read_pitcher_boxscore", return_value=_TEAM_PITCHER):
        with caplog.at_level(logging.INFO):
            compute_team_season_summary("2025")
    assert any("Team" in r.message and "shape" in r.message for r in caplog.records)
