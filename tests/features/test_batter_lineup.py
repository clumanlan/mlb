import json
from unittest.mock import patch

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from batter_lineup import (
    _normalize_batter_boxscore_lineup,
    compute_batter_lineup_features,
    parse_lineup_payload,
    compute_batter_lineup_features_for_date,
)

BUCKET = "mlbdk"


def _boxscore_row(**overrides):
    row = {
        "personId": "100",
        "gamepk": "1",
        "batting_order": 3,
        "game_date": pd.Timestamp("2024-04-01"),
        "team_id": "10",
        "ab": 4,
    }
    row.update(overrides)
    return row


def test_normalize_batter_boxscore_lineup_drops_nulls_and_dedupes():
    """Point-in-time-irrelevant, but data-quality-relevant: a null
    batting_order means the player wasn't in that day's starting lineup
    (bench/DNP) and shouldn't produce a lineup-slot feature row at all.
    A duplicate (gamepk, personId) pair (e.g. a raw double-count) must
    collapse to one row, matching the shared create_batting_order
    (data/modules/preprocessing.py) convention."""
    df = pd.DataFrame([
        _boxscore_row(personId="100", gamepk="1", batting_order=3),
        _boxscore_row(personId="100", gamepk="1", batting_order=3),  # duplicate
        _boxscore_row(personId="200", gamepk="1", batting_order=None),  # bench/DNP
    ])

    result = _normalize_batter_boxscore_lineup(df)

    assert set(result.columns) == {"personId", "gamepk", "event_timestamp", "batting_order"}
    assert len(result) == 1
    assert result.iloc[0]["personId"] == "100"
    assert result["batting_order"].isna().sum() == 0
    assert result["batting_order"].dtype == "int64"


def test_normalize_batter_boxscore_lineup_delegates_to_create_batting_order():
    """Regression test for feature-definition duplication: the null-filter +
    dedup decision for batting_order must live in exactly one place — the
    shared create_batting_order (data/modules/preprocessing.py, moved out
    of hit_predictor 2026-09-15 so this feature-store layer and every
    consuming model are peers rather than this layer depending on one
    model's internals — see DECISIONS.md), which n_pa_predictor's own
    training pipeline already calls via processing/pipeline.py::
    build_batter_game_frame. This store-facing normalizer must delegate to
    it rather than reimplementing the same filtering logic a second time,
    or a future change to the canonical rule (e.g. the dedup key) would
    silently stop applying to the feature this store serves."""
    df = pd.DataFrame([_boxscore_row(personId="100", gamepk="1", batting_order=3)])
    fake_create_result = pd.DataFrame([{"gamepk": "1", "batter_id": "100", "batting_order": 3}])

    with patch(
        "batter_lineup.create_batting_order", return_value=fake_create_result,
    ) as mock_create:
        result = _normalize_batter_boxscore_lineup(df)

    mock_create.assert_called_once()
    pd.testing.assert_frame_equal(mock_create.call_args[0][0], df)
    assert result.iloc[0]["personId"] == "100"
    assert result.iloc[0]["batting_order"] == 3
    assert result.iloc[0]["event_timestamp"] == pd.Timestamp("2024-04-01")


def test_normalize_batter_boxscore_lineup_keeps_personid_not_batter_id():
    """The Feast `player` entity's join key is personId — this pipeline
    must NOT rename it to batter_id the way the shared create_batting_order
    does for its own unrelated purposes."""
    df = pd.DataFrame([_boxscore_row(personId="100", gamepk="1", batting_order=5)])

    result = _normalize_batter_boxscore_lineup(df)

    assert "personId" in result.columns
    assert "batter_id" not in result.columns


def test_compute_batter_lineup_features_calls_reader_and_normalizes():
    """Thin wrapper contract: whatever read_batter_boxscore(year) returns,
    the wrapper's output equals applying the pure normalizer to it —
    exercising the reader-call + normalize composition, not new logic."""
    fixture = pd.DataFrame([
        _boxscore_row(personId="300", gamepk="9", batting_order=7),
        _boxscore_row(personId="400", gamepk="9", batting_order=None),
    ])

    with patch("batter_lineup.read_batter_boxscore", return_value=fixture) as mock_reader:
        result = compute_batter_lineup_features("2024")

    mock_reader.assert_called_once_with("2024")
    expected = _normalize_batter_boxscore_lineup(fixture)
    pd.testing.assert_frame_equal(result, expected)


def _lineup_payload(**overrides):
    """Shaped exactly like daily_lineup_fetch/handler.py's build_lineup_payload output."""
    payload = {
        "game_pk": 999,
        "date": "2024-04-01",
        "confirmed_at": "2024-04-01T22:00:00Z",
        "home_team_id": 10,
        "away_team_id": 20,
        "home_lineup": [{"batting_order": 1, "player_id": "100"}],
        "away_lineup": [{"batting_order": 1, "player_id": "200"}],
        "home_pitcher_id": 50,
        "away_pitcher_id": 60,
    }
    payload.update(overrides)
    return payload


def test_parse_lineup_payload_flattens_and_renames():
    """home_lineup and away_lineup combine into one row each; player_id
    (daily_lineup_fetch's own naming) renames to personId so it matches
    the Feast `player` entity's join key, same as the backfill side."""
    payload = _lineup_payload()

    result = parse_lineup_payload(payload)

    assert set(result.columns) == {"personId", "gamepk", "event_timestamp", "batting_order"}
    assert len(result) == 2
    assert set(result["personId"]) == {"100", "200"}
    assert (result["gamepk"] == "999").all()
    assert (result["event_timestamp"] == "2024-04-01").all()
    assert result["batting_order"].dtype == "int64"


@pytest.fixture()
def aws_infra(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
        )
        yield s3


def test_compute_batter_lineup_features_for_date_combines_all_games_that_day(aws_infra):
    """Lists every {game_pk}.json under raw_data/games/lineups/{year}/{date}/
    and combines them — a real day has multiple games, not one."""
    s3 = aws_infra
    date = "2024-04-01"
    for payload in (_lineup_payload(game_pk=111), _lineup_payload(game_pk=222)):
        s3.put_object(
            Bucket=BUCKET,
            Key=f"raw_data/games/lineups/2024/{date}/{payload['game_pk']}.json",
            Body=json.dumps(payload),
        )

    result = compute_batter_lineup_features_for_date(date)

    assert set(result["gamepk"]) == {"111", "222"}
    assert len(result) == 4  # 2 games x 2 players each


def test_backfill_and_incremental_produce_identical_schema():
    """The entire reason this pipeline exists: training (backfill, box
    scores) and serving (incremental, announced lineups) must agree on
    exactly what a 'batting_order feature' looks like, or a model trained
    on one shape and served the other silently breaks."""
    backfill_result = _normalize_batter_boxscore_lineup(
        pd.DataFrame([_boxscore_row(personId="1", gamepk="1", batting_order=2)])
    )
    incremental_result = parse_lineup_payload(_lineup_payload())

    assert list(backfill_result.columns) == list(incremental_result.columns)
    assert dict(backfill_result.dtypes) == dict(incremental_result.dtypes)


def test_backfill_and_incremental_agree_on_batting_order_value():
    """Value consistency, not just schema (CLAUDE.md's promotion checklist,
    added 2026-09-15 after the schema-only test above was found insufficient
    on its own — same gap Chronon/Nubank's train-serve-skew research warns
    about). A realized box score row and an announced lineup entry
    describing the SAME real game/player/slot must normalize to the same
    batting_order — these are two independent data sources (post-game box
    score vs. pre-game announced lineup) describing one real-world fact, so
    this asserts they agree on it, not that one derives from the other."""
    boxscore_row = _boxscore_row(personId="100", gamepk="1", batting_order=5)
    lineup_payload = _lineup_payload(
        game_pk=1, date="2024-04-01",
        home_lineup=[{"batting_order": 5, "player_id": "100"}],
        away_lineup=[],
    )

    backfill_row = _normalize_batter_boxscore_lineup(pd.DataFrame([boxscore_row])).iloc[0]
    incremental_row = parse_lineup_payload(lineup_payload).iloc[0]

    assert backfill_row["batting_order"] == incremental_row["batting_order"]
    assert backfill_row["gamepk"] == incremental_row["gamepk"]
    assert backfill_row["event_timestamp"] == incremental_row["event_timestamp"]
