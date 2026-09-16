import pandas as pd

from preprocessing import create_batting_order


def _boxscore_row(**overrides):
    row = {"gamepk": "1", "personId": "100", "batting_order": 3}
    row.update(overrides)
    return row


def test_create_batting_order_drops_nulls_and_dedupes():
    """Moved here from hit_predictor's processing/pipeline.py (as
    _create_batting_order, 2026-09-15) — this is generic box-score cleanup
    with zero hit_predictor-specific business logic, and by the time of the
    move it was already depended on by 4+ models plus the feature store, so
    it belongs in the neutral shared layer both sides can depend on as
    peers, not inside one specific model's directory. A null batting_order
    means the player wasn't in that day's starting lineup (bench/DNP) and
    shouldn't produce a row; a duplicate (gamepk, personId) pair collapses
    to one row."""
    df = pd.DataFrame([
        _boxscore_row(personId="100", gamepk="1", batting_order=3),
        _boxscore_row(personId="100", gamepk="1", batting_order=3),  # duplicate
        _boxscore_row(personId="200", gamepk="1", batting_order=None),  # bench/DNP
    ])

    result = create_batting_order(df)

    assert set(result.columns) == {"gamepk", "batter_id", "batting_order"}
    assert len(result) == 1
    assert result.iloc[0]["batter_id"] == "100"
    assert result["batting_order"].isna().sum() == 0


def test_create_batting_order_renames_personid_to_batter_id():
    df = pd.DataFrame([_boxscore_row(personId="100", gamepk="1", batting_order=5)])

    result = create_batting_order(df)

    assert "batter_id" in result.columns
    assert "personId" not in result.columns
    assert result.iloc[0]["batter_id"] == "100"


def test_create_batting_order_casts_to_int():
    df = pd.DataFrame([_boxscore_row(batting_order=7.0)])

    result = create_batting_order(df)

    assert result.iloc[0]["batting_order"] == 7
    assert isinstance(result.iloc[0]["batting_order"], (int, pd.Int64Dtype().type)) or \
        pd.api.types.is_integer_dtype(result["batting_order"])
