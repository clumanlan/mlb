from unittest.mock import patch

import pandas as pd

from batter_pa_volume import (
    _append_phantom_rows,
    _normalize_batter_pa_rolling,
    compute_batter_pa_volume_features,
    compute_batter_pa_volume_features_for_date,
)


def test_normalize_batter_pa_rolling_renames_to_store_schema():
    """Output of build_batter_pa_rolling_stats (batter_id, gamepk, game_date,
    game_season, batter_n_pa_roll_season_games_n,
    batter_n_pa_roll_season_avg_n_pa_per_game) must map onto this store's
    naming: personId (not batter_id), event_timestamp (not game_date), and
    the cadence-bearing column names this repo's own convention uses."""
    rolled = pd.DataFrame([{
        "batter_id": "100",
        "gamepk": "1",
        "game_date": pd.Timestamp("2024-04-05"),
        "game_season": 2024,
        "batter_n_pa_roll_season_games_n": 3,
        "batter_n_pa_roll_season_avg_n_pa_per_game": 4.0,
    }])

    result = _normalize_batter_pa_rolling(rolled)

    assert set(result.columns) == {
        "personId", "gamepk", "event_timestamp",
        "batter_pa_roll_season_games_n", "batter_pa_roll_season_avg_n_pa_per_game",
    }
    assert result.iloc[0]["personId"] == "100"
    assert result.iloc[0]["event_timestamp"] == pd.Timestamp("2024-04-05")
    assert result.iloc[0]["batter_pa_roll_season_games_n"] == 3
    assert result["batter_pa_roll_season_games_n"].dtype == "int64"


def _pbp_row(**overrides):
    row = {"gamepk": "1", "play_id": "p1", "batter_id": "1", "play_result": "Single"}
    row.update(overrides)
    return row


def test_compute_batter_pa_volume_features_rolls_prior_games_only():
    """End-to-end through the REAL build_batter_game_frame and
    build_batter_pa_rolling_stats (not mocked — already trusted) with only
    the raw S3 readers mocked. Batter '1' plays 2 games: game 1 has 2 PAs,
    game 2 has 3 PAs. Game 2's rolling value must reflect only game 1
    (point-in-time safety), not include game 2's own PAs."""
    pbp = pd.DataFrame([
        _pbp_row(gamepk="1", play_id="g1p1", play_result="Single"),
        _pbp_row(gamepk="1", play_id="g1p2", play_result="Strikeout"),
        _pbp_row(gamepk="2", play_id="g2p1", play_result="Single"),
        _pbp_row(gamepk="2", play_id="g2p2", play_result="Single"),
        _pbp_row(gamepk="2", play_id="g2p3", play_result="Strikeout"),
    ])
    batter_boxscore = pd.DataFrame([
        {"gamepk": "1", "personId": "1", "batting_order": 3},
        {"gamepk": "2", "personId": "1", "batting_order": 3},
    ])
    schedule = pd.DataFrame([
        {"gamepk": "1", "game_date": pd.Timestamp("2024-04-01")},
        {"gamepk": "2", "game_date": pd.Timestamp("2024-04-02")},
    ])

    with patch("batter_pa_volume.read_playbyplay", return_value=pbp), \
         patch("batter_pa_volume.read_batter_boxscore", return_value=batter_boxscore), \
         patch("batter_pa_volume.read_schedule", return_value=schedule):
        result = compute_batter_pa_volume_features("2024")

    game2_row = result[result["gamepk"] == "2"].iloc[0]
    assert game2_row["personId"] == "1"
    assert game2_row["batter_pa_roll_season_games_n"] == 1
    assert game2_row["batter_pa_roll_season_avg_n_pa_per_game"] == 2.0

    game1_row = result[result["gamepk"] == "1"].iloc[0]
    assert game1_row["batter_pa_roll_season_games_n"] == 0
    assert pd.isna(game1_row["batter_pa_roll_season_avg_n_pa_per_game"])


def test_append_phantom_rows_adds_one_unknown_row_per_todays_player():
    """today_players is literally compute_batter_lineup_features_for_date's own output
    shape (personId, gamepk, event_timestamp) — reused directly, not a new
    lookup. A brand-new batter with zero history must still get a phantom
    row (min_periods=1 handles the no-prior-games case downstream)."""
    batter_game = pd.DataFrame([
        {"batter_id": "1", "gamepk": "1", "game_date": pd.Timestamp("2024-04-01"),
         "game_season": 2024, "n_pa": 4},
        {"batter_id": "2", "gamepk": "1", "game_date": pd.Timestamp("2024-04-01"),
         "game_season": 2024, "n_pa": 3},
    ])
    today_players = pd.DataFrame([
        {"personId": "1", "gamepk": "9", "event_timestamp": pd.Timestamp("2024-04-10"), "batting_order": 3},
        {"personId": "2", "gamepk": "9", "event_timestamp": pd.Timestamp("2024-04-10"), "batting_order": 4},
        {"personId": "3", "gamepk": "10", "event_timestamp": pd.Timestamp("2024-04-10"), "batting_order": 1},
    ])

    result = _append_phantom_rows(batter_game, today_players)

    assert len(result) == 5  # 2 original + 3 phantom
    phantoms = result[result["game_date"] == pd.Timestamp("2024-04-10")]
    assert len(phantoms) == 3
    assert set(phantoms["batter_id"]) == {"1", "2", "3"}
    assert phantoms["n_pa"].isna().all()
    assert (phantoms["game_season"] == 2024).all()
    # originals untouched
    assert (result[result["gamepk"] == "1"]["n_pa"].dropna() == [4, 3]).all()


def test_compute_batter_pa_volume_features_for_date_uses_lineup_to_know_who_plays():
    """Incremental mode end-to-end: today's players come from
    compute_batter_lineup_features_for_date, history from the same raw readers as
    backfill. The batter's snapshot must reflect only prior games."""
    today_players = pd.DataFrame([
        {"personId": "1", "gamepk": "9", "event_timestamp": pd.Timestamp("2024-04-10"), "batting_order": 3},
    ])
    pbp = pd.DataFrame([
        _pbp_row(gamepk="1", play_id="g1p1", play_result="Single"),
        _pbp_row(gamepk="1", play_id="g1p2", play_result="Single"),
        _pbp_row(gamepk="1", play_id="g1p3", play_result="Strikeout"),
        _pbp_row(gamepk="1", play_id="g1p4", play_result="Groundout"),
    ])
    batter_boxscore = pd.DataFrame([{"gamepk": "1", "personId": "1", "batting_order": 3}])
    schedule = pd.DataFrame([{"gamepk": "1", "game_date": pd.Timestamp("2024-04-01")}])

    with patch("batter_pa_volume.compute_batter_lineup_features_for_date", return_value=today_players), \
         patch("batter_pa_volume.read_playbyplay", return_value=pbp), \
         patch("batter_pa_volume.read_batter_boxscore", return_value=batter_boxscore), \
         patch("batter_pa_volume.read_schedule", return_value=schedule):
        result = compute_batter_pa_volume_features_for_date("2024-04-10")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["personId"] == "1"
    assert row["gamepk"] == "9"
    assert row["event_timestamp"] == pd.Timestamp("2024-04-10")
    assert row["batter_pa_roll_season_games_n"] == 1
    assert row["batter_pa_roll_season_avg_n_pa_per_game"] == 4.0


def test_backfill_and_incremental_produce_identical_schema():
    """Same reason this test exists on batter_lineup: both entry
    points must agree on exactly what this feature group looks like."""
    today_players = pd.DataFrame([
        {"personId": "1", "gamepk": "9", "event_timestamp": pd.Timestamp("2024-04-10"), "batting_order": 3},
    ])
    pbp = pd.DataFrame([_pbp_row(gamepk="1", play_id="g1p1", play_result="Single")])
    batter_boxscore = pd.DataFrame([{"gamepk": "1", "personId": "1", "batting_order": 3}])
    schedule = pd.DataFrame([{"gamepk": "1", "game_date": pd.Timestamp("2024-04-01")}])

    with patch("batter_pa_volume.read_playbyplay", return_value=pbp), \
         patch("batter_pa_volume.read_batter_boxscore", return_value=batter_boxscore), \
         patch("batter_pa_volume.read_schedule", return_value=schedule):
        backfill_result = compute_batter_pa_volume_features("2024")

    with patch("batter_pa_volume.compute_batter_lineup_features_for_date", return_value=today_players), \
         patch("batter_pa_volume.read_playbyplay", return_value=pbp), \
         patch("batter_pa_volume.read_batter_boxscore", return_value=batter_boxscore), \
         patch("batter_pa_volume.read_schedule", return_value=schedule):
        incremental_result = compute_batter_pa_volume_features_for_date("2024-04-10")

    assert list(backfill_result.columns) == list(incremental_result.columns)
    assert dict(backfill_result.dtypes) == dict(incremental_result.dtypes)


def test_backfill_and_incremental_agree_on_rolling_value_for_same_history():
    """Value consistency, not just schema (CLAUDE.md's promotion checklist,
    added 2026-09-15 — same gap Chronon/Nubank's train-serve-skew research
    warns about). Uses a REAL 3rd game as backfill's ground truth for what
    that game's rolling avg_n_pa_per_game should be (reflecting games 1-2),
    then recomputes the same value incrementally — history trimmed to only
    games 1-2 (game 3's outcome treated as 'not known yet'), game 3 supplied
    only via today's announced lineup (phantom row). The incremental path's
    phantom-row trick must land on the EXACT same number backfill computes
    once game 3 is real, or a model served live would silently see a
    different feature value than training did for the same situation."""
    pbp = pd.DataFrame([
        _pbp_row(gamepk="1", play_id="g1p1", play_result="Single"),
        _pbp_row(gamepk="1", play_id="g1p2", play_result="Strikeout"),
        _pbp_row(gamepk="2", play_id="g2p1", play_result="Single"),
        _pbp_row(gamepk="2", play_id="g2p2", play_result="Single"),
        _pbp_row(gamepk="2", play_id="g2p3", play_result="Strikeout"),
        _pbp_row(gamepk="3", play_id="g3p1", play_result="Single"),
        _pbp_row(gamepk="3", play_id="g3p2", play_result="Single"),
        _pbp_row(gamepk="3", play_id="g3p3", play_result="Single"),
        _pbp_row(gamepk="3", play_id="g3p4", play_result="Strikeout"),
    ])
    batter_boxscore = pd.DataFrame([
        {"gamepk": "1", "personId": "1", "batting_order": 3},
        {"gamepk": "2", "personId": "1", "batting_order": 3},
        {"gamepk": "3", "personId": "1", "batting_order": 3},
    ])
    schedule = pd.DataFrame([
        {"gamepk": "1", "game_date": pd.Timestamp("2024-04-01")},
        {"gamepk": "2", "game_date": pd.Timestamp("2024-04-02")},
        {"gamepk": "3", "game_date": pd.Timestamp("2024-04-03")},
    ])

    with patch("batter_pa_volume.read_playbyplay", return_value=pbp), \
         patch("batter_pa_volume.read_batter_boxscore", return_value=batter_boxscore), \
         patch("batter_pa_volume.read_schedule", return_value=schedule):
        backfill_result = compute_batter_pa_volume_features("2024")
    backfill_game3 = backfill_result[backfill_result["gamepk"] == "3"].iloc[0]

    history_pbp = pbp[pbp["gamepk"].isin(["1", "2"])]
    history_boxscore = batter_boxscore[batter_boxscore["gamepk"].isin(["1", "2"])]
    history_schedule = schedule[schedule["gamepk"].isin(["1", "2"])]
    today_players = pd.DataFrame([
        {"personId": "1", "gamepk": "3", "event_timestamp": pd.Timestamp("2024-04-03"), "batting_order": 3},
    ])

    with patch("batter_pa_volume.compute_batter_lineup_features_for_date", return_value=today_players), \
         patch("batter_pa_volume.read_playbyplay", return_value=history_pbp), \
         patch("batter_pa_volume.read_batter_boxscore", return_value=history_boxscore), \
         patch("batter_pa_volume.read_schedule", return_value=history_schedule):
        incremental_result = compute_batter_pa_volume_features_for_date("2024-04-03")
    incremental_row = incremental_result.iloc[0]

    assert incremental_row["batter_pa_roll_season_games_n"] == backfill_game3["batter_pa_roll_season_games_n"]
    assert incremental_row["batter_pa_roll_season_avg_n_pa_per_game"] == backfill_game3["batter_pa_roll_season_avg_n_pa_per_game"]
