from feature import batter_lineup_fv, batter_pa_volume_fv


def test_batter_lineup_fv_uses_player_entity_and_batting_order_schema():
    """First real FeatureView for n_pa_predictor's frozen feature set (FTI
    step 2) — must join on the player entity and expose exactly the
    batting_order column batter_lineup.py's backfill/incremental output
    produces, pointed at the real S3 path materialize_backfill.py writes."""
    assert batter_lineup_fv.entities == ["player"]
    schema_names = {f.name for f in batter_lineup_fv.schema}
    assert schema_names == {"batting_order"}
    assert batter_lineup_fv.batch_source.path == "s3://mlbdk/feast/features/batter_lineup/"


def test_batter_pa_volume_fv_uses_player_entity_and_rolling_schema():
    assert batter_pa_volume_fv.entities == ["player"]
    schema_names = {f.name for f in batter_pa_volume_fv.schema}
    assert schema_names == {
        "batter_pa_roll_season_games_n", "batter_pa_roll_season_avg_n_pa_per_game",
    }
    assert batter_pa_volume_fv.batch_source.path == "s3://mlbdk/feast/features/batter_pa_volume/"
