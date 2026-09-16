from datetime import timedelta

from feast import Entity, Field, FeatureView, FileSource
from feast.data_format import ParquetFormat
from feast.types import Int64, Float32
from feast.value_type import ValueType

# The 4 FeatureViews/FileSources previously defined here (team_batter_base_fv,
# player_batter_base_fv, starting_pitcher_base_fv, bullpen_pitcher_base_fv)
# were removed 2026-09-15 — first-generation feature-store attempt, superseded
# before a single `feast apply` ever ran (see DECISIONS.md's 2026-09-15 entry).

team = Entity(
    name="team",
    join_keys=["team_id"],
    value_type=ValueType.STRING,
    description="MLB Team"
)

player = Entity(
    name="player",
    join_keys=["personId"],
    value_type=ValueType.STRING,
    description="MLB Player"
)

# First real FeatureViews registered against this store (FTI Production plan
# step 2, 2026-09-15) — n_pa_predictor's v1 low_pa classifier's frozen
# feature set. Sources point at what materialize_backfill.py
# (src/features/transforms/jobs/) writes; batter_lineup.py/batter_pa_volume.py
# are the canonical compute functions (see CLAUDE.md's Feature Engineering
# section for the naming/promotion convention these follow).

batter_lineup_source = FileSource(
    name="batter_lineup_source",
    path="s3://mlbdk/feast/features/batter_lineup/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
)

batter_lineup_fv = FeatureView(
    name="batter_lineup_fv",
    entities=[player],
    ttl=timedelta(days=7),
    schema=[
        Field(name="batting_order", dtype=Int64),
    ],
    source=batter_lineup_source,
    online=True,
)

batter_pa_volume_source = FileSource(
    name="batter_pa_volume_source",
    path="s3://mlbdk/feast/features/batter_pa_volume/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
)

batter_pa_volume_fv = FeatureView(
    name="batter_pa_volume_fv",
    entities=[player],
    ttl=timedelta(days=7),
    schema=[
        Field(name="batter_pa_roll_season_games_n", dtype=Int64),
        Field(name="batter_pa_roll_season_avg_n_pa_per_game", dtype=Float32),
    ],
    source=batter_pa_volume_source,
    online=True,
)
