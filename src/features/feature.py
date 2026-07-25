from datetime import timedelta
from feast import Entity, FeatureView, FileSource
from feast.data_format import ParquetFormat

team = Entity(
    name="team",
    join_keys=["team_id"],
    description="MLB Team"
)

player = Entity(
    name="player",
    join_keys=["personId"],
    description="MLB Player"
)

team_batter_base_source = FileSource(
    path="s3://mlbdk/feast/features/team_batter_base/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
)

player_batter_base_source = FileSource(
    path="s3://mlbdk/feast/features/player_batter_base/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
)

starting_pitcher_base_source = FileSource(
    path="s3://mlbdk/feast/features/starting_pitcher_base/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
)

bullpen_pitcher_base_source = FileSource(
    path="s3://mlbdk/feast/features/bullpen_pitcher_base/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
)

team_batter_base_fv = FeatureView(
    name="team_batter_base_fv",
    entities=[team],
    ttl=timedelta(days=7),
    source=team_batter_base_source,
    online=True,
)

player_batter_base_fv = FeatureView(
    name="player_batter_base_fv",
    entities=[player],
    ttl=timedelta(days=7),
    source=player_batter_base_source,
    online=True,
)

starting_pitcher_base_fv = FeatureView(
    name="starting_pitcher_base_fv",
    entities=[player],
    ttl=timedelta(days=7),
    source=starting_pitcher_base_source,
    online=True,
)

bullpen_pitcher_base_fv = FeatureView(
    name="bullpen_pitcher_base_fv",
    entities=[team],
    ttl=timedelta(days=7),
    source=bullpen_pitcher_base_source,
    online=True,
)
