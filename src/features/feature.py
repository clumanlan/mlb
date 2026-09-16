from feast import Entity

# The 4 FeatureViews/FileSources previously defined here (team_batter_base_fv,
# player_batter_base_fv, starting_pitcher_base_fv, bullpen_pitcher_base_fv)
# were removed 2026-09-15 — first-generation feature-store attempt, superseded
# before a single `feast apply` ever ran (see DECISIONS.md's 2026-09-15 entry).
# These two entities are the reusable part: the n_pa_predictor feature-store
# work (src/features/transforms/batter_lineup.py,
# batter_pa_volume.py) already joins on the `player` entity's key.

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
