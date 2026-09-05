"""
Builds a per-start game log (one row per pitcher-game) for TEST_SEASON=2025,
joining pitches-thrown from pitcher_boxscore onto the batters-faced/strikeout
counts already derivable from diagnostics/data.csv (PA-grain, SP-only).
Feeds pitcher_view.py's season line chart. Not a model feature -- pitches
thrown per game is a REALIZED in-game stat, never fed to v6.

Run from repo root with:
    PYTHONPATH=src python src/models/k_predictor/diagnostics/build_game_log.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2), and
diagnostics/data.csv (run build_data.py first).
"""
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd
import yaml

OUT_DIR = Path(__file__).parent
K_PREDICTOR_DIR = Path(__file__).resolve().parent.parent

with open(K_PREDICTOR_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)
BUCKET, REGION, TEST_SEASON = cfg["bucket"], cfg["region"], cfg["test_season"]

pa_df = pd.read_csv(OUT_DIR / "data.csv")

game_log = pa_df.groupby(["gamepk", "pitcher_id", "pitcher_name", "game_date"], as_index=False).agg(
    batters_faced=("batter_id", "count"),
    strikeouts=("is_strikeout", "sum"),
    mean_pred_prob=("pred_prob", "mean"),
)

boto_session = boto3.Session(region_name=REGION)
pitcher_boxscore = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/processed_data/prepared/pitcher_boxscore/{TEST_SEASON}/", boto3_session=boto_session,
)
pitches = pitcher_boxscore[["gamepk", "personId", "p"]].rename(
    columns={"personId": "pitcher_id", "p": "pitches_thrown"}
)
pitches["gamepk"] = pitches["gamepk"].astype(str)
pitches["pitcher_id"] = pitches["pitcher_id"].astype(str)
game_log["gamepk"] = game_log["gamepk"].astype(str)
game_log["pitcher_id"] = game_log["pitcher_id"].astype(str)

game_log = game_log.merge(pitches, on=["gamepk", "pitcher_id"], how="left")
game_log = game_log.sort_values(["pitcher_name", "game_date"])

out_path = OUT_DIR / "game_log.csv"
game_log.to_csv(out_path, index=False)
print(f"Wrote {len(game_log):,} pitcher-starts to {out_path}")
print(f"Missing pitches_thrown: {game_log['pitches_thrown'].isna().sum()}")
