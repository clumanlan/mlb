"""
n_pa_predictor v1 low_pa classifier: feature-ablation study.
Run from src/models/n_pa_predictor/ with: python experiments/v1_low_pa_classifier/ablation/run_ablation.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Reuses the SAME data-loading/assembly steps as
experiments/v1_low_pa_classifier/train.py (steps 1-4 there) so the dataset
this ablation runs against is identical to the one the locked production
result (XGBoost @ 0.85 threshold, precision 64.7%, 95% CI [55.6%, 72.8%],
n=116, val season 2024) was measured on. Duplicated rather than imported,
matching this repo's existing convention (train.py's own header comment:
"Identical assembly to baseline/model/run.py" — sibling scripts in this
model duplicate assembly rather than share a loader function).
"""
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import mlflow
import pandas as pd
mlflow.set_tracking_uri("file:./mlruns")

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.processing.features import rolling_stats

import models.n_pa_predictor.processing.pipeline as pipeline
from models.n_pa_predictor.processing.features.batter_playing_time import build_batter_pa_rolling_stats
from models.n_pa_predictor.processing.features.ablation import cluster_correlated_features
from models.n_pa_predictor.utils.threshold_eval import run_ablation_sweep, fit_and_get_importances
from models.n_pa_predictor.utils.mlflow_logging import log_evaluation_to_mlflow, get_git_sha

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
LOW_PA_THRESHOLD = 3
STARTER_IP_SHRINKAGE_K = 5.0
THRESHOLD = 0.85
CORR_THRESHOLD = 0.8

pd.set_option("display.max_columns", None)


# ── 1. Config ────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET          = cfg["bucket"]
REGION          = cfg["region"]
TRAIN_SEASONS   = cfg["train_seasons"]
FEATURE_SEASONS = cfg["feature_seasons"]
DATE_COL        = cfg["date_column"]
TEST_SEASON     = cfg["test_season"]
VAL_SEASON      = cfg["val_season"]

TARGET = "low_pa"

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))


# ── 2. Load data from S3 ─────────────────────────────────────────────────────
def read_parquet_seasons(path_tpl, seasons, chunked=False):
    frames = []
    for season in seasons:
        path = path_tpl.format(bucket=BUCKET, season=season)
        print(f"  {path}")
        if chunked:
            for chunk in wr.s3.read_parquet(path=path, chunked=True, boto3_session=boto_session):
                if "spin_direction" in chunk.columns:
                    chunk["spin_direction"] = chunk["spin_direction"].astype("float64")
                frames.append(chunk)
        else:
            frames.append(wr.s3.read_parquet(path=path, boto3_session=boto_session))
    return pd.concat(frames, ignore_index=True)


print("\nLoading play-by-play...")
pbp = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/playbyplay/{season}/", TRAIN_SEASONS, chunked=True,
)
print("\nLoading schedule...")
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/", all_boxscore_seasons,
)
print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", all_boxscore_seasons,
)
print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", all_boxscore_seasons,
)
print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)


# ── 3. Build batter-game DataFrame ────────────────────────────────────────────
print("\nBuilding batter-game DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = hp_pipeline.build_pbp_features(pbp, schedule, player_info)

batter_game = pipeline.build_batter_game_frame(pbp, batter_boxscore, schedule)
batter_game["game_season"] = batter_game["game_date"].dt.year
batter_game[TARGET] = (batter_game["n_pa"] <= LOW_PA_THRESHOLD).astype(int)

batter_team = pbp[["gamepk", "batter_id", "batter_team_id"]].drop_duplicates(["gamepk", "batter_id"])
starter_by_team = pbp[["gamepk", "pitcher_team_id", "starting_pitcher_id"]].drop_duplicates(["gamepk", "pitcher_team_id"])
home = schedule[["gamepk", "home_id", "away_id"]].rename(columns={"home_id": "team_id", "away_id": "opp_team_id"})
away = schedule[["gamepk", "home_id", "away_id"]].rename(columns={"away_id": "team_id", "home_id": "opp_team_id"})
opp_team = pd.concat([home, away], ignore_index=True)
opp_team["gamepk"] = opp_team["gamepk"].astype(str)
opp_team["team_id"] = opp_team["team_id"].astype(str)
opp_team["opp_team_id"] = opp_team["opp_team_id"].astype(str)

batter_game = batter_game.merge(batter_team, on=["gamepk", "batter_id"], how="left")
batter_game = batter_game.merge(
    opp_team.rename(columns={"team_id": "batter_team_id"}), on=["gamepk", "batter_team_id"], how="left",
)
batter_game = batter_game.merge(
    starter_by_team.rename(columns={"pitcher_team_id": "opp_team_id", "starting_pitcher_id": "opp_starting_pitcher_id"}),
    on=["gamepk", "opp_team_id"], how="left",
)
batter_game["is_home"] = (batter_game["batter_team_id"] == batter_game.merge(
    schedule[["gamepk", "home_id"]], on="gamepk", how="left"
)["home_id"])

pitcher_start_ip_last_season = season_stats.build_pitcher_start_ip_stats(pitcher_boxscore, pbp)
league_avg_start_ip = season_stats.build_league_avg_start_ip(pitcher_start_ip_last_season)
pitcher_start_ip_this_season = game_context.build_pitcher_start_ip_this_season(pitcher_boxscore, pbp)
expected_start_innings = game_context.build_expected_start_innings(
    pitcher_start_ip_last_season, pitcher_start_ip_this_season, league_avg_start_ip,
    k=STARTER_IP_SHRINKAGE_K,
)[["personId", "gamepk", "expected_start_innings", "expected_start_innings_weight"]]
expected_start_innings["personId"] = expected_start_innings["personId"].astype(str)
expected_start_innings["gamepk"] = expected_start_innings["gamepk"].astype(str)
batter_game = batter_game.merge(
    expected_start_innings.rename(columns={"personId": "opp_starting_pitcher_id"}),
    on=["gamepk", "opp_starting_pitcher_id"], how="left",
)

pitcher_rolling_season_stats = rolling_stats.build_pitcher_rolling_stats_all_roles(
    pitcher_boxscore, pbp, window="season"
)
opp_starter_whip = (
    pitcher_rolling_season_stats[pitcher_rolling_season_stats["pitcher_role"] == "sp"]
    [["pitcher_key_id", "gamepk", "pitcher_roll_season_whip"]]
    .rename(columns={"pitcher_key_id": "opp_starting_pitcher_id", "pitcher_roll_season_whip": "opp_starter_whip_season"})
)
opp_starter_whip["opp_starting_pitcher_id"] = opp_starter_whip["opp_starting_pitcher_id"].astype(str)
opp_starter_whip["gamepk"] = opp_starter_whip["gamepk"].astype(str)
batter_game = batter_game.merge(opp_starter_whip, on=["gamepk", "opp_starting_pitcher_id"], how="left")

batter_pa_rolling = build_batter_pa_rolling_stats(batter_game, window="season")
batter_game = batter_game.merge(
    batter_pa_rolling.drop(columns=["game_date", "game_season"]),
    on=["batter_id", "gamepk"], how="left",
)

team_win_loss_season = game_context.build_team_win_loss_record(schedule, window="season")
team_win_loss_season["team_id"] = team_win_loss_season["team_id"].astype(str)
batter_game = batter_game.merge(
    team_win_loss_season.drop(columns=["game_date", "game_datetime", "game_season"]).rename(
        columns={"team_id": "batter_team_id"}
    ),
    on=["batter_team_id", "gamepk"], how="left",
)


# ── 4. Season-based train / val split ─────────────────────────────────────────
FEATURE_COLS = [
    "batting_order",
    "is_home",
    "expected_start_innings",
    "expected_start_innings_weight",
    "opp_starter_whip_season",
    "batter_n_pa_roll_season_games_n",
    "batter_n_pa_roll_season_avg_n_pa_per_game",
    "team_roll_season_win_pct",
    "team_roll_season_runs_scored",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in batter_game.columns]

model_df = batter_game[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()

# is_home is boolean; XGBoost handles bool fine, but the correlation/importance
# helpers expect numeric-comparable dtypes throughout.
for df_ in (train_df, val_df):
    df_["is_home"] = df_["is_home"].astype(int)
    df_[FEATURE_COLS] = df_[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    df_[FEATURE_COLS] = df_[FEATURE_COLS].fillna(df_[FEATURE_COLS].median())

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}")
print(f"Feature cols ({len(FEATURE_COLS)}): {FEATURE_COLS}")


# ── 5. Correlation clusters + feature importances (reported, not just used) ──
clusters = cluster_correlated_features(train_df, FEATURE_COLS, threshold=CORR_THRESHOLD)
importance = fit_and_get_importances(train_df, FEATURE_COLS, TARGET)

print(f"\nCorrelation families (Spearman |r| >= {CORR_THRESHOLD}):")
for cluster in clusters:
    print(f"  {cluster}")

print("\nXGBoost feature importances (baseline, all 9 features):")
print(importance.sort_values(ascending=False).round(4).to_string())


# ── 6. Ablation sweep ──────────────────────────────────────────────────────────
sweep_df = run_ablation_sweep(
    train_df, val_df, FEATURE_COLS, TARGET, threshold=THRESHOLD, corr_threshold=CORR_THRESHOLD,
)

print(f"\n{'='*90}")
print(f"ABLATION SWEEP (val={VAL_SEASON}, threshold={THRESHOLD})")
print("=" * 90)
print(sweep_df.to_string(index=False))
print("=" * 90)

# The last accepted row's features_used IS the minimal survivor set: every
# row's features_used reflects what was actually adopted going forward
# (run_ablation_sweep only carries a dedup/drop forward when it doesn't
# break the CI), and elimination stops at the first rejection.
accepted = sweep_df[sweep_df["within_baseline_ci"]]
final_survivors = list(accepted.iloc[-1]["features_used"])

print(f"\nRecommended minimal FEATURE_COLS ({len(final_survivors)}): {final_survivors}")


# ── 7. Write ablation_results.md ──────────────────────────────────────────────
baseline_row = sweep_df.iloc[0]
lines = []
lines.append("# n_pa_predictor v1 low_pa classifier — feature-ablation results\n")
lines.append(
    f"Val season {VAL_SEASON}, XGBoost @ {THRESHOLD} confidence threshold. "
    f"Baseline (locked production, all {len(FEATURE_COLS)} features): "
    f"precision {baseline_row['precision']:.3f}, "
    f"95% Wilson CI [{baseline_row['ci_low']*100:.1f}%, {baseline_row['ci_high']*100:.1f}%], "
    f"n={baseline_row['n']}.\n"
)
lines.append("## Methodology\n")
lines.append(
    "1. Cluster the 9 frozen features by Spearman rank correlation "
    f"(threshold {CORR_THRESHOLD}) into redundant families.\n"
    "2. Within each family with more than one member, keep only the "
    "highest-XGBoost-importance member.\n"
    "3. Backward-eliminate remaining features one at a time, weakest "
    "importance first, retraining and re-checking precision-at-"
    f"{THRESHOLD} each time.\n"
    "4. Stop eliminating once a drop's 95% Wilson CI no longer overlaps "
    "the baseline's — that feature is load-bearing, not a passenger.\n"
)
lines.append("## Correlation families\n")
for cluster in clusters:
    lines.append(f"- {cluster}\n")
lines.append("\n## XGBoost feature importances (baseline model)\n")
lines.append("```\n" + importance.sort_values(ascending=False).round(4).to_string() + "\n```\n")
lines.append("\n## Sweep results\n")
lines.append("```\n" + sweep_df.to_string(index=False) + "\n```\n")
lines.append(f"\n## Recommendation\n")
lines.append(
    f"Minimal feature set ({len(final_survivors)} of {len(FEATURE_COLS)}): "
    f"`{final_survivors}`.\n"
)

with open(OUT_DIR / "ablation_results.md", "w") as f:
    f.writelines(lines)

print(f"\nWrote {OUT_DIR / 'ablation_results.md'}")


# ── 8. MLflow logging — one run per sweep variant ─────────────────────────────
# Same experiment ("n_pa_predictor") and logging utility train.py uses, so
# ablation runs are queryable/comparable alongside every other run for this
# model, not siloed off in a markdown table only.
git_sha = get_git_sha()
for row in sweep_df.itertuples():
    metrics = {
        k: v for k, v in {
            "precision": row.precision, "n": row.n,
            "ci_low": row.ci_low, "ci_high": row.ci_high,
        }.items() if pd.notna(v)
    }
    log_evaluation_to_mlflow(
        metrics=metrics,
        params={
            "variant": row.variant,
            "dropped_feature": row.dropped_feature or "",
            "n_features": len(row.features_used),
            "features_used": ",".join(row.features_used),
            "threshold": THRESHOLD,
            "corr_threshold": CORR_THRESHOLD,
            "val_season": VAL_SEASON,
        },
        tags={
            "model_type": "XGBoost", "stage": "ablation", "variant": row.variant,
            "within_baseline_ci": str(row.within_baseline_ci),
            "val_season": str(VAL_SEASON), "git_sha": git_sha,
        },
        artifact_paths=[str(OUT_DIR / "ablation_results.md")] if row.variant == "baseline" else None,
    )
print("\nLogged ablation sweep to MLflow (experiment: n_pa_predictor)")
