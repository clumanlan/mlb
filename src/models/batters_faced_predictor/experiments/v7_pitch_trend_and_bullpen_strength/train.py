"""
batters_faced_predictor experiment v7: pitch-count TREND (trailing-3-start
delta vs. the pitcher's own season average) + own-team BULLPEN STRENGTH
(season + trailing-5-game whip/k_rate/bb_rate/hr_rate/strike_rate, pooled by
team via the existing role-aware rolling machinery).
Run from src/models/batters_faced_predictor/ with:
  python experiments/v7_pitch_trend_and_bullpen_strength/train.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    v7 SUMMARY                                #
# ---------------------------------------------------------------------------- #
# Two independent, previously-flagged-but-untried threads, both zero new
# production code — pure composition of already-existing, already-tested
# functions (same "zero new production code" shape as v1/v2/v5).
#
#   1. PITCH-COUNT TREND (NEW this pass):
#      v1 added pitch-count LEVEL (pitcher_roll_season_pitch_count_avg,
#      season-to-date) and it ranked #2 in feature importance without moving
#      aggregate MAE — flagged in both v1_results.md and the v2 roadmap
#      "Next" note as the natural follow-up: does the TREND (is a pitcher's
#      recent pitch count running above/below his own season baseline) carry
#      signal the level alone doesn't? Built via
#      rolling_stats.build_pbp_pitcher_rolling_feats(sp_pbp, window=3, ...)
#      — the SAME function v1's season-level pitch-count feature already
#      uses, just called a second time with an int window instead of
#      'season' (already-supported, no code change). Trend expressed as
#      ratio + direction, same pattern v2's pa_trend_ratio/pa_trend_direction
#      already established for this model.
#
#   2. OWN-TEAM BULLPEN STRENGTH (NEW this pass, new hypothesis):
#      Every opposing-team feature tried so far (v1's on-base/walk rate, v5's
#      win/loss scoring level, v6's scoring volatility) looks at the OTHER
#      team. v7 flips the lens onto the starting pitcher's OWN bullpen: if a
#      manager trusts a strong bullpen, that may itself predict an earlier
#      hook (freeing up more of the swing role's fresher innings), separate
#      from anything about the pitcher's own workload or the opposing
#      lineup. Built via rolling_stats.build_pitcher_rolling_stats_all_roles
#      (pitcher_boxscore, pbp, window) — the same function k_predictor's v1/
#      v2 and n_pa_predictor's baseline already use for role-aware pooling;
#      it already stacks sp rows (per-pitcher) and bullpen rows (pooled by
#      team_id, since a specific reliever's identity isn't pre-game knowable)
#      under a common pitcher_key_id. Filtered to pitcher_role == 'bullpen'
#      and joined on the STARTER's own team (pitcher_team_id, not opp_team_id
#      — this is a self-team feature, unlike every prior opposing-team pass).
#      whip/k_rate/bb_rate/hr_rate/strike_rate, at season and trailing-5-game
#      windows (same two-window convention as v5/v6).
#
# Evaluated the same way as v1-v6: MAE primary vs v2's own tuned-XGBoost
# floor (2.6471), RMSE/Bias/Pearson r secondary, same
# expected_batters_faced_weight-quartile stratification, same bf_gap-quartile
# floor re-check, same established-starter (11+ starts) bucket re-check. Same
# tuned-XGBoost hyperparameters as v1-v6 (carried over unchanged, per repo
# convention — see experiments/xgb_vs_cascade_diagnostic/run.py).
# ---------------------------------------------------------------------------- #
import yaml
from datetime import datetime
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
from models.batters_faced_predictor.utils.mlflow_logging import log_evaluation_to_mlflow, get_git_sha

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.processing.features import rolling_stats

import models.batters_faced_predictor.processing.pipeline as pipeline

STAGE = Path(__file__).parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PA_SHRINKAGE_K = 5.0
SHORT_WINDOW_GAMES = 5

pd.set_option('display.max_columns', None)


# ── 1. Config ────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET          = cfg["bucket"]
REGION          = cfg["region"]
TRAIN_SEASONS   = cfg["train_seasons"]
FEATURE_SEASONS = cfg["feature_seasons"]
TARGET          = cfg["target_column"]
DATE_COL        = cfg["date_column"]
TEST_SEASON     = cfg["test_season"]
VAL_SEASON      = cfg["val_season"]
MODEL_NAME      = cfg["model_name"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)

EARLY_STOP_SEASON = 2023
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]

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
    "s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", TRAIN_SEASONS,
)
print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)


# ── 3. Build start-grain DataFrame ────────────────────────────────────────────
print("\nBuilding start-grain DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
pbp = hp_pipeline.build_pbp_features(pbp, schedule, player_info)
pbp = pbp.merge(schedule[["gamepk", "game_datetime"]], on="gamepk", how="left")

start_outcome = pipeline.create_start_pa_outcome(pbp)

# ------------------------- 0. BATTERS-FACED CASCADE (baseline features) ------ #
pitcher_start_pa_last_season = season_stats.build_pitcher_start_pa_stats(pbp)
league_avg_start_pa = season_stats.build_league_avg_start_pa(pitcher_start_pa_last_season)
team_avg_start_pa = season_stats.build_team_avg_start_pa(pitcher_start_pa_last_season)
pitcher_start_pa_this_season = game_context.build_pitcher_start_pa_this_season(pbp)

expected_pa = game_context.build_expected_batters_faced(
    pitcher_start_pa_last_season, pitcher_start_pa_this_season, team_avg_start_pa, league_avg_start_pa,
    k=PA_SHRINKAGE_K,
)
expected_pa["personId"] = expected_pa["personId"].astype(str)
expected_pa["gamepk"] = expected_pa["gamepk"].astype(str)

start_outcome = start_outcome.drop(columns=["game_date", "game_season"]).merge(
    expected_pa, on=["personId", "gamepk"], how="left",
)

# ------------------------- 1. OPPOSING-LINEUP ON-BASE/WALK RATE (v1) --------- #
home_away = schedule[["gamepk", "home_id", "away_id"]].drop_duplicates("gamepk").assign(
    gamepk=lambda x: x["gamepk"].astype(str), home_id=lambda x: x["home_id"].astype(str),
    away_id=lambda x: x["away_id"].astype(str),
)
start_outcome["pitcher_team_id"] = start_outcome["pitcher_team_id"].astype(str)
start_outcome = start_outcome.merge(home_away, on="gamepk", how="left")
start_outcome["opp_team_id"] = np.where(
    start_outcome["pitcher_team_id"] == start_outcome["home_id"],
    start_outcome["away_id"], start_outcome["home_id"],
)
start_outcome["is_home"] = (start_outcome["pitcher_team_id"] == start_outcome["home_id"]).astype(int)

team_onbase_rolling = rolling_stats.build_team_batter_onbase_rolling_feats(pbp, batter_boxscore, window="season")
team_onbase_rolling = team_onbase_rolling.rename(columns={"batter_team_id": "opp_team_id"})[
    ["opp_team_id", "gamepk", "team_roll_season_walk_rate", "team_roll_season_on_base_rate"]
].assign(gamepk=lambda x: x["gamepk"].astype(str), opp_team_id=lambda x: x["opp_team_id"].astype(str))
start_outcome = start_outcome.merge(team_onbase_rolling, on=["opp_team_id", "gamepk"], how="left")

# ------------------------- 2. TEAM REST DAYS (v1, pitcher's own team) -------- #
team_rest_days = game_context.build_team_rest_days(schedule)[["team_id", "gamepk", "team_days_since_last_game"]]
team_rest_days = team_rest_days.assign(
    team_id=lambda x: x["team_id"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
).rename(columns={"team_id": "pitcher_team_id", "team_days_since_last_game": "pitcher_team_days_since_last_game"})
start_outcome = start_outcome.merge(team_rest_days, on=["pitcher_team_id", "gamepk"], how="left")

# ------------------------- 3. PITCH-EFFICIENCY PROXY (v1) -------------------- #
sp_pbp = pbp[pbp["pitcher_role"] == "sp"]
pitcher_pitch_efficiency_season = rolling_stats.build_pbp_pitcher_rolling_feats(
    sp_pbp, window="season", pitcher_role="sp", entity_col="pitcher_id",
)[["pitcher_id", "gamepk", "pitcher_roll_season_pitch_count_avg"]].rename(columns={"pitcher_id": "personId"})
pitcher_pitch_efficiency_season = pitcher_pitch_efficiency_season.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)
start_outcome = start_outcome.merge(pitcher_pitch_efficiency_season, on=["personId", "gamepk"], how="left")

# ------------------------- 4. HANDEDNESS (baseline feature) ------------------ #
pitcher_hand = (
    pbp[["pitcher_id", "pitcher_throw_hand"]]
    .drop_duplicates(subset=["pitcher_id"])
    .rename(columns={"pitcher_id": "personId"})
)
start_outcome = start_outcome.merge(pitcher_hand, on="personId", how="left")

# ------------------------- 5. TRAILING-3-START PA TREND (v2) ----------------- #
pitcher_start_pa_last3 = game_context.build_pitcher_start_pa_this_season(pbp, window=3)[
    ["personId", "gamepk", "pitcher_last3_start_pa_avg_pa_per_start", "pitcher_last3_start_pa_starts_n"]
].assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str))
start_outcome = start_outcome.merge(pitcher_start_pa_last3, on=["personId", "gamepk"], how="left")

start_outcome["pa_trend_ratio"] = start_outcome["pitcher_last3_start_pa_avg_pa_per_start"] / (
    start_outcome["pitcher_this_season_start_pa_avg_pa_per_start"].replace(0, np.nan)
)
start_outcome["pa_trend_direction"] = np.sign(
    start_outcome["pitcher_last3_start_pa_avg_pa_per_start"] - start_outcome["pitcher_this_season_start_pa_avg_pa_per_start"]
)

# ------------------------- 6. PITCHER'S OWN REST DAYS (v2) ------------------- #
rest_days_raw = game_context.build_pitcher_rest_days(pbp)
rest_days_for_merge = rest_days_raw[["personId", "gamepk", "pitcher_days_since_last_start"]].assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)
start_outcome = start_outcome.merge(rest_days_for_merge, on=["personId", "gamepk"], how="left")

# ------------------------- 7. WORKLOAD DENSITY (v2) -------------------------- #
workload = game_context.build_pitcher_workload_density(pitcher_boxscore, pbp, rest_days_raw)[
    ["personId", "gamepk", "pitcher_last_start_pitches", "pitcher_workload_density"]
].assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str))
start_outcome = start_outcome.merge(workload, on=["personId", "gamepk"], how="left")

starts_n_safe = start_outcome["pitcher_this_season_start_pa_starts_n"].fillna(0)
start_outcome["pitcher_workload_density_shrunk"] = start_outcome["pitcher_workload_density"] * (
    starts_n_safe / (starts_n_safe + PA_SHRINKAGE_K)
)

# ------------------------- 8. OPPOSING-TEAM SCORING STRENGTH (v5) ------------ #
opp_win_loss_season = game_context.build_team_win_loss_record(schedule, window="season")
opp_win_loss_season = opp_win_loss_season.rename(columns={"team_id": "opp_team_id"})[[
    "opp_team_id", "gamepk",
    "team_roll_season_win_pct", "team_roll_season_runs_scored", "team_roll_season_run_diff",
]].rename(columns={
    "team_roll_season_win_pct": "opp_team_roll_season_win_pct",
    "team_roll_season_runs_scored": "opp_team_roll_season_runs_scored",
    "team_roll_season_run_diff": "opp_team_roll_season_run_diff",
}).assign(gamepk=lambda x: x["gamepk"].astype(str), opp_team_id=lambda x: x["opp_team_id"].astype(str))
start_outcome = start_outcome.merge(opp_win_loss_season, on=["opp_team_id", "gamepk"], how="left")

opp_win_loss_short = game_context.build_team_win_loss_record(schedule, window=SHORT_WINDOW_GAMES)
short_win_pct_col = f"team_roll_last{SHORT_WINDOW_GAMES}g_win_pct"
short_runs_col = f"team_roll_last{SHORT_WINDOW_GAMES}g_runs_scored"
short_run_diff_col = f"team_roll_last{SHORT_WINDOW_GAMES}g_run_diff"
opp_win_loss_short = opp_win_loss_short.rename(columns={"team_id": "opp_team_id"})[[
    "opp_team_id", "gamepk", short_win_pct_col, short_runs_col, short_run_diff_col,
]].rename(columns={
    short_win_pct_col: "opp_team_roll_last5g_win_pct",
    short_runs_col: "opp_team_roll_last5g_runs_scored",
    short_run_diff_col: "opp_team_roll_last5g_run_diff",
}).assign(gamepk=lambda x: x["gamepk"].astype(str), opp_team_id=lambda x: x["opp_team_id"].astype(str))
start_outcome = start_outcome.merge(opp_win_loss_short, on=["opp_team_id", "gamepk"], how="left")

# ------------------------- 9. OPPOSING-TEAM SCORING VOLATILITY (v6) ---------- #
opp_volatility_season = game_context.build_team_scoring_volatility(schedule, window="season")
opp_volatility_season = opp_volatility_season.rename(columns={"team_id": "opp_team_id"})[[
    "opp_team_id", "gamepk",
    "team_roll_season_runs_scored_mean", "team_roll_season_runs_scored_std", "team_roll_season_runs_scored_max",
]].rename(columns={
    "team_roll_season_runs_scored_mean": "opp_team_roll_season_runs_scored_mean",
    "team_roll_season_runs_scored_std": "opp_team_roll_season_runs_scored_std",
    "team_roll_season_runs_scored_max": "opp_team_roll_season_runs_scored_max",
}).assign(gamepk=lambda x: x["gamepk"].astype(str), opp_team_id=lambda x: x["opp_team_id"].astype(str))
start_outcome = start_outcome.merge(opp_volatility_season, on=["opp_team_id", "gamepk"], how="left")

opp_volatility_short = game_context.build_team_scoring_volatility(schedule, window=SHORT_WINDOW_GAMES)
short_mean_col = f"team_roll_last{SHORT_WINDOW_GAMES}g_runs_scored_mean"
short_std_col = f"team_roll_last{SHORT_WINDOW_GAMES}g_runs_scored_std"
short_max_col = f"team_roll_last{SHORT_WINDOW_GAMES}g_runs_scored_max"
opp_volatility_short = opp_volatility_short.rename(columns={"team_id": "opp_team_id"})[[
    "opp_team_id", "gamepk", short_mean_col, short_std_col, short_max_col,
]].rename(columns={
    short_mean_col: "opp_team_roll_last5g_runs_scored_mean",
    short_std_col: "opp_team_roll_last5g_runs_scored_std",
    short_max_col: "opp_team_roll_last5g_runs_scored_max",
}).assign(gamepk=lambda x: x["gamepk"].astype(str), opp_team_id=lambda x: x["opp_team_id"].astype(str))
start_outcome = start_outcome.merge(opp_volatility_short, on=["opp_team_id", "gamepk"], how="left")

# ------------------------- 10. PITCH-COUNT TREND (NEW, v7) ------------------- #
# Same rolling function v1's season-level pitch-count feature already uses
# (rolling_stats.build_pbp_pitcher_rolling_feats), just called again with an
# int window instead of 'season' — trailing-3-start pitch-count average.
pitcher_pitch_efficiency_last3 = rolling_stats.build_pbp_pitcher_rolling_feats(
    sp_pbp, window=3, pitcher_role="sp", entity_col="pitcher_id",
)[["pitcher_id", "gamepk", "pitcher_roll_last3g_pitch_count_avg"]].rename(columns={"pitcher_id": "personId"})
pitcher_pitch_efficiency_last3 = pitcher_pitch_efficiency_last3.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)
start_outcome = start_outcome.merge(pitcher_pitch_efficiency_last3, on=["personId", "gamepk"], how="left")

start_outcome["pitch_count_trend_ratio"] = start_outcome["pitcher_roll_last3g_pitch_count_avg"] / (
    start_outcome["pitcher_roll_season_pitch_count_avg"].replace(0, np.nan)
)
start_outcome["pitch_count_trend_direction"] = np.sign(
    start_outcome["pitcher_roll_last3g_pitch_count_avg"] - start_outcome["pitcher_roll_season_pitch_count_avg"]
)

# ------------------------- 11. OWN-TEAM BULLPEN STRENGTH (NEW, v7) ----------- #
# rolling_stats.build_pitcher_rolling_stats_all_roles already stacks sp rows
# (per-pitcher) and bullpen rows (pooled by team_id, since a specific
# reliever's identity isn't pre-game knowable) under a common pitcher_key_id
# — the same function k_predictor's v1/v2 and n_pa_predictor's baseline
# already use. Filtered to the bullpen rows and joined on the STARTER's own
# team (pitcher_team_id) — a self-team feature, unlike every prior
# opposing-team pass (v1, v5, v6).
BULLPEN_RATE_COLS = ["whip", "k_rate", "bb_rate", "hr_rate", "strike_rate"]


def _own_team_bullpen_feats(window, out_prefix):
    all_roles = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window=window)
    bullpen = all_roles[all_roles["pitcher_role"] == "bullpen"]
    in_prefix = rolling_stats._rolling_prefix("pitcher", window)
    rate_cols = [f"{in_prefix}{c}" for c in BULLPEN_RATE_COLS]
    bullpen = bullpen[["pitcher_key_id", "gamepk"] + rate_cols].rename(
        columns={col: f"{out_prefix}{c}" for col, c in zip(rate_cols, BULLPEN_RATE_COLS)}
    ).rename(columns={"pitcher_key_id": "pitcher_team_id"})
    return bullpen.assign(
        pitcher_team_id=lambda x: x["pitcher_team_id"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
    )


bullpen_season = _own_team_bullpen_feats(window="season", out_prefix="bullpen_roll_season_")
start_outcome = start_outcome.merge(bullpen_season, on=["pitcher_team_id", "gamepk"], how="left")

bullpen_short = _own_team_bullpen_feats(window=SHORT_WINDOW_GAMES, out_prefix="bullpen_roll_last5g_")
start_outcome = start_outcome.merge(bullpen_short, on=["pitcher_team_id", "gamepk"], how="left")


# ── 4. Season-based train / val / test split ─────────────────────────────────
FEATURE_COLS = [
    # baseline (cascade inputs)
    "pitcher_last_season_start_pa_avg_pa_per_start",
    "pitcher_last_season_start_pa_n_starts",
    "pitcher_this_season_start_pa_avg_pa_per_start",
    "pitcher_this_season_start_pa_starts_n",
    "team_last_season_avg_pa_per_start",
    "league_last_season_avg_pa_per_start",
    "expected_batters_faced",
    "expected_batters_faced_weight",
    "pitcher_throw_hand",
    # v1 features
    "team_roll_season_walk_rate",
    "team_roll_season_on_base_rate",
    "pitcher_team_days_since_last_game",
    "is_home",
    "pitcher_roll_season_pitch_count_avg",
    # v2 features
    "pitcher_last3_start_pa_avg_pa_per_start",
    "pitcher_last3_start_pa_starts_n",
    "pa_trend_ratio",
    "pa_trend_direction",
    "pitcher_days_since_last_start",
    "pitcher_last_start_pitches",
    "pitcher_workload_density",
    "pitcher_workload_density_shrunk",
    # v5 features
    "opp_team_roll_season_win_pct",
    "opp_team_roll_season_runs_scored",
    "opp_team_roll_season_run_diff",
    "opp_team_roll_last5g_win_pct",
    "opp_team_roll_last5g_runs_scored",
    "opp_team_roll_last5g_run_diff",
    # v6 features
    "opp_team_roll_season_runs_scored_mean",
    "opp_team_roll_season_runs_scored_std",
    "opp_team_roll_season_runs_scored_max",
    "opp_team_roll_last5g_runs_scored_mean",
    "opp_team_roll_last5g_runs_scored_std",
    "opp_team_roll_last5g_runs_scored_max",
    # v7 features (this pass) — pitch-count trend
    "pitcher_roll_last3g_pitch_count_avg",
    "pitch_count_trend_ratio",
    "pitch_count_trend_direction",
    # v7 features (this pass) — own-team bullpen strength
    "bullpen_roll_season_whip",
    "bullpen_roll_season_k_rate",
    "bullpen_roll_season_bb_rate",
    "bullpen_roll_season_hr_rate",
    "bullpen_roll_season_strike_rate",
    "bullpen_roll_last5g_whip",
    "bullpen_roll_last5g_k_rate",
    "bullpen_roll_last5g_bb_rate",
    "bullpen_roll_last5g_hr_rate",
    "bullpen_roll_last5g_strike_rate",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in start_outcome.columns]
CASCADE_COL = "expected_batters_faced"
WEIGHT_COL = "expected_batters_faced_weight"

model_df = start_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].dropna(subset=[TARGET]).copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

core_df  = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_df = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"realized_batters_faced — train mean: {train_df[TARGET].mean():.2f}  val mean: {val_df[TARGET].mean():.2f}")

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_val   = val_df[FEATURE_COLS]
y_val   = val_df[TARGET]

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(X_train[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]


def encode(X_tr, X_ev, cat_cols, num_cols):
    X_tr = X_tr.copy()
    X_ev = X_ev.copy()

    if num_cols:
        X_tr[num_cols] = X_tr[num_cols].apply(pd.to_numeric, errors="coerce")
        X_ev[num_cols] = X_ev[num_cols].apply(pd.to_numeric, errors="coerce")
    if cat_cols:
        X_tr[cat_cols] = X_tr[cat_cols].astype(object).fillna(np.nan)
        X_ev[cat_cols] = X_ev[cat_cols].astype(object).fillna(np.nan)

    num_imp = SimpleImputer(strategy="median")
    Xtr_num = num_imp.fit_transform(X_tr[num_cols]) if num_cols else np.empty((len(X_tr), 0))
    Xev_num = num_imp.transform(X_ev[num_cols])     if num_cols else np.empty((len(X_ev), 0))

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        Xtr_cat_imp = cat_imp.fit_transform(X_tr[cat_cols])
        Xev_cat_imp = cat_imp.transform(X_ev[cat_cols])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr_cat = enc.fit_transform(Xtr_cat_imp)
        Xev_cat = enc.transform(Xev_cat_imp)
    else:
        Xtr_cat = np.empty((len(X_tr), 0))
        Xev_cat = np.empty((len(X_ev), 0))

    return np.hstack([Xtr_num, Xtr_cat]), np.hstack([Xev_num, Xev_cat])


def encode_multi(X_tr, X_evs, cat_cols, num_cols):
    """Same fit-on-train/transform-many-evals shape as encode() above, but
    for the tuned XGBoost's two eval frames (early-stop season, val season)
    off ONE shared fit on core_df."""
    X_tr = X_tr.copy()
    X_evs = [X.copy() for X in X_evs]
    if num_cols:
        X_tr[num_cols] = X_tr[num_cols].apply(pd.to_numeric, errors="coerce")
        for X in X_evs:
            X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
    if cat_cols:
        X_tr[cat_cols] = X_tr[cat_cols].astype(object).fillna(np.nan)
        for X in X_evs:
            X[cat_cols] = X[cat_cols].astype(object).fillna(np.nan)

    num_imp = SimpleImputer(strategy="median")
    Xtr_num = num_imp.fit_transform(X_tr[num_cols]) if num_cols else np.empty((len(X_tr), 0))
    Xevs_num = [num_imp.transform(X[num_cols]) if num_cols else np.empty((len(X), 0)) for X in X_evs]

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        Xtr_cat = cat_imp.fit_transform(X_tr[cat_cols])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr_cat = enc.fit_transform(Xtr_cat)
        Xevs_cat = [enc.transform(cat_imp.transform(X[cat_cols])) for X in X_evs]
    else:
        Xtr_cat = np.empty((len(X_tr), 0))
        Xevs_cat = [np.empty((len(X), 0)) for X in X_evs]

    Xtr = np.hstack([Xtr_num, Xtr_cat])
    Xevs = [np.hstack([n, c]) for n, c in zip(Xevs_num, Xevs_cat)]
    return Xtr, Xevs


Xtr, Xval = encode(X_train, X_val, cat_cols, num_cols)


# ── 5. Train models, evaluate on val ─────────────────────────────────────────
results = {}


def _eval(name, y_true, y_pred):
    error = y_true - y_pred
    results[name] = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "bias": error.mean(),
        "pearson_r": np.corrcoef(y_true, y_pred)[0, 1] if y_pred.std() > 0 else np.nan,
        "pred": y_pred,
    }


print("\nEvaluating the shrinkage cascade (expected_batters_faced) as the benchmark...")
_eval("Cascade (expected_batters_faced)", y_val.to_numpy(), X_val[CASCADE_COL].to_numpy())

print("Training linear regression...")
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xval_sc = scaler.transform(Xval)
lr = LinearRegression()
lr.fit(Xtr_sc, y_train)
_eval("Linear regression (v7)", y_val.to_numpy(), lr.predict(Xval_sc))

print("Training XGBoost (v7, default)...")
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(Xtr, y_train)
_eval("XGBoost (v7, default)", y_val.to_numpy(), xgb_model.predict(Xval))

print("Training XGBoost (v7, tuned)...")
# Hyperparameters and early-stopping setup carried over unchanged from v1-v6.
Xcore, [Xearly_enc, Xval_core] = encode_multi(
    core_df[FEATURE_COLS], [early_df[FEATURE_COLS], val_df[FEATURE_COLS]], cat_cols, num_cols,
)
y_core, y_early = core_df[TARGET], early_df[TARGET]

xgb_tuned = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.02, max_depth=3, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5,
    random_state=42, verbosity=0, early_stopping_rounds=50, eval_metric="mae",
)
xgb_tuned.fit(Xcore, y_core, eval_set=[(Xearly_enc, y_early)], verbose=False)
print(f"  Best iteration (early stopping on {EARLY_STOP_SEASON}): {xgb_tuned.best_iteration}")
_eval("XGBoost (v7, tuned)", y_val.to_numpy(), xgb_tuned.predict(Xval_core))


# ── 6. Print results ──────────────────────────────────────────────────────────
CASCADE_NAME = "Cascade (expected_batters_faced)"
cascade_mae = results[CASCADE_NAME]["mae"]

V2_TUNED_MAE = 2.6471  # v2_workload_and_rest/train.py's own result, for direct comparison

print(f"\n{'=' * 72}")
print(f"EXPERIMENT RESULTS — {MODEL_NAME} v7 (pitch-count trend + own-team bullpen strength)")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} starts)  |  Test season {TEST_SEASON} locked")
print("Primary: MAE (lower=better)  |  Secondary: RMSE, Bias, Pearson r")
print("=" * 72)
print(f"{'Model':<32} {'MAE':>8} {'vs cascade':>12}  {'RMSE':>8} {'Bias':>8} {'Pearson r':>10}")
print("-" * 72)
for name, res in results.items():
    delta = f"{res['mae'] - cascade_mae:+.4f}" if name != CASCADE_NAME else "—"
    print(f"{name:<32} {res['mae']:>8.4f} {delta:>12}  {res['rmse']:>8.4f} {res['bias']:>+8.4f} {res['pearson_r']:>10.4f}")
print("-" * 72)
print(f"{'(v2 tuned XGBoost, for reference)':<32} {V2_TUNED_MAE:>8.4f}")
print("=" * 72)

print("\nInterpretation:")
candidates = {n: r for n, r in results.items() if n != CASCADE_NAME}
beats_floor = {n: r for n, r in candidates.items() if r["mae"] < cascade_mae}
best_model_name, best_model = min(candidates.items(), key=lambda t: t[1]["mae"])

if not beats_floor:
    print(f"  No model beats the cascade floor ({CASCADE_NAME}, MAE {cascade_mae:.4f}).")
else:
    print(f"  {best_model_name} beats the cascade floor ({CASCADE_NAME}, MAE {cascade_mae:.4f}) —")
    print(f"  {best_model_name} MAE {best_model['mae']:.4f}.")

vs_v2 = best_model["mae"] - V2_TUNED_MAE
if vs_v2 < -0.02:
    print(f"  vs v2's tuned XGBoost (MAE {V2_TUNED_MAE:.4f}): {vs_v2:+.4f} — pitch-count trend and/or")
    print("  own-team bullpen strength add real signal beyond v2's feature set.")
elif vs_v2 > 0.02:
    print(f"  vs v2's tuned XGBoost (MAE {V2_TUNED_MAE:.4f}): {vs_v2:+.4f} — worse than v2 — check for a")
    print("  wiring bug before concluding the new features don't help.")
else:
    print(f"  vs v2's tuned XGBoost (MAE {V2_TUNED_MAE:.4f}): {vs_v2:+.4f} — flat, no demonstrated")
    print("  improvement from the new features this pass.")
print("=" * 72)


# ── 6b. Quartile breakdown by expected_batters_faced_weight ──────────────────
print(f"\n{'=' * 72}\nMAE/RMSE/Bias/Pearson r BY {WEIGHT_COL} QUARTILE\n{'=' * 72}")

val_eval = val_df.copy()
val_eval["weight_q"] = pd.qcut(val_eval[WEIGHT_COL], 4, labels=["Q1 (thinnest)", "Q2", "Q3", "Q4 (most reliable)"], duplicates="drop")

quartile_rows = []
print(f"{'weight quartile':<20} {'n':>6} {'model':<32} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Pearson r':>10}")
for q, grp in val_eval.groupby("weight_q", observed=True):
    idx = grp.index
    y_true_q = val_df.loc[idx, TARGET].to_numpy()
    for name, res in results.items():
        pred_full = pd.Series(res["pred"], index=val_df.index)
        y_pred_q = pred_full.loc[idx].to_numpy()
        error_q = y_true_q - y_pred_q
        mae_q = np.abs(error_q).mean()
        rmse_q = np.sqrt((error_q ** 2).mean())
        bias_q = error_q.mean()
        r_q = np.corrcoef(y_true_q, y_pred_q)[0, 1] if y_pred_q.std() > 0 else np.nan
        print(f"{str(q):<20} {len(grp):>6} {name:<32} {mae_q:>8.4f} {rmse_q:>8.4f} {bias_q:>+8.4f} {r_q:>10.4f}")
        quartile_rows.append({
            "weight_quartile": str(q), "n": len(grp), "model": name,
            "mae": mae_q, "rmse": rmse_q, "bias": bias_q, "pearson_r": r_q,
        })


# ── 6c. bf_gap-quartile floor re-check with the NEW estimate ─────────────────
print(f"\n{'=' * 72}\nbf_gap-QUARTILE FLOOR RE-CHECK — {best_model_name} vs the original cascade\n{'=' * 72}")

check_df = val_df.copy()
check_df["cascade_pred"] = results[CASCADE_NAME]["pred"]
check_df["new_pred"] = best_model["pred"]
check_df["cascade_bf_gap"] = (check_df[TARGET] - check_df["cascade_pred"]).abs()
check_df["new_bf_gap"] = (check_df[TARGET] - check_df["new_pred"]).abs()

print(f"\n{'bf_gap quartile (by NEW estimate)':<35} {'n':>6} {'cascade_bf_gap MAE':>20} {'new_bf_gap MAE':>16}")
check_df["new_bf_gap_q"] = pd.qcut(check_df["new_bf_gap"], 4, labels=["Q1 (closest)", "Q2", "Q3", "Q4 (furthest)"], duplicates="drop")
for q, grp in check_df.groupby("new_bf_gap_q", observed=True):
    print(f"{str(q):<35} {len(grp):>6} {grp['cascade_bf_gap'].mean():>20.4f} {grp['new_bf_gap'].mean():>16.4f}")

print(f"\nOverall: cascade bf_gap MAE {check_df['cascade_bf_gap'].mean():.4f}  |  new estimate bf_gap MAE {check_df['new_bf_gap'].mean():.4f}")
print("=" * 72)


# ── 6d. Established-starter bucket re-check (same convention as v2-v6) ──────
print(f"\n{'=' * 72}\nESTABLISHED-STARTER (11+ starts) BUCKET RE-CHECK vs v2\n{'=' * 72}")
established = val_df["pitcher_this_season_start_pa_starts_n"] >= 11
cascade_established_mae = check_df.loc[established, "cascade_bf_gap"].mean()
new_established_mae = check_df.loc[established, "new_bf_gap"].mean()
print(f"n={established.sum()}  cascade MAE {cascade_established_mae:.4f}  {best_model_name} MAE {new_established_mae:.4f}")
print("=" * 72)


# ── 7. Plots ───────────────────────────────────────────────────────────────
PLOT_DIR = STAGE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 4))
train_df[TARGET].plot(kind="hist", bins=30, color="steelblue", ax=ax)
ax.set_title(f"realized_batters_faced distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_xlabel("realized_batters_faced")
ax.set_ylabel("starts")
plt.tight_layout()
plt.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'target_distribution.png'}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_val, best_model["pred"], alpha=0.3, s=12, color="steelblue")
lims = [min(y_val.min(), best_model["pred"].min()), max(y_val.max(), best_model["pred"].max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("realized_batters_faced (actual)")
ax.set_ylabel("predicted")
ax.set_title(f"{best_model_name} predicted vs. actual — val ({VAL_SEASON})")
plt.tight_layout()
plt.savefig(PLOT_DIR / "residuals.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'residuals.png'}")

feature_names = num_cols + cat_cols
importances = pd.Series(xgb_tuned.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, max(4, len(importances) * 0.35)))
ax.barh(importances.index, importances.values, color="steelblue")
ax.set_title("Feature importance — XGBoost (v7, tuned)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'feature_importance.png'}")


# ── 8. MLflow logging ─────────────────────────────────────────────────────────
for name, res in results.items():
    metrics = {"mae": res["mae"], "rmse": res["rmse"], "bias": res["bias"], "pearson_r": res["pearson_r"]}
    artifact_paths = [
        PLOT_DIR / "target_distribution.png",
        PLOT_DIR / "residuals.png",
        PLOT_DIR / "feature_importance.png",
    ]
    params = {
        "model_type": name,
        "n_features": len(FEATURE_COLS),
        "fit_seasons": str(FIT_SEASONS),
    }
    if name == best_model_name:
        params["bf_gap_mae_new"] = check_df["new_bf_gap"].mean()
        params["bf_gap_mae_cascade"] = check_df["cascade_bf_gap"].mean()
        params["established_bucket_mae_new"] = new_established_mae
        params["established_bucket_mae_cascade"] = cascade_established_mae

    log_evaluation_to_mlflow(
        metrics=metrics,
        params=params,
        tags={
            "stage": "v7_pitch_trend_and_bullpen_strength",
            "target": TARGET,
            "val_season": str(VAL_SEASON),
            "git_sha": get_git_sha() or "unknown",
        },
        artifact_paths=artifact_paths,
    )
print(f"\nLogged all runs to MLflow (experiment: {MODEL_NAME}).")


# ── 10. Write v7_results.md ───────────────────────────────────────────────────
md_lines = [
    f"# v7 Results — {MODEL_NAME} (pitch-count trend + own-team bullpen strength)",
    "",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
    "**Task:** Regression (starting-pitcher-start grain)  ",
    "**Target:** realized_batters_faced  ",
    "**Primary metric:** MAE (lower = better)  ",
    "",
    "## Results (evaluated on val)",
    "",
    "| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |",
    "|-------|-----|---------------|------|------|-----------|",
]
for name, res in results.items():
    delta = f"{res['mae'] - cascade_mae:+.4f}" if name != CASCADE_NAME else "—"
    md_lines.append(f"| {name} | {res['mae']:.4f} | {delta} | {res['rmse']:.4f} | {res['bias']:+.4f} | {res['pearson_r']:.4f} |")
md_lines += [
    f"| (v2 tuned XGBoost, for reference) | {V2_TUNED_MAE:.4f} | | | | |",
    "",
    f"## MAE/RMSE/Bias/Pearson r by {WEIGHT_COL} quartile",
    "",
    "| Weight quartile | n | Model | MAE | RMSE | Bias | Pearson r |",
    "|---|---|---|---|---|---|---|",
]
for row in quartile_rows:
    md_lines.append(
        f"| {row['weight_quartile']} | {row['n']} | {row['model']} | {row['mae']:.4f} | "
        f"{row['rmse']:.4f} | {row['bias']:+.4f} | {row['pearson_r']:.4f} |"
    )
md_lines += [
    "",
    "## bf_gap-quartile floor re-check",
    "",
    f"Overall: cascade bf_gap MAE {check_df['cascade_bf_gap'].mean():.4f} | "
    f"new estimate ({best_model_name}) bf_gap MAE {check_df['new_bf_gap'].mean():.4f}",
    "",
    "| bf_gap quartile (by new estimate) | n | cascade_bf_gap MAE | new_bf_gap MAE |",
    "|---|---|---|---|",
]
for q, grp in check_df.groupby("new_bf_gap_q", observed=True):
    md_lines.append(f"| {q} | {len(grp)} | {grp['cascade_bf_gap'].mean():.4f} | {grp['new_bf_gap'].mean():.4f} |")
md_lines += [
    "",
    "## Established-starter (11+ starts) bucket re-check",
    "",
    "Same slice v2-v6 check against — this pass does not specifically target",
    "that failure mode (that thread is closed, see ROADMAP.md's v4 entry), so this",
    "is a sanity re-check for regressions, not the headline result.",
    "",
    f"n={established.sum()} | cascade bf_gap MAE {cascade_established_mae:.4f} | "
    f"{best_model_name} bf_gap MAE {new_established_mae:.4f}",
    "",
    "## Setup",
    "",
    f"- Features: {FEATURE_COLS}",
    "- New this pass (zero new production code — pure composition of",
    "  already-existing, already-tested functions):",
    "  1. PITCH-COUNT TREND: pitcher_roll_last3g_pitch_count_avg via",
    "     rolling_stats.build_pbp_pitcher_rolling_feats(sp_pbp, window=3, ...)",
    "     — the same function v1's season-level pitch_count_avg already uses,",
    "     called again with an int window. pitch_count_trend_ratio/_direction",
    "     computed inline, same pattern as v2's pa_trend_ratio/_direction.",
    "  2. OWN-TEAM BULLPEN STRENGTH: bullpen_roll_{season,last5g}_{whip,",
    "     k_rate,bb_rate,hr_rate,strike_rate} via",
    "     rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore,",
    "     pbp, window) — the same role-aware pooling function k_predictor's",
    "     v1/v2 and n_pa_predictor's baseline already use — filtered to",
    "     pitcher_role == 'bullpen' and joined on the STARTER's own team",
    "     (pitcher_team_id), not the opposing team.",
    "- Hypothesis 1: a pitcher's recent pitch count trending above/below his",
    "  own season baseline (not just the level, already tried in v1) predicts",
    "  an earlier or later hook.",
    "- Hypothesis 2: a manager who trusts a strong bullpen may pull the",
    "  starter earlier regardless of the starter's own workload/performance —",
    "  the first SELF-team feature tried on this model (v1/v5/v6 all looked",
    "  at the OPPOSING team).",
    "- v1-v6 features stay in the feature list unchanged (additive) — this",
    "  experiment's feature set is v6's own plus the 13 new columns above.",
    "- XGBoost (v7, tuned) hyperparameters and its held-out-season",
    "  early-stopping setup are carried over unchanged from v1-v6.",
]
results_path = BASE_DIR / "v7_results.md"
results_path.write_text("\n".join(md_lines) + "\n")
print(f"\nSaved {results_path}")
