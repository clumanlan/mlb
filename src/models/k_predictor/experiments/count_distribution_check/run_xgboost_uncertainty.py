"""
k_predictor: total-strikeout count-distribution + uncertainty report, scored by
v6's chosen production model (tuned XGBoost: max_depth=2, learning_rate=0.03,
fit on CORE_FIT_SEASONS with early stopping against the held-out latest fit
season) instead of run.py's v2 LR.

Run from src/models/k_predictor/ with:
    python experiments/count_distribution_check/run_xgboost_uncertainty.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    SUMMARY                                   #
# ---------------------------------------------------------------------------- #
# run.py already built the plumbing (expected_batters_faced cascade -> synthetic
# batter-slot expansion -> per-slot classifier score -> exact Poisson-binomial
# combination -> nominal-vs-empirical coverage check) scored by v2's LR. This
# sibling script reuses that plumbing UNCHANGED and swaps only two things:
#   1. The scored model: v6's tuned XGBoost (the standing production candidate,
#      picked over LR per its game-grain reliability/resolution edge) on v3's
#      full 42-feature set, instead of v2's 22-feature LR. Fits ONLY the one
#      already-known winning config directly (max_depth=2, learning_rate=0.03)
#      rather than re-running v6's 9-config grid search -- we already know
#      which config wins, no need to pay for the other 8 again.
#   2. What gets reported: instead of one hand-picked covered/miss example
#      pair, this ranks every 2024 start by |predicted_mean_k - realized_k|
#      and pulls several from BOTH tails (best and worst), plus two
#      population-level plots that sit between the aggregate coverage numbers
#      and the individual examples -- predicted-vs-realized K for every start,
#      and |residual| vs. predicted 80%-interval width (does a wider predicted
#      distribution actually track a bigger miss, or is the width uninformative
#      about where the real risk is?).
#
# Everything else (cascade, slot expansion, Poisson-binomial combine, coverage
# check math) is identical to run.py -- see that file's own docstring for the
# fuller rationale on what this pipeline deliberately does and doesn't model.
# ---------------------------------------------------------------------------- #
import json
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import expected_role
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.utils.eval import evaluate_hit_predictor

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip
from models.hit_predictor.utils.count_distribution import poisson_binomial_pmf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WHIP_SHRINKAGE_K = 20.0
PA_SHRINKAGE_K = 5.0
MAX_SLOTS = 45
SHORT_PITCHER_WINDOW = 3
SHORT_TEAM_WINDOW = 5
N_EXAMPLES_PER_TAIL = 5

OUT_DIR = Path(__file__).parent / "xgboost_uncertainty"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Config + load ───────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET, REGION = cfg["bucket"], cfg["region"]
TRAIN_SEASONS, FEATURE_SEASONS = cfg["train_seasons"], cfg["feature_seasons"]
TARGET, DATE_COL = cfg["target_column"], cfg["date_column"]
TEST_SEASON, VAL_SEASON = cfg["test_season"], cfg["val_season"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))


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
pbp = read_parquet_seasons("s3://{bucket}/processed_data/prepared/playbyplay/{season}/", TRAIN_SEASONS, chunked=True)
print("\nLoading schedule...")
schedule = read_parquet_seasons("s3://{bucket}/processed_data/games/schedule/{season}/", all_boxscore_seasons)
print("\nLoading game info...")
game_info = read_parquet_seasons("s3://{bucket}/processed_data/games/game_info/{season}/", TRAIN_SEASONS)
print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", all_boxscore_seasons)
print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", all_boxscore_seasons)
print("\nLoading player info...")
player_info = wr.s3.read_parquet(path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session)


# ── 2. Build PA-grain frame + v3/v6's full 42-feature set ─────────────────────
# Identical to v6_tuned/train.py sections 3-4 -- see that file for the full
# per-feature-family rationale. Kept here rather than imported since train.py
# is a script, not a module (matches this repo's existing convention of each
# experiment being copy-adapted, not shared as a library).
print("\nBuilding PA-grain DataFrame...")
schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)
# build_pitcher_start_pa_this_season (section 4 below) needs game_datetime to
# order same-date doubleheaders correctly -- pbp's own pipeline only ever adds
# game_date, never game_datetime (see run.py's identical comment), so this is
# the first pbp-only caller here and needs its own explicit merge from schedule.
pbp = pbp.merge(schedule[["gamepk", "game_datetime"]], on="gamepk", how="left")

pa_outcome = pipeline.create_pa_outcome_strikeout(pbp, batter_boxscore, game_info, schedule)
pa_outcome["game_season"] = pa_outcome["game_date"].dt.year

pitcher_start_depth_stats = season_stats.build_pitcher_start_depth_stats(pbp)
league_avg_start_depth = season_stats.build_league_avg_start_depth(pitcher_start_depth_stats)
pa_outcome = expected_role.assign_expected_pitcher_role(pa_outcome, pitcher_start_depth_stats, league_avg_start_depth)

pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_pa_strikeout_rate"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})
pitcher_box_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_whip"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})
batter_season_stats = season_stats.build_pbp_batter_feats(pbp)[["batter_id", "game_season", "batter_last_season_pa_strikeout_rate"]]

pa_outcome = pa_outcome.merge(pitcher_role_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left")
pa_outcome = pa_outcome.merge(pitcher_box_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left")
pa_outcome = pa_outcome.merge(batter_season_stats, on=["game_season", "batter_id"], how="left")

pitcher_box_rolling = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window="season")
box_rolling_cols = [
    "pitcher_roll_season_ip", "pitcher_roll_season_whip", "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate", "pitcher_roll_season_hr_rate", "pitcher_roll_season_games_n",
]
pa_outcome = pa_outcome.merge(
    pitcher_box_rolling.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + box_rolling_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

pitcher_pbp_rolling = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window="season")
pbp_rolling_cols = [
    "pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate",
    "pitcher_roll_season_command_swinging_strike_rate", "pitcher_roll_season_pa_pitch_count_mean",
]
pa_outcome = pa_outcome.merge(
    pitcher_pbp_rolling.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + pbp_rolling_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome["pitcher_roll_season_avg_ip_per_game"] = (
    pa_outcome["pitcher_roll_season_ip"] / pa_outcome["pitcher_roll_season_games_n"].replace(0, np.nan)
)

batter_pbp_rolling = rolling_stats.build_pbp_batter_rolling_feats(pbp, window="season")
pa_outcome = pa_outcome.merge(
    batter_pbp_rolling[["batter_id", "gamepk", "batter_roll_season_pa_strikeout_rate"]],
    on=["batter_id", "gamepk"], how="left",
)

team_batter_rolling = rolling_stats.build_team_batter_strikeout_rolling_feats(pbp, batter_boxscore, window="season")
opp_team_rolling = team_batter_rolling.rename(columns={
    "team_roll_season_pa_strikeout_rate": "opp_team_roll_season_pa_strikeout_rate",
})[["batter_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")

pitching_team_rolling = team_batter_rolling.rename(columns={
    "batter_team_id": "pitcher_team_id",
    "team_roll_season_pa_strikeout_rate": "pitching_team_roll_season_pa_strikeout_rate",
})[["pitcher_team_id", "gamepk", "pitching_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")

# trailing-3-start pitcher form (v3 #1)
pitcher_box_rolling3 = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window=SHORT_PITCHER_WINDOW)
box_rolling3_cols = [
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_whip",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_k_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_bb_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_hr_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n",
]
pa_outcome = pa_outcome.merge(
    pitcher_box_rolling3.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + box_rolling3_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pitcher_pbp_rolling3 = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window=SHORT_PITCHER_WINDOW)
pbp_rolling3_cols = [
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_total", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_strikeout_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_command_swinging_strike_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean",
]
pa_outcome = pa_outcome.merge(
    pitcher_pbp_rolling3.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + pbp_rolling3_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_avg_ip_per_game"] = (
    pa_outcome[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip"]
    / pa_outcome[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n"].replace(0, np.nan)
)

# opposing-team K-rate volatility, season + trailing-5 (v3 #2)
opp_team_volatility_season = rolling_stats.build_team_strikeout_volatility(pbp, batter_boxscore, window="season")
pa_outcome = pa_outcome.merge(
    opp_team_volatility_season.rename(columns={
        "team_roll_season_pa_strikeout_rate_mean": "opp_team_roll_season_pa_strikeout_rate_mean",
        "team_roll_season_pa_strikeout_rate_std": "opp_team_roll_season_pa_strikeout_rate_std",
        "team_roll_season_pa_strikeout_rate_max": "opp_team_roll_season_pa_strikeout_rate_max",
    })[["batter_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate_mean",
        "opp_team_roll_season_pa_strikeout_rate_std", "opp_team_roll_season_pa_strikeout_rate_max"]],
    on=["batter_team_id", "gamepk"], how="left",
)
opp_team_volatility_short = rolling_stats.build_team_strikeout_volatility(pbp, batter_boxscore, window=SHORT_TEAM_WINDOW)
short_team_prefix = f"team_roll_last{SHORT_TEAM_WINDOW}g_pa_strikeout_rate"
pa_outcome = pa_outcome.merge(
    opp_team_volatility_short.rename(columns={
        f"{short_team_prefix}_mean": f"opp_{short_team_prefix}_mean",
        f"{short_team_prefix}_std": f"opp_{short_team_prefix}_std",
        f"{short_team_prefix}_max": f"opp_{short_team_prefix}_max",
    })[["batter_team_id", "gamepk", f"opp_{short_team_prefix}_mean",
        f"opp_{short_team_prefix}_std", f"opp_{short_team_prefix}_max"]],
    on=["batter_team_id", "gamepk"], how="left",
)

shrunk_whip = build_pitcher_shrunk_whip(
    pitcher_box_rolling, season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp), window="season", k=WHIP_SHRINKAGE_K,
)
pa_outcome = pa_outcome.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role", "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

FEATURE_COLS = [
    "expected_pitcher_role", "pitcher_throw_hand", "batter_bat_side",
    "pitcher_last_season_pa_strikeout_rate", "pitcher_last_season_whip", "batter_last_season_pa_strikeout_rate",
    "pitcher_roll_season_ip", "pitcher_roll_season_whip", "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate", "pitcher_roll_season_hr_rate", "pitcher_roll_season_games_n",
    "pitcher_roll_season_avg_ip_per_game", "pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate",
    "batter_roll_season_pa_strikeout_rate", "opp_team_roll_season_pa_strikeout_rate", "pitching_team_roll_season_pa_strikeout_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_whip",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_k_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_bb_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_hr_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_avg_ip_per_game", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_total",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_strikeout_rate",
    "opp_team_roll_season_pa_strikeout_rate_mean", "opp_team_roll_season_pa_strikeout_rate_std",
    "opp_team_roll_season_pa_strikeout_rate_max",
    f"opp_{short_team_prefix}_mean", f"opp_{short_team_prefix}_std", f"opp_{short_team_prefix}_max",
    "pitcher_roll_season_command_swinging_strike_rate", "pitcher_roll_season_pa_pitch_count_mean",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_command_swinging_strike_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean",
    "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight",
    "weather_condition", "weather_temp", "expected_times_through_order",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in pa_outcome.columns]
print(f"Feature count: {len(FEATURE_COLS)} (v3/v6's set)")

model_df = pa_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(model_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

EARLY_STOP_SEASON = max(FIT_SEASONS)
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]
print(f"Core fit seasons: {CORE_FIT_SEASONS}  |  early-stop season: {EARLY_STOP_SEASON}  |  val: {VAL_SEASON}")

core_train_df = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_stop_df = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()


# ── 3. Fit v6's winning XGBoost config directly (no grid search — already know
# the winner: max_depth=2, learning_rate=0.03) ────────────────────────────────
def fit_transform(fit_df, cols):
    fit_df = fit_df[cols].copy()
    n_cols = [c for c in cols if c in num_cols]
    c_cols = [c for c in cols if c in cat_cols]
    if n_cols:
        fit_df[n_cols] = fit_df[n_cols].apply(pd.to_numeric, errors="coerce")
    if c_cols:
        fit_df[c_cols] = fit_df[c_cols].astype(object).fillna(np.nan)
    return fit_df


num_imp = SimpleImputer(strategy="median")
Xcore_num = num_imp.fit_transform(fit_transform(core_train_df, num_cols)) if num_cols else np.empty((len(core_train_df), 0))
if cat_cols:
    cat_imp = SimpleImputer(strategy="most_frequent")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xcore_cat = enc.fit_transform(cat_imp.fit_transform(fit_transform(core_train_df, cat_cols)))
else:
    cat_imp, enc = None, None
    Xcore_cat = np.empty((len(core_train_df), 0))
Xcore = np.hstack([Xcore_num, Xcore_cat])
y_core = core_train_df[TARGET]


def transform(df):
    df = df[FEATURE_COLS].copy()
    x_num = num_imp.transform(fit_transform(df, num_cols)) if num_cols else np.empty((len(df), 0))
    if cat_cols:
        x_cat = enc.transform(cat_imp.transform(fit_transform(df, cat_cols)))
    else:
        x_cat = np.empty((len(df), 0))
    return np.hstack([x_num, x_cat])


Xearly = transform(early_stop_df)
y_early = early_stop_df[TARGET]

print("\nFitting XGBoost (max_depth=2, learning_rate=0.03, early-stopped)...")
best_xgb = xgb.XGBClassifier(
    n_estimators=2000, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, verbosity=0, eval_metric="logloss", early_stopping_rounds=30,
)
best_xgb.fit(Xcore, y_core, eval_set=[(Xearly, y_early)], verbose=False)
print(f"best_iteration={best_xgb.best_iteration}")


def score(df_feat):
    """Apply the SAME fitted imputers/encoder/model to any frame with
    FEATURE_COLS columns — real PA rows or synthetic slot rows alike."""
    return best_xgb.predict_proba(transform(df_feat))[:, 1]


# ── 4. Expected batters faced, val season 2024 SP starts only (unchanged from
# run.py, extended with the additional v3 game-level features) ────────────────
print("\nBuilding expected_batters_faced cascade...")
pitcher_start_pa_last_season = season_stats.build_pitcher_start_pa_stats(pbp)
league_avg_start_pa = season_stats.build_league_avg_start_pa(pitcher_start_pa_last_season)
team_avg_start_pa = season_stats.build_team_avg_start_pa(pitcher_start_pa_last_season)
pitcher_start_pa_this_season = game_context.build_pitcher_start_pa_this_season(pbp)

expected_pa = game_context.build_expected_batters_faced(
    pitcher_start_pa_last_season, pitcher_start_pa_this_season, team_avg_start_pa, league_avg_start_pa, k=PA_SHRINKAGE_K,
)
pitcher_starts_2024 = expected_pa[expected_pa["game_season"] == VAL_SEASON].copy()
print(f"2024 SP starts with an expected_batters_faced estimate: {len(pitcher_starts_2024):,}")

pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_role_season_stats.rename(columns={"expected_pitcher_key_id": "personId"})[
        ["personId", "game_season", "pitcher_last_season_pa_strikeout_rate"]
    ],
    on=["personId", "game_season"], how="left",
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_box_season_stats.rename(columns={"expected_pitcher_key_id": "personId"})[["personId", "game_season", "pitcher_last_season_whip"]],
    on=["personId", "game_season"], how="left",
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_box_rolling.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + box_rolling_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_pbp_rolling.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + pbp_rolling_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_2024["pitcher_roll_season_avg_ip_per_game"] = (
    pitcher_starts_2024["pitcher_roll_season_ip"] / pitcher_starts_2024["pitcher_roll_season_games_n"].replace(0, np.nan)
)
pitcher_starts_2024 = pitcher_starts_2024.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")
pitcher_starts_2024 = pitcher_starts_2024.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId", "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]],
    on=["gamepk", "personId"], how="left",
)

# opp_team_id: the volatility features below are keyed by the OPPOSING
# (batting) team, not the pitcher's own team -- pitcher_starts_2024 only
# carries pitcher_team_id, so derive the opponent the same way
# batters_faced_predictor's residual_error_analysis scripts already do
# (pitcher_team_id == home_id -> opponent is away_id, else home_id).
schedule_teams = schedule[["gamepk", "home_id", "away_id"]].drop_duplicates("gamepk")
pitcher_starts_2024 = pitcher_starts_2024.merge(schedule_teams, on="gamepk", how="left")
pitcher_starts_2024["opp_team_id"] = np.where(
    pitcher_starts_2024["pitcher_team_id"] == pitcher_starts_2024["home_id"],
    pitcher_starts_2024["away_id"], pitcher_starts_2024["home_id"],
)

pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_box_rolling3.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + box_rolling3_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_pbp_rolling3.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + pbp_rolling3_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_2024[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_avg_ip_per_game"] = (
    pitcher_starts_2024[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip"]
    / pitcher_starts_2024[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n"].replace(0, np.nan)
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    opp_team_volatility_season.rename(columns={
        "batter_team_id": "opp_team_id",
        "team_roll_season_pa_strikeout_rate_mean": "opp_team_roll_season_pa_strikeout_rate_mean",
        "team_roll_season_pa_strikeout_rate_std": "opp_team_roll_season_pa_strikeout_rate_std",
        "team_roll_season_pa_strikeout_rate_max": "opp_team_roll_season_pa_strikeout_rate_max",
    })[
        ["opp_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate_mean",
         "opp_team_roll_season_pa_strikeout_rate_std", "opp_team_roll_season_pa_strikeout_rate_max"]
    ],
    on=["opp_team_id", "gamepk"], how="left",
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    opp_team_volatility_short.rename(columns={
        "batter_team_id": "opp_team_id",
        f"{short_team_prefix}_mean": f"opp_{short_team_prefix}_mean",
        f"{short_team_prefix}_std": f"opp_{short_team_prefix}_std",
        f"{short_team_prefix}_max": f"opp_{short_team_prefix}_max",
    })[
        ["opp_team_id", "gamepk", f"opp_{short_team_prefix}_mean",
         f"opp_{short_team_prefix}_std", f"opp_{short_team_prefix}_max"]
    ],
    on=["opp_team_id", "gamepk"], how="left",
)
throw_hand = pbp[["pitcher_id", "gamepk", "pitcher_throw_hand"]].drop_duplicates().rename(columns={"pitcher_id": "personId"})
pitcher_starts_2024 = pitcher_starts_2024.merge(throw_hand, on=["personId", "gamepk"], how="left")
weather = game_info[["gamepk", "weather_condition", "weather_temp"]].drop_duplicates("gamepk")
pitcher_starts_2024 = pitcher_starts_2024.merge(weather, on="gamepk", how="left")
pitcher_starts_2024["expected_pitcher_role"] = "sp"


# ── 5. Expand to synthetic batter slots, attach batter + opp-team features ────
# FIXED 2026-09-02 (same bug/fix as score_2026_test_dates.py, see ROADMAP.md
# item 6(e) / v13_results.md): build_batter_slot_expansion merges
# batting_order onto (gamepk, lineup_position) with NO team-awareness. Every
# gamepk here has TWO starts (home SP, away SP), each needing a DIFFERENT
# team's 9 batters -- passing the full unscoped batting_order (as this script
# always has) let roughly half of all synthetic slots collide against the
# WRONG team (a pitcher's own teammates, whom he never actually faces).
# Fixed here by team-scoping batting_order per start -- expand home-team and
# away-team starters separately (each unambiguous against the correctly
# opposing lineup), then concatenate.
print("Expanding to synthetic batter slots...")
batting_order = hp_pipeline._create_batting_order(batter_boxscore)
batter_team_lookup = batter_boxscore[["gamepk", "personId", "team_id"]].rename(
    columns={"personId": "batter_id", "team_id": "batter_team_id"}
).drop_duplicates()

batting_order_with_team = batting_order.merge(
    batter_team_lookup.rename(columns={"batter_team_id": "team_id"}), on=["gamepk", "batter_id"], how="left",
)


def opp_scoped_batting_order(starts_subset):
    context = starts_subset[["gamepk", "opp_team_id"]].drop_duplicates()
    scoped = context.merge(batting_order_with_team, on="gamepk", how="left")
    scoped = scoped[scoped["team_id"] == scoped["opp_team_id"]]
    return scoped[["gamepk", "batter_id", "batting_order"]]


starts_home_sp = pitcher_starts_2024[pitcher_starts_2024["pitcher_team_id"] == pitcher_starts_2024["home_id"]]
starts_away_sp = pitcher_starts_2024[pitcher_starts_2024["pitcher_team_id"] == pitcher_starts_2024["away_id"]]

slots_home = game_context.build_batter_slot_expansion(starts_home_sp, opp_scoped_batting_order(starts_home_sp), max_slots=MAX_SLOTS)
slots_away = game_context.build_batter_slot_expansion(starts_away_sp, opp_scoped_batting_order(starts_away_sp), max_slots=MAX_SLOTS)
slots = pd.concat([slots_home, slots_away], ignore_index=True)
slots = slots.merge(batter_team_lookup, on=["gamepk", "batter_id"], how="left")

bat_side = pbp[["batter_id", "gamepk", "batter_bat_side"]].drop_duplicates()
slots = slots.merge(bat_side, on=["batter_id", "gamepk"], how="left")
slots = slots.merge(batter_season_stats, on=["batter_id", "game_season"], how="left")
slots = slots.merge(
    batter_pbp_rolling[["batter_id", "gamepk", "batter_roll_season_pa_strikeout_rate"]],
    on=["batter_id", "gamepk"], how="left",
)
slots = slots.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")

print(f"{len(pitcher_starts_2024):,} starts -> {len(slots):,} synthetic batter-slots "
      f"(mean {len(slots) / max(len(pitcher_starts_2024), 1):.1f} slots/start)")


# ── 6. Score each slot, combine into a total-K distribution per start ─────────
print("Scoring synthetic slots with XGBoost...")
slots["k_prob"] = score(slots)

print("Combining via exact Poisson-binomial...")
results = []
for (gamepk, person_id), grp in slots.groupby(["gamepk", "personId"]):
    probs = grp["k_prob"].to_numpy()
    pmf = poisson_binomial_pmf(list(probs))
    results.append({
        "gamepk": gamepk, "personId": person_id, "n_slots": len(probs),
        "predicted_mean_k": probs.sum(), "pmf": pmf,
    })
pred_df = pd.DataFrame(results)


# ── 7. Compare against realized total K + realized batters faced ──────────────
print("Comparing against realized outcomes...")
role_lookup = season_stats._pitcher_role_lookup(pbp)[["gamepk", "pitcher_id", "pitcher_role"]].rename(columns={"pitcher_id": "personId"})
pitcher_box_tagged = pitcher_boxscore.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
).merge(
    role_lookup.assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str)),
    on=["gamepk", "personId"], how="left",
)
realized_k = pitcher_box_tagged[
    (pitcher_box_tagged["pitcher_role"] == "sp") & (pitcher_box_tagged["game_season"] == VAL_SEASON)
][["personId", "gamepk", "k"]].rename(columns={"k": "realized_k"})

sp_pbp = pbp[pbp["pitcher_role"] == "sp"]
realized_pa = rolling_stats._pitcher_pa_outcome_per_game(sp_pbp, entity_col="pitcher_id")[
    ["pitcher_id", "gamepk", "game_season", "pa_total"]
].rename(columns={"pitcher_id": "personId", "pa_total": "realized_batters_faced"})
realized_pa = realized_pa[realized_pa["game_season"] == VAL_SEASON]

pred_df = pred_df.merge(realized_k, on=["personId", "gamepk"], how="inner")
pred_df = pred_df.merge(realized_pa[["personId", "gamepk", "realized_batters_faced"]], on=["personId", "gamepk"], how="left")
pred_df = pred_df.merge(
    pitcher_starts_2024[["personId", "gamepk", "expected_batters_faced", "expected_batters_faced_weight"]],
    on=["personId", "gamepk"], how="left",
)
pred_df["bf_gap"] = (pred_df["realized_batters_faced"] - pred_df["expected_batters_faced"]).abs()
pred_df["residual"] = pred_df["predicted_mean_k"] - pred_df["realized_k"]
pred_df["abs_error"] = pred_df["residual"].abs()

pred_df["weight_q"] = pd.qcut(pred_df["expected_batters_faced_weight"], 4, labels=["Q1 (thinnest)", "Q2", "Q3", "Q4 (most reliable)"], duplicates="drop")

print(f"\n{'=' * 72}\nTOTAL-K COUNT-DISTRIBUTION CHECK (XGBoost) — val season {VAL_SEASON}, {len(pred_df):,} starts\n{'=' * 72}")
print(f"Predicted mean K:  {pred_df['predicted_mean_k'].mean():.3f}")
print(f"Realized mean K:   {pred_df['realized_k'].mean():.3f}")
print(f"MAE (predicted mean vs. realized): {pred_df['abs_error'].mean():.3f}")

LINE = round(pred_df["realized_k"].median()) + 0.5


def p_over_line(pmf, line):
    k_over = int(np.floor(line)) + 1
    return pmf[k_over:].sum() if k_over < len(pmf) else 0.0


pred_df["p_over_line"] = pred_df["pmf"].apply(lambda pmf: p_over_line(pmf, LINE))
pred_df["realized_over_line"] = (pred_df["realized_k"] > LINE).astype(int)

print(f"\n{'=' * 72}\nTHRESHOLD CHECK — P(total K > {LINE}) vs. realized\n{'=' * 72}")
threshold_metrics = evaluate_hit_predictor(
    y_true=pred_df["realized_over_line"], y_prob=pred_df["p_over_line"],
    n_bins=8, min_n=30, base_rate=pred_df["realized_over_line"].mean(),
)


# ── 8. Coverage check — is the predicted distribution honestly wide? ──────────
def interval_bounds(pmf, level):
    cdf = np.cumsum(pmf)
    alpha = (1 - level) / 2
    lower = int(np.searchsorted(cdf, alpha, side="left"))
    upper = int(np.searchsorted(cdf, 1 - alpha, side="left"))
    return lower, upper


LEVELS = [0.50, 0.80, 0.95]
print(f"\n{'=' * 72}\nCOVERAGE CHECK\n{'=' * 72}")
print(f"{'level (nominal)':<18} {'n':>6} {'empirical coverage':>20} {'gap (empirical - nominal)':>28}")
for level in LEVELS:
    bounds = pred_df["pmf"].apply(lambda pmf: interval_bounds(pmf, level))
    lower = bounds.apply(lambda t: t[0])
    upper = bounds.apply(lambda t: t[1])
    covered = (pred_df["realized_k"] >= lower) & (pred_df["realized_k"] <= upper)
    pred_df[f"covered_{int(level * 100)}"] = covered
    pred_df[f"lower_{int(level * 100)}"] = lower
    pred_df[f"upper_{int(level * 100)}"] = upper
    emp = covered.mean()
    print(f"{level:<18.0%} {len(pred_df):>6} {emp:>20.1%} {emp - level:>+28.1%}")

pred_df["interval_width_80"] = pred_df["upper_80"] - pred_df["lower_80"]

coverage_by_level = {str(int(l * 100)): round(float(pred_df[f"covered_{int(l*100)}"].mean()), 4) for l in LEVELS}
coverage_by_weight_quartile_80 = {
    str(q): round(float(grp["covered_80"].mean()), 4) for q, grp in pred_df.groupby("weight_q", observed=True)
}


# ── 9. Population-level plots (the "middle" between aggregate numbers and ─────
# individual examples): predicted vs. realized for every start, and whether
# interval width actually tracks the size of the miss.
print("\nBuilding population-level plots...")

fig, ax = plt.subplots(figsize=(7, 7))
colors = {"Q1 (thinnest)": "#d62728", "Q2": "#ff7f0e", "Q3": "#1f77b4", "Q4 (most reliable)": "#2ca02c"}
for q, grp in pred_df.groupby("weight_q", observed=True):
    ax.scatter(grp["predicted_mean_k"], grp["realized_k"], s=14, alpha=0.5,
               label=str(q), color=colors.get(str(q), "gray"))
lims = [0, max(pred_df["predicted_mean_k"].max(), pred_df["realized_k"].max()) + 1]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Predicted mean K (pmf expectation)")
ax.set_ylabel("Realized K")
ax.set_title(f"Predicted vs. realized strikeouts — every {VAL_SEASON} SP start (XGBoost)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "predicted_vs_realized_scatter.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(pred_df["interval_width_80"], pred_df["abs_error"], s=14, alpha=0.4, color="#1f77b4")
corr_width_error = pred_df[["interval_width_80", "abs_error"]].corr().iloc[0, 1]
ax.set_xlabel("Predicted 80% interval width (K)")
ax.set_ylabel("|residual| = |predicted mean K - realized K|")
ax.set_title(f"Does a wider predicted distribution track a bigger miss?\ncorr = {corr_width_error:.3f}")
plt.tight_layout()
plt.savefig(PLOT_DIR / "residual_vs_interval_width.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(pred_df["residual"], bins=30, color="#1f77b4", edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Residual (predicted mean K - realized K)")
ax.set_ylabel("Starts")
ax.set_title(f"Residual distribution — {VAL_SEASON} SP starts (XGBoost)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "residual_histogram.png", dpi=130)
plt.close()

print(f"corr(interval_width_80, abs_error) = {corr_width_error:.3f}")


# ── 10. Down to examples — ranked by residual, both tails ────────────────────
print("\nSelecting worked examples by residual (best + worst)...")
name_lookup = pitcher_boxscore[["personId", "player_name"]].drop_duplicates("personId")
examples_df = pred_df.merge(name_lookup, on="personId", how="left")
examples_df = examples_df[examples_df["n_slots"] >= 10].copy()  # drop unusably-short outings from the visual set

best = examples_df.nsmallest(N_EXAMPLES_PER_TAIL, "abs_error")
worst = examples_df.nlargest(N_EXAMPLES_PER_TAIL, "abs_error")


def render_example(row, tag):
    pmf = np.array(row["pmf"])
    ks = np.arange(len(pmf))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ks, pmf, color="#1f77b4", alpha=0.8)
    ax.axvline(row["realized_k"], color="#d62728", linewidth=2, label=f"Realized K = {int(row['realized_k'])}")
    ax.axvline(row["predicted_mean_k"], color="black", linestyle="--", linewidth=1,
               label=f"Predicted mean = {row['predicted_mean_k']:.2f}")
    ax.set_xlabel("Total strikeouts (K)")
    ax.set_ylabel("Predicted probability")
    ax.set_title(f"{row['player_name']} — {row['gamepk']}\n"
                 f"|residual|={row['abs_error']:.2f}, 80% interval=[{int(row['lower_80'])},{int(row['upper_80'])}], "
                 f"weight={row['expected_batters_faced_weight']:.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"example_{tag}_{row['personId']}_{row['gamepk']}.png"
    plt.savefig(PLOT_DIR / fname, dpi=130)
    plt.close()
    return fname


def example_record(row, tag, fname):
    return {
        "tag": tag, "plot": fname, "player_name": row["player_name"],
        "gamepk": str(row["gamepk"]), "personId": str(row["personId"]),
        "n_slots": int(row["n_slots"]), "predicted_mean_k": round(float(row["predicted_mean_k"]), 2),
        "realized_k": int(row["realized_k"]), "abs_error": round(float(row["abs_error"]), 3),
        "lower_80": int(row["lower_80"]), "upper_80": int(row["upper_80"]),
        "covered_80": bool(row["covered_80"]),
        "expected_batters_faced_weight": round(float(row["expected_batters_faced_weight"]), 3),
    }


example_records = []
for i, row in enumerate(best.itertuples(index=False), start=1):
    row = row._asdict() if hasattr(row, "_asdict") else dict(zip(best.columns, row))
    fname = render_example(row, f"best{i}")
    example_records.append(example_record(row, "best", fname))
for i, row in enumerate(worst.itertuples(index=False), start=1):
    row = row._asdict() if hasattr(row, "_asdict") else dict(zip(worst.columns, row))
    fname = render_example(row, f"worst{i}")
    example_records.append(example_record(row, "worst", fname))


# ── 11. Persist everything the report needs ────────────────────────────────────
summary = {
    "val_season": VAL_SEASON,
    "n_starts": int(len(pred_df)),
    "best_iteration": int(best_xgb.best_iteration),
    "predicted_mean_k": round(float(pred_df["predicted_mean_k"].mean()), 3),
    "realized_mean_k": round(float(pred_df["realized_k"].mean()), 3),
    "mae": round(float(pred_df["abs_error"].mean()), 3),
    "coverage_by_level": coverage_by_level,
    "coverage_by_weight_quartile_80": coverage_by_weight_quartile_80,
    "corr_interval_width_vs_abs_error": round(float(corr_width_error), 4),
    "threshold_check": {
        "line": LINE,
        "reliability": round(float(threshold_metrics["reliability"]), 4),
        "resolution": round(float(threshold_metrics["resolution"]), 4),
        "roc_auc": round(float(threshold_metrics["roc_auc"]), 4),
        "pr_auc": round(float(threshold_metrics["pr_auc"]), 4),
    },
    "examples": example_records,
}
out_path = OUT_DIR / "uncertainty_summary.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
pred_df.drop(columns=["pmf"]).to_parquet(OUT_DIR / "pred_df.parquet", index=False)
print(f"\nSummary written to {out_path}")
print(f"Plots written to {PLOT_DIR}")
