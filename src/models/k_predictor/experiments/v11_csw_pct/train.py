"""
k_predictor experiment v11: CSW% (called-strike rate + swinging-strike rate)
-- on top of v10's count-leverage/park-K feature set.
Run from src/models/k_predictor/ with: python experiments/v11_csw_pct/train.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    v11 SUMMARY                               #
# ---------------------------------------------------------------------------- #
# A residual-correlation screen (throwaway analysis, not a TDD build) tested 8
# candidate features against v6's held-out val-season residuals BEFORE
# committing any of them to a real build -- the recipe: get out-of-fold
# residuals from the current best model, hold each candidate OUT of training,
# correlate it against the residual (Spearman + mutual info), then check
# whether the bottom-decile-to-top-decile residual movement is monotonic (real
# signal) or flat (likely spurious despite a nonzero correlation).
#
# CSW% -- command_called_strike_rate + command_swinging_strike_rate -- came
# back the clear standout: Spearman r = -0.1705 against the residual, ~4x the
# next-strongest of the 8 candidates, with a clean monotonic decile shape
# (bottom-decile residual +0.0020 -> top-decile -0.0113). The called-strike-
# rate component ALONE is weak in isolation (r = 0.0341) -- the signal lives
# in the combination, not either half, exactly matching sabermetrics
# literature's finding that CSW% (not called-strike rate or whiff rate alone)
# is the strongest single predictor of K% (R^2 ~= 0.59). Six pitch-tunneling/
# arsenal candidates from the same screen were parked: weak signal, two showed
# a flat decile shape despite a decent-looking correlation (likely spurious),
# plus a 58.7% data-coverage gap (no Statcast pitch-mix history for rookies/
# thin-sample pitchers) -- not part of this experiment.
#
# Both components already exist in this project -- CSW% is their SUM, not new
# feature engineering:
#   - ROLLING (what the screen actually tested): build_pbp_pitcher_rolling_feats
#     already computes command_called_strike_rate and command_swinging_strike_rate
#     from already-rolled counts. New command_csw_rate column, TDD'd in
#     tests/hit_predictor/test_rolling_stats.py (2 new tests: the sum
#     invariant, and propagation through build_pbp_pitcher_rolling_feats_all_roles).
#   - SEASON/LAST-SEASON (added for symmetry per user's scope decision -- NOT
#     independently screened, so this run is what confirms or contradicts its
#     signal): season_stats.py's _create_pitcher_stuff_command_stats has the
#     same two components at season grain. New command_csw_rate column, TDD'd
#     in tests/hit_predictor/test_season_stats.py (2 new tests, same shape).
#
# Carries forward v6's exact winning hyperparameters (LR: C=0.1/L1/no class
# weighting; XGBoost: max_depth=2, learning_rate=0.03) and v10's full feature
# set (114 columns) unchanged, per this project's established convention.
# Evaluated the same way as v1-v10: PR-AUC primary vs best naive floor (PA
# grain), then the game-grain aggregation check, feature importance read off
# for both new CSW% columns specifically.
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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, ConfusionMatrixDisplay, confusion_matrix

import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
from models.k_predictor.utils.mlflow_logging import log_evaluation_to_mlflow, get_git_sha

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import expected_role
from models.hit_predictor.processing.features import interaction_feats
from models.hit_predictor.processing.features import park_factors
from models.hit_predictor.utils.eval import (
    run_pa_vs_game_grain_check, aggregate_pa_predictions_to_game,
    evaluate_hit_predictor, summarize_verdict, plot_calibration_curve,
)

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip
from models.k_predictor.processing.features.batter_workload import (
    build_batter_shrunk_k_rate, build_batter_shrunk_obp_slg, build_opposing_lineup_extremum,
)

STAGE = Path(__file__).parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent

WHIP_SHRINKAGE_K = 20.0
BATTER_SHRINKAGE_K = 50.0  # matches the batter-shrinkage precedent set in
                           # experiments/count_distribution_check/run_naive_batter_uncertainty.py
SHORT_PITCHER_WINDOW = 3   # trailing-3-start pitcher form
SHORT_TEAM_WINDOW = 5      # trailing-5-game opposing-lineup volatility

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
FIT_SEASONS.remove(2017)  # same reason as baseline/hit_predictor — needs a prior season's pbp for the shift

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

print("\nLoading game info...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/", TRAIN_SEASONS,
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


# ── 3. Build PA-grain DataFrame ───────────────────────────────────────────────
print("\nBuilding PA-grain DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome_strikeout(pbp, batter_boxscore, game_info, schedule)
pa_outcome["game_season"] = pa_outcome["game_date"].dt.year

# ------------------------- EXPECTED (PRE-GAME) PITCHER ROLE ------------------ #
pitcher_start_depth_stats = season_stats.build_pitcher_start_depth_stats(pbp)
league_avg_start_depth = season_stats.build_league_avg_start_depth(pitcher_start_depth_stats)
pa_outcome = expected_role.assign_expected_pitcher_role(
    pa_outcome, pitcher_start_depth_stats, league_avg_start_depth
)
assert "expected_times_through_order" in pa_outcome.columns, (
    "expected_role.assign_expected_pitcher_role should already produce this column"
)

# ------------------------- 1. SEASON (LAST SEASON) --------------------------- #
# NEW in v11: widened selection to include pitcher_last_season_command_csw_rate
# (already computed by build_pbp_pitcher_feats_all_roles, never selected before).
pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_pa_strikeout_rate",
     "pitcher_last_season_command_csw_rate"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})

pitcher_box_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_whip"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})

# Full (unsliced) frame kept for the K-rate shrinkage builder below — the
# sliced batter_season_stats used for the merge just below still needs its
# own copy of the columns it actually merges.
batter_pbp_season_full = season_stats.build_pbp_batter_feats(pbp)
batter_season_stats = batter_pbp_season_full[["batter_id", "game_season", "batter_last_season_pa_strikeout_rate"]]

pa_outcome = pa_outcome.merge(
    pitcher_role_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome = pa_outcome.merge(
    pitcher_box_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome = pa_outcome.merge(batter_season_stats, on=["game_season", "batter_id"], how="left")

assert "pitcher_last_season_command_csw_rate" in pa_outcome.columns, (
    "pitcher_last_season_command_csw_rate missing from pa_outcome — merge failed upstream"
)

# ------------------------- 2. ROLLING RAW (THIS SEASON, from v1/v2) ---------- #
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
    # NEW in v9 — widened selection (zero new feature-engineering, these are
    # already computed by build_pbp_pitcher_rolling_feats_all_roles above) so
    # find_player_vs_league_pairs has a pitcher-side column to match against
    # every v8 league PA-outcome category, not just strikeout rate.
    "pitcher_roll_season_pa_walk_rate", "pitcher_roll_season_pa_hbp_rate",
    "pitcher_roll_season_pa_single_rate", "pitcher_roll_season_pa_xbh_rate",
    "pitcher_roll_season_pa_hr_rate",
    # NEW in v10 — count-leverage/put-away rate.
    "pitcher_roll_season_pa_two_strike_reach_rate", "pitcher_roll_season_pa_put_away_rate",
    # NEW in v11 — CSW% (called-strike + swinging-strike rate), the confirmed
    # standout from the residual-correlation screen.
    "pitcher_roll_season_command_csw_rate",
]
pa_outcome = pa_outcome.merge(
    pitcher_pbp_rolling.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + pbp_rolling_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

assert "pitcher_roll_season_command_csw_rate" in pa_outcome.columns, (
    "pitcher_roll_season_command_csw_rate missing from pa_outcome — merge failed upstream"
)

pa_outcome["pitcher_roll_season_avg_ip_per_game"] = (
    pa_outcome["pitcher_roll_season_ip"] / pa_outcome["pitcher_roll_season_games_n"].replace(0, np.nan)
)

batter_pbp_rolling = rolling_stats.build_pbp_batter_rolling_feats(pbp, window="season")
# NEW in v9 — widened selection, same reasoning as the pitcher side above:
# build_pbp_batter_rolling_feats already computes all six PA-outcome
# categories, only strikeout rate was previously selected into pa_outcome.
batter_pbp_rolling_cols = [
    "batter_roll_season_pa_strikeout_rate", "batter_roll_season_pa_walk_rate",
    "batter_roll_season_pa_hbp_rate", "batter_roll_season_pa_single_rate",
    "batter_roll_season_pa_xbh_rate", "batter_roll_season_pa_hr_rate",
    # NEW in v10 — count-leverage/put-away rate.
    "batter_roll_season_pa_two_strike_reach_rate", "batter_roll_season_pa_put_away_rate",
]
pa_outcome = pa_outcome.merge(
    batter_pbp_rolling[["batter_id", "gamepk"] + batter_pbp_rolling_cols],
    on=["batter_id", "gamepk"], how="left",
)

team_batter_rolling = rolling_stats.build_team_batter_strikeout_rolling_feats(
    pbp, batter_boxscore, window="season"
)
opp_team_rolling = team_batter_rolling.rename(columns={
    "team_roll_season_pa_strikeout_rate": "opp_team_roll_season_pa_strikeout_rate",
})[["batter_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")

pitching_team_rolling = team_batter_rolling.rename(columns={
    "batter_team_id": "pitcher_team_id",
    "team_roll_season_pa_strikeout_rate": "pitching_team_roll_season_pa_strikeout_rate",
})[["pitcher_team_id", "gamepk", "pitching_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")

# ------------------------- 2b. TRAILING-3-START PITCHER FORM (from v3) ------- #
pitcher_box_rolling3 = rolling_stats.build_pitcher_rolling_stats_all_roles(
    pitcher_boxscore, pbp, window=SHORT_PITCHER_WINDOW,
)
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

# ------------------------- 2c. OPPOSING-TEAM K-RATE VOLATILITY (from v3) ----- #
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

# ------------------------- 3. ROLLING SHRUNK TO LAST SEASON (pitcher WHIP) --- #
shrunk_whip = build_pitcher_shrunk_whip(
    pitcher_box_rolling, season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp),
    window="season", k=WHIP_SHRINKAGE_K,
)
shrunk_whip_cols = ["pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]
pa_outcome = pa_outcome.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + shrunk_whip_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

# ------------------------- 4. TOUGHEST-OUT + STAR-POWER (from v7) ----------- #
# PRIMARY: opp_team_toughest_out_shrunk_k_rate = MIN(shrunk batter K rate)
# across the opposing starting lineup.
batter_shrunk_k = build_batter_shrunk_k_rate(
    batter_pbp_rolling, batter_pbp_season_full, window="season", k=BATTER_SHRINKAGE_K,
)
toughest_out = build_opposing_lineup_extremum(
    batter_boxscore, pbp, batter_shrunk_k,
    metric_col="batter_shrunk_k_rate", out_col="opp_team_toughest_out_shrunk_k_rate",
    agg="min", shrunk_id_col="batter_id",
)
pa_outcome = pa_outcome.merge(toughest_out, on=["batter_team_id", "gamepk"], how="left")

# EXPLORATORY: opp_team_best_batter_shrunk_obp / _slg = MAX(shrunk OBP/SLG)
# across the same lineup, independently.
batter_box_rolling_obp_slg = rolling_stats.build_batter_rolling_stats(batter_boxscore, window="season")
batter_box_season_obp_slg = season_stats.build_batter_stats(batter_boxscore)
batter_shrunk_obp_slg = build_batter_shrunk_obp_slg(
    batter_box_rolling_obp_slg, batter_box_season_obp_slg, window="season", k=BATTER_SHRINKAGE_K,
)
best_batter_obp = build_opposing_lineup_extremum(
    batter_boxscore, pbp, batter_shrunk_obp_slg,
    metric_col="batter_shrunk_obp", out_col="opp_team_best_batter_shrunk_obp",
    agg="max", shrunk_id_col="personId",
)
pa_outcome = pa_outcome.merge(best_batter_obp, on=["batter_team_id", "gamepk"], how="left")

best_batter_slg = build_opposing_lineup_extremum(
    batter_boxscore, pbp, batter_shrunk_obp_slg,
    metric_col="batter_shrunk_slg", out_col="opp_team_best_batter_shrunk_slg",
    agg="max", shrunk_id_col="personId",
)
pa_outcome = pa_outcome.merge(best_batter_slg, on=["batter_team_id", "gamepk"], how="left")

# ------------------------- 5. LEAGUE-WIDE ROLLING CONTEXT (from v8) --------- #
# Does the model know what season/era it's in? Both tables pool across the
# ENTIRE league (every team, every batter, unfiltered by role/lineup) and are
# keyed by (game_season, game_date) ONLY -- see the summary block above and
# implementation_plan.md for the full design rationale.
league_pa_outcome = rolling_stats.build_league_pa_outcome_rolling_feats(pbp, window="season")
pa_outcome = pa_outcome.merge(league_pa_outcome, on=["game_season", "game_date"], how="left")

league_batter_rates = rolling_stats.build_league_batter_rolling_stats(batter_boxscore, window="season")
pa_outcome = pa_outcome.merge(league_batter_rates, on=["game_season", "game_date"], how="left")

# Sanity check on the new join shape -- first feature table in this repo
# keyed by date alone rather than gamepk/team_id/personId. Every PA sharing a
# game_date must see an IDENTICAL league value.
assert pa_outcome.groupby("game_date")["league_roll_season_pa_strikeout_rate"].nunique(dropna=False).max() == 1, (
    "league_roll_season_pa_strikeout_rate must be identical for every PA on the same game_date"
)

# ------------------------- 6. NEW in v9: BATTER'S OWN SLASH-LINE ------------ #
# batter_box_rolling_obp_slg (above) was already computed for v7's best-batter
# extremum feature but never merged onto pa_outcome for the BATTER'S OWN
# value -- needed here so find_player_vs_league_pairs has a batter-side
# slash-line column to match against v8's league_roll_season_{ba,slg,obp,
# iso,babip}. Keyed on personId/gamepk (box-score-derived) while pa_outcome
# uses batter_id/gamepk (pbp-derived) -- rename + explicit str-cast both
# merge keys before merging, same defensive dtype-cast pattern
# build_pitcher_rolling_stats_all_roles already uses for this exact kind of
# cross-source-table merge (a silent dtype mismatch produces an all-NaN
# merge, not an error).
batter_slash_line = batter_box_rolling_obp_slg.rename(columns={"personId": "batter_id"}).assign(
    batter_id=lambda x: x["batter_id"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)[["batter_id", "gamepk", "batter_roll_season_ba", "batter_roll_season_slg",
   "batter_roll_season_obp", "batter_roll_season_iso", "batter_roll_season_babip"]]
pa_outcome = pa_outcome.merge(batter_slash_line, on=["batter_id", "gamepk"], how="left")

# ------------------------- 7. NEW in v9: PLAYER-VS-LEAGUE INTERACTIONS ------ #
# Auto-discover every player-level rolling column with a same-window league
# counterpart now that both sides are fully assembled in pa_outcome (v1-v8's
# merges plus sections 6's slash-line merge and the widened pbp-rolling
# selections above) -- comprehensive by construction, not a hand-picked list.
vs_league_pairs = interaction_feats.find_player_vs_league_pairs(pa_outcome.columns.tolist())
assert len(vs_league_pairs) >= 10, (
    f"expected at least 10 player-vs-league pairs, found {len(vs_league_pairs)} — check upstream merges"
)

_pre_vs_league_cols = set(pa_outcome.columns)
pa_outcome = interaction_feats.build_player_vs_league_features(pa_outcome, vs_league_pairs)
NEW_V9_DERIVED_COLS = sorted(set(pa_outcome.columns) - _pre_vs_league_cols)

# Raw columns genuinely new to FEATURE_COLS this version (walk/hbp/single/
# xbh/hr rate for batter+pitcher, batter slash-line) — excludes columns
# already wired in since v2/v7 (e.g. opp_team/pitching_team K rate, which
# also happen to auto-pair above but aren't new to the model this version).
NEW_V9_RAW_COLS = sorted(set(batter_pbp_rolling_cols + pbp_rolling_cols) & {
    "batter_roll_season_pa_walk_rate", "batter_roll_season_pa_hbp_rate",
    "batter_roll_season_pa_single_rate", "batter_roll_season_pa_xbh_rate",
    "batter_roll_season_pa_hr_rate",
    "pitcher_roll_season_pa_walk_rate", "pitcher_roll_season_pa_hbp_rate",
    "pitcher_roll_season_pa_single_rate", "pitcher_roll_season_pa_xbh_rate",
    "pitcher_roll_season_pa_hr_rate",
} | {
    "batter_roll_season_ba", "batter_roll_season_slg", "batter_roll_season_obp",
    "batter_roll_season_iso", "batter_roll_season_babip",
})


# ------------------------- 8. NEW in v10: PARK-SPECIFIC STRIKEOUT TENDENCY -- #
park_strikeout_factor = park_factors.build_park_strikeout_factor(schedule, pitcher_boxscore)
pa_outcome = pa_outcome.merge(park_strikeout_factor, on=["venue_id", "game_season"], how="left")
assert "park_last_season_strikeout_factor" in pa_outcome.columns, (
    "build_park_strikeout_factor merge failed — check venue_id/game_season key alignment"
)

NEW_V10_RAW_COLS = [
    "pitcher_roll_season_pa_two_strike_reach_rate", "pitcher_roll_season_pa_put_away_rate",
    "batter_roll_season_pa_two_strike_reach_rate", "batter_roll_season_pa_put_away_rate",
    "park_last_season_strikeout_factor",
]

# ------------------------- 9. NEW in v11: CSW% ------------------------------ #
# command_csw_rate = command_called_strike_rate + command_swinging_strike_rate,
# already merged above at both grains (section 1's season merge, section 2's
# rolling merge) -- both assertions already confirmed the columns landed.
NEW_V11_RAW_COLS = [
    "pitcher_last_season_command_csw_rate",
    "pitcher_roll_season_command_csw_rate",
]


# ── 4. Season-based train / val / test split ─────────────────────────────────
FEATURE_COLS = [
    "expected_pitcher_role",
    "pitcher_throw_hand",
    "batter_bat_side",
    # 1. season (last season)
    "pitcher_last_season_pa_strikeout_rate",
    "pitcher_last_season_whip",
    "batter_last_season_pa_strikeout_rate",
    # 2. rolling raw (this season) — pitcher side, from v1
    "pitcher_roll_season_ip",
    "pitcher_roll_season_whip",
    "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate",
    "pitcher_roll_season_hr_rate",
    "pitcher_roll_season_games_n",
    "pitcher_roll_season_avg_ip_per_game",
    "pitcher_roll_season_pa_total",
    "pitcher_roll_season_pa_strikeout_rate",
    # 2b. rolling raw (this season) — batter + team side, from v2
    "batter_roll_season_pa_strikeout_rate",
    "opp_team_roll_season_pa_strikeout_rate",
    "pitching_team_roll_season_pa_strikeout_rate",
    # 2c. trailing-3-start pitcher form, from v3
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_whip",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_k_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_bb_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_hr_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_avg_ip_per_game",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_total",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_strikeout_rate",
    # 2d. opposing-team K-rate volatility, from v3 — season + trailing-5
    "opp_team_roll_season_pa_strikeout_rate_mean",
    "opp_team_roll_season_pa_strikeout_rate_std",
    "opp_team_roll_season_pa_strikeout_rate_max",
    f"opp_{short_team_prefix}_mean",
    f"opp_{short_team_prefix}_std",
    f"opp_{short_team_prefix}_max",
    # 2e. whiff rate + pitch efficiency, from v3 — season + trailing-3
    "pitcher_roll_season_command_swinging_strike_rate",
    "pitcher_roll_season_pa_pitch_count_mean",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_command_swinging_strike_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean",
    # 3. rolling shrunk to last season (pitcher WHIP)
    "pitcher_shrunk_whip",
    "pitcher_shrunk_whip_weight",
    # 4. game context, from v2
    "weather_condition",
    "weather_temp",
    "expected_times_through_order",
    # 5. toughest-out (primary) + best-batter star power (exploratory), from v7
    "opp_team_toughest_out_shrunk_k_rate",
    "opp_team_best_batter_shrunk_obp",
    "opp_team_best_batter_shrunk_slg",
    # 6. NEW in v8 — league-wide rolling context (PA-outcome side)
    "league_roll_season_pa_strikeout_rate",
    "league_roll_season_pa_walk_rate",
    "league_roll_season_pa_hbp_rate",
    "league_roll_season_pa_single_rate",
    "league_roll_season_pa_xbh_rate",
    "league_roll_season_pa_hr_rate",
    # 6b. NEW in v8 — league-wide rolling context (slash-line side)
    "league_roll_season_ba",
    "league_roll_season_slg",
    "league_roll_season_obp",
    "league_roll_season_iso",
    "league_roll_season_babip",
] + NEW_V9_RAW_COLS + NEW_V9_DERIVED_COLS + NEW_V10_RAW_COLS + NEW_V11_RAW_COLS
FEATURE_COLS = [c for c in FEATURE_COLS if c in pa_outcome.columns]
NEW_V8_COLS = [
    "league_roll_season_pa_strikeout_rate", "league_roll_season_pa_walk_rate",
    "league_roll_season_pa_hbp_rate", "league_roll_season_pa_single_rate",
    "league_roll_season_pa_xbh_rate", "league_roll_season_pa_hr_rate",
    "league_roll_season_ba", "league_roll_season_slg", "league_roll_season_obp",
    "league_roll_season_iso", "league_roll_season_babip",
]
for c in NEW_V8_COLS:
    assert c in FEATURE_COLS, f"{c} missing from pa_outcome — merge failed upstream"
for c in NEW_V9_RAW_COLS + NEW_V9_DERIVED_COLS:
    assert c in FEATURE_COLS, f"{c} missing from pa_outcome — merge failed upstream"
for c in NEW_V10_RAW_COLS:
    assert c in FEATURE_COLS, f"{c} missing from pa_outcome — merge failed upstream"
for c in NEW_V11_RAW_COLS:
    assert c in FEATURE_COLS, f"{c} missing from pa_outcome — merge failed upstream"

NAIVE_ROLE_COL = "expected_pitcher_role"

GAME_GRAIN_KEY_COLS = ["gamepk", "batter_id"]  # grouping keys for the game-grain check below, not features
model_df = pa_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"] + GAME_GRAIN_KEY_COLS].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"Feature count: {len(FEATURE_COLS)} (v10's 114 + {len(NEW_V11_RAW_COLS)} new v11 CSW% columns)")
print(f"Strikeout rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")
print("New v11 columns (CSW% — called-strike rate + swinging-strike rate):")
for c in NEW_V11_RAW_COLS:
    print(f"  {c}: {model_df[c].notna().mean():.1%} non-null, mean {model_df[c].mean():.4f}")

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


Xtr, Xval = encode(X_train, X_val, cat_cols, num_cols)


# ── 5. Train models, evaluate on val ─────────────────────────────────────────
# Carries forward v6's exact winning hyperparameters rather than re-grid-searching
# — same "carry tuned hyperparameters forward unchanged" convention
# batters_faced_predictor's v2-v7 already established.
results = {}


def _eval(name, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    results[name] = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "prob": y_prob,
        "pred": y_pred,
    }


print("\nEvaluating naive (most frequent class)...")
naive_global = DummyClassifier(strategy="most_frequent")
naive_global.fit(Xtr, y_train)
_eval("Naive (most frequent)", y_val, naive_global.predict_proba(Xval)[:, 1])

print("Evaluating naive (per expected-pitcher-role K rate)...")
role_rate = train_df.groupby(NAIVE_ROLE_COL)[TARGET].mean()
naive_role_pred = X_val[NAIVE_ROLE_COL].map(role_rate).fillna(y_train.mean())
_eval("Naive (per-role K rate)", y_val, naive_role_pred.to_numpy())

# ------------------------- LOGISTIC REGRESSION: v6's fixed winning config --- #
print("\nFitting logistic regression (v6's winning config: C=0.1, L1, no class weighting)...")
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xval_sc = scaler.transform(Xval)

lr_model = LogisticRegression(C=0.1, penalty="l1", class_weight=None, solver="liblinear", max_iter=2000)
lr_model.fit(Xtr_sc, y_train)
_eval("Logistic regression (v6 config)", y_val, lr_model.predict_proba(Xval_sc)[:, 1])

# ------------------------- XGBOOST: v6's fixed winning config, early-stopped - #
print("\nFitting XGBoost (v6's winning config: max_depth=2, learning_rate=0.03, early-stopped)...")
import xgboost as xgb

EARLY_STOP_SEASON = max(FIT_SEASONS)
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]
print(f"  Core fit seasons: {CORE_FIT_SEASONS}  |  early-stop season: {EARLY_STOP_SEASON}  |  final selection: {VAL_SEASON} (untouched)")

core_train_df = train_df[train_df["game_season"].isin(CORE_FIT_SEASONS)]
early_stop_df = train_df[train_df["game_season"] == EARLY_STOP_SEASON]


def fit_transform_xgb(fit_df, *apply_dfs):
    fit_df = fit_df[FEATURE_COLS].copy()
    apply_dfs = [df[FEATURE_COLS].copy() for df in apply_dfs]
    if num_cols:
        fit_df[num_cols] = fit_df[num_cols].apply(pd.to_numeric, errors="coerce")
        for df in apply_dfs:
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    if cat_cols:
        fit_df[cat_cols] = fit_df[cat_cols].astype(object).fillna(np.nan)
        for df in apply_dfs:
            df[cat_cols] = df[cat_cols].astype(object).fillna(np.nan)

    num_imp_x = SimpleImputer(strategy="median")
    Xfit_num = num_imp_x.fit_transform(fit_df[num_cols]) if num_cols else np.empty((len(fit_df), 0))
    Xapply_num = [num_imp_x.transform(df[num_cols]) if num_cols else np.empty((len(df), 0)) for df in apply_dfs]

    if cat_cols:
        cat_imp_x = SimpleImputer(strategy="most_frequent")
        Xfit_cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xfit_cat_imp = cat_imp_x.fit_transform(fit_df[cat_cols])
        Xfit_cat_enc = Xfit_cat.fit_transform(Xfit_cat_imp)
        Xapply_cat = [Xfit_cat.transform(cat_imp_x.transform(df[cat_cols])) for df in apply_dfs]
    else:
        Xfit_cat_enc = np.empty((len(fit_df), 0))
        Xapply_cat = [np.empty((len(df), 0)) for df in apply_dfs]

    Xfit = np.hstack([Xfit_num, Xfit_cat_enc])
    Xapply = [np.hstack([n, c]) for n, c in zip(Xapply_num, Xapply_cat)]
    return Xfit, Xapply


Xcore, (Xearly, Xval_xgb) = fit_transform_xgb(core_train_df, early_stop_df, val_df)
y_core, y_early = core_train_df[TARGET], early_stop_df[TARGET]

xgb_model = xgb.XGBClassifier(
    n_estimators=2000, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, verbosity=0, eval_metric="logloss", early_stopping_rounds=30,
)
xgb_model.fit(Xcore, y_core, eval_set=[(Xearly, y_early)], verbose=False)
print(f"  XGBoost best_iteration: {xgb_model.best_iteration}")
_eval("XGBoost (v6 config)", y_val, xgb_model.predict_proba(Xval_xgb)[:, 1])


# ── 6. Print results ──────────────────────────────────────────────────────────
naive_pr_auc = results["Naive (most frequent)"]["pr_auc"]
role_pr_auc = results["Naive (per-role K rate)"]["pr_auc"]
best_naive_name, best_naive_pr_auc = max(
    (("Naive (most frequent)", naive_pr_auc), ("Naive (per-role K rate)", role_pr_auc)),
    key=lambda t: t[1],
)

# v6's tuned XGBoost remains the standing best model after v7-v10's flat
# results — hardcoded here for a direct comparison printout, same convention
# as every prior version's script.
STANDING_BEST_XGB_PR_AUC = 0.2838
STANDING_BEST_GAME_RELIABILITY = 0.0001
STANDING_BEST_GAME_RESOLUTION = 0.0137

print(f"\n{'='*72}")
print(f"EXPERIMENT RESULTS — {MODEL_NAME} v11 (CSW%)")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} PAs)  |  Test season {TEST_SEASON} locked")
print("Primary: PR-AUC (higher=better)  |  Secondary: ROC-AUC (higher=better)")
print("=" * 72)
print(f"{'Model':<28} {'PR-AUC':>8} {'vs best naive':>14}  {'ROC-AUC':>8}")
print("-" * 72)
for name, res in results.items():
    delta = f"{res['pr_auc'] - best_naive_pr_auc:+.4f}" if name not in ("Naive (most frequent)", "Naive (per-role K rate)") else "—"
    print(f"{name:<28} {res['pr_auc']:>8.4f} {delta:>14}  {res['roc_auc']:>8.4f}")
print("-" * 72)
print(f"{'(v6 XGBoost tuned, standing best, for reference)':<28} {STANDING_BEST_XGB_PR_AUC:>8.4f}")
print("=" * 72)

print("\nInterpretation:")
candidates = {n: r for n, r in results.items() if n not in ("Naive (most frequent)", "Naive (per-role K rate)")}
beats_floor = {n: r for n, r in candidates.items() if r["pr_auc"] > best_naive_pr_auc}
best_name, best = max(candidates.items(), key=lambda t: t[1]["pr_auc"])

if not beats_floor:
    print(f"  No model beats the best naive floor ({best_naive_name}, PR-AUC {best_naive_pr_auc:.4f}).")
else:
    print(f"  {best_name} beats the best naive floor ({best_naive_name}, PR-AUC {best_naive_pr_auc:.4f}) —")
    print(f"  {best_name} PR-AUC {best['pr_auc']:.4f}.")

vs_standing = best["pr_auc"] - STANDING_BEST_XGB_PR_AUC
if vs_standing > 0.005:
    print(f"  vs v6's tuned XGBoost (PR-AUC {STANDING_BEST_XGB_PR_AUC:.4f}): {vs_standing:+.4f} — real gain from CSW%.")
elif vs_standing < -0.005:
    print(f"  vs v6's tuned XGBoost (PR-AUC {STANDING_BEST_XGB_PR_AUC:.4f}): {vs_standing:+.4f} — worse, check for a wiring bug.")
else:
    print(f"  vs v6's tuned XGBoost (PR-AUC {STANDING_BEST_XGB_PR_AUC:.4f}): {vs_standing:+.4f} — flat, same PA-grain-metric pattern as most single-pass feature additions in this project.")
print("  Per this project's own metrics convention (PR-AUC is a PA-grain triage filter,")
print("  not a decision metric): this result alone does not confirm or rule out a real")
print("  improvement — that verdict comes from the game-grain check below.")
print("=" * 72)


# ── 7. Game-grain aggregation check ───────────────────────────────────────────
print("\n" + "=" * 72)
print("GAME-GRAIN CHECK — batter-game \"1+ strikeout\" (same approach as v1-v10)")
print("=" * 72)

best_model_name, best_model = max(candidates.items(), key=lambda t: t[1]["pr_auc"])
print(f"Rolling up {best_model_name}'s val predictions...")

pa_metrics, game_metrics, game_results = run_pa_vs_game_grain_check(
    val_df, y_val, best_model["prob"], group_cols=("batter_id", "gamepk"),
)

naive_pa_results = val_df[["batter_id", "gamepk"]].copy()
naive_pa_results["is_hit"] = np.asarray(y_val)  # cosmetic column name, see comment above
naive_pa_results["pred_prob"] = naive_role_pred.to_numpy()
naive_game_results = aggregate_pa_predictions_to_game(naive_pa_results, group_cols=("batter_id", "gamepk"))
naive_game_metrics = evaluate_hit_predictor(
    y_true=naive_game_results["game_is_hit"], y_prob=naive_game_results["game_pred_prob"],
    n_bins=10, min_n=200, base_rate=naive_game_results["game_is_hit"].mean(),
)

print(f"\n{len(val_df):,} PAs -> {len(game_results):,} batter-game rows "
      f"(mean {game_results['n_pa'].mean():.2f} PA/game)")
print(f"Batter-game strikeout rate (1+ K): {game_results['game_is_hit'].mean():.3f}")
print(f"\n{'Metric':<14} {'Naive (per-role)':>18} {best_model_name:>22}")
for key in ("reliability", "resolution", "roc_auc", "brier", "log_loss", "ece"):
    print(f"{key:<14} {naive_game_metrics[key]:>18.4f} {game_metrics[key]:>22.4f}")

game_verdict = summarize_verdict(naive_game_metrics, game_metrics)
print(f"\nVerdict vs naive at game grain: {game_verdict['verdict']}")
print(f"  reliability delta: {game_verdict['reliability_delta']:+.4f} (lower=better, negative=more honest)")
print(f"  resolution delta:  {game_verdict['resolution_delta']:+.4f} (higher=better, positive=more real spread)")
print(f"\n  (v6's game-grain result, standing best, for reference: reliability {STANDING_BEST_GAME_RELIABILITY:.4f}, "
      f"resolution {STANDING_BEST_GAME_RESOLUTION:.4f}, verdict real_improvement)")
print("=" * 72)

PLOT_DIR = STAGE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
plot_calibration_curve(
    game_results["game_is_hit"],
    {
        "Naive (per-role K rate)": {"proba": naive_game_results["game_pred_prob"]},
        best_model_name: {"proba": game_results["game_pred_prob"]},
    },
    n_bins=10, min_n=50,
    save_path=PLOT_DIR / "game_grain_calibration.png",
)
print(f"Saved {PLOT_DIR / 'game_grain_calibration.png'}")


# ── 8. Plots ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
train_df[TARGET].value_counts().sort_index().plot(kind="bar", color="steelblue", ax=ax)
ax.set_title(f"is_strikeout distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_xlabel("is_strikeout")
ax.set_ylabel("PAs")
plt.tight_layout()
plt.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'target_distribution.png'}")

fig, ax = plt.subplots(figsize=(5, 5))
cm = confusion_matrix(y_val, results["XGBoost (v6 config)"]["pred"])
ConfusionMatrixDisplay(cm, display_labels=["No K", "K"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"XGBoost confusion matrix — val ({VAL_SEASON})")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'confusion_matrix.png'}")

feature_names = num_cols + cat_cols
importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, max(4, len(importances) * 0.3)))
colors = [
    "purple" if name in NEW_V11_RAW_COLS
    else "forestgreen" if name in NEW_V10_RAW_COLS
    else "darkorange" if name in NEW_V9_RAW_COLS
    else "crimson" if name in NEW_V9_DERIVED_COLS
    else "steelblue"
    for name in importances.index
]
ax.barh(importances.index, importances.values, color=colors)
ax.set_title("Feature importance — XGBoost (v11 CSW% in purple, v10 in green, v9 raw in orange, v9 derived in red)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'feature_importance.png'}")

rank_order = importances.sort_values(ascending=False)
print("\nv11 new CSW% column ranks (of {} features):".format(len(importances)))
for c in NEW_V11_RAW_COLS:
    rank = list(rank_order.index).index(c) + 1
    print(f"  {c}: rank #{rank}, importance {rank_order[c]:.4f}")


# ── 9. MLflow logging ─────────────────────────────────────────────────────────
for name, res in results.items():
    metrics = {"pr_auc": res["pr_auc"], "roc_auc": res["roc_auc"]}
    artifact_paths = [
        PLOT_DIR / "target_distribution.png",
        PLOT_DIR / "confusion_matrix.png",
        PLOT_DIR / "feature_importance.png",
    ]
    params = {
        "model_type": name,
        "n_features": len(FEATURE_COLS),
        "whip_shrinkage_k": WHIP_SHRINKAGE_K,
        "batter_shrinkage_k": BATTER_SHRINKAGE_K,
        "short_pitcher_window": SHORT_PITCHER_WINDOW,
        "short_team_window": SHORT_TEAM_WINDOW,
        "fit_seasons": str(FIT_SEASONS),
    }
    if name == best_model_name:
        metrics.update({
            "game_reliability": game_metrics["reliability"],
            "game_resolution": game_metrics["resolution"],
            "game_roc_auc": game_metrics["roc_auc"],
        })
        params["game_verdict"] = game_verdict["verdict"]
        artifact_paths.append(PLOT_DIR / "game_grain_calibration.png")

    log_evaluation_to_mlflow(
        metrics=metrics,
        params=params,
        tags={
            "stage": "v11_csw_pct",
            "target": TARGET,
            "val_season": str(VAL_SEASON),
            "git_sha": get_git_sha() or "unknown",
        },
        artifact_paths=artifact_paths,
    )
print("\nLogged all runs to MLflow (experiment: k_predictor).")
