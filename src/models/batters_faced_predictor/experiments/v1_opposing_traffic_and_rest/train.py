"""
batters_faced_predictor experiment v1: opposing-lineup traffic (on-base/walk
rate), team rest days, home/away, pitch-efficiency proxy.
Run from src/models/batters_faced_predictor/ with:
  python experiments/v1_opposing_traffic_and_rest/train.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    v1 SUMMARY                                #
# ---------------------------------------------------------------------------- #
# Baseline (baseline/model/run.py) fed a regressor literally the same inputs
# the expected_batters_faced shrinkage cascade already has access to — no
# model beat the cascade's own MAE 2.8609 that way. This experiment tests the
# actual hypothesis: new information the cascade doesn't see.
#
#   1. OPPOSING-LINEUP ON-BASE/WALK RATE (NEW):
#      rolling_stats.build_team_batter_onbase_rolling_feats(pbp, batter_boxscore,
#      window='season') — new shared function (TDD'd,
#      tests/hit_predictor/test_rolling_stats.py), mirrors
#      build_team_batter_strikeout_rolling_feats (k_predictor v2) exactly.
#      Mechanism: more traffic (walks + hits) per inning means more total
#      batters faced for the same number of outs recorded, independent of
#      strikeout rate. Joined on the OPPOSING team for this start (the
#      batting side, not the pitcher's own team) — derived from schedule's
#      home_id/away_id vs. the pitcher's own team_id.
#   2. TEAM REST DAYS: pitcher_team_days_since_last_game, via the pitcher's
#      OWN team (game_context.build_team_rest_days, already exists — same
#      function short_outing_predictor's v1 already uses for the same
#      mechanism, reused here unmodified).
#   3. HOME/AWAY: is_home, derived from schedule (pitcher_team_id == home_id).
#      Mechanism: no DH in NL road games pre-2022 universal-DH rule affects
#      pitcher removal-for-pinch-hitter timing — a plausible, cheap-to-test
#      lever on total batters faced.
#   4. PITCH-EFFICIENCY PROXY: pitcher_roll_season_pitch_count_avg, via
#      rolling_stats.build_pbp_pitcher_rolling_feats(sp_pbp, window='season',
#      pitcher_role='sp') — season-to-date average pitches/start, already
#      computed, just not previously wired into this model. A trailing-N
#      TREND (delta vs. season average) is a further refinement not
#      attempted here — flagged as a "next" if this proxy alone doesn't move
#      the floor.
#
# Evaluated the same way as the baseline (MAE primary, RMSE/Bias/Pearson r
# secondary, same expected_batters_faced_weight-quartile stratification) for
# a direct, apples-to-apples comparison against both the cascade and the
# baseline's own LR/XGBoost numbers.
#
# XGBoost is trained BOTH default and tuned here, same as the (re-run)
# baseline — experiments/xgb_vs_cascade_diagnostic/run.py found default
# XGBoost overfits badly at this data size (worked examples showed it
# predicting implausibly low batters-faced counts for established starters)
# and a tuned XGBoost (same hyperparameters used here) beat the cascade on
# the baseline's smaller feature set. See ROADMAP.md's 2026-08-26 follow-up
# entry.
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

# Internal early-stopping validation set for the tuned XGBoost — same
# convention as baseline/model/run.py and experiments/xgb_vs_cascade_diagnostic/run.py.
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

# ------------------------- 1. OPPOSING-LINEUP ON-BASE/WALK RATE (NEW) -------- #
# Opposing team for this start = whichever of home_id/away_id is NOT the
# pitcher's own team — the batting side he actually faces.
home_away = schedule[["gamepk", "home_id", "away_id"]].drop_duplicates("gamepk").assign(
    gamepk=lambda x: x["gamepk"].astype(str),
    home_id=lambda x: x["home_id"].astype(str),
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

# ------------------------- 2. TEAM REST DAYS (pitcher's own team) ------------ #
team_rest_days = game_context.build_team_rest_days(schedule)[["team_id", "gamepk", "team_days_since_last_game"]]
team_rest_days = team_rest_days.assign(
    team_id=lambda x: x["team_id"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
).rename(columns={"team_id": "pitcher_team_id", "team_days_since_last_game": "pitcher_team_days_since_last_game"})
start_outcome = start_outcome.merge(team_rest_days, on=["pitcher_team_id", "gamepk"], how="left")

# ------------------------- 3. PITCH-EFFICIENCY PROXY (NEW) ------------------- #
sp_pbp = pbp[pbp["pitcher_role"] == "sp"]
pitcher_pitch_efficiency = rolling_stats.build_pbp_pitcher_rolling_feats(
    sp_pbp, window="season", pitcher_role="sp", entity_col="pitcher_id",
)[["pitcher_id", "gamepk", "pitcher_roll_season_pitch_count_avg"]].rename(columns={"pitcher_id": "personId"})
pitcher_pitch_efficiency = pitcher_pitch_efficiency.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)
start_outcome = start_outcome.merge(pitcher_pitch_efficiency, on=["personId", "gamepk"], how="left")

# ------------------------- 4. HANDEDNESS (baseline feature) ------------------ #
pitcher_hand = (
    pbp[["pitcher_id", "pitcher_throw_hand"]]
    .drop_duplicates(subset=["pitcher_id"])
    .rename(columns={"pitcher_id": "personId"})
)
start_outcome = start_outcome.merge(pitcher_hand, on="personId", how="left")


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
    # v1 new features
    "team_roll_season_walk_rate",
    "team_roll_season_on_base_rate",
    "pitcher_team_days_since_last_game",
    "is_home",
    "pitcher_roll_season_pitch_count_avg",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in start_outcome.columns]
CASCADE_COL = "expected_batters_faced"
WEIGHT_COL = "expected_batters_faced_weight"

model_df = start_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].dropna(subset=[TARGET]).copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

# Core/early split for the tuned XGBoost's early stopping only — LR/default
# XGBoost/cascade still use the full FIT_SEASONS train_df above, unchanged.
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
    off ONE shared fit on core_df — avoids fitting the imputers/encoder
    twice with what would otherwise be two separate encode() calls."""
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
_eval("Linear regression (v1)", y_val.to_numpy(), lr.predict(Xval_sc))

print("Training XGBoost (v1, default)...")
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(Xtr, y_train)
_eval("XGBoost (v1, default)", y_val.to_numpy(), xgb_model.predict(Xval))

print("Training XGBoost (v1, tuned)...")
# Hyperparameters and early-stopping setup carried over unchanged from
# experiments/xgb_vs_cascade_diagnostic/run.py / baseline/model/run.py.
Xcore, [Xearly_enc, Xval_core] = encode_multi(
    core_df[FEATURE_COLS], [early_df[FEATURE_COLS], val_df[FEATURE_COLS]], cat_cols, num_cols
)
y_core, y_early = core_df[TARGET], early_df[TARGET]

xgb_tuned = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.02, max_depth=3, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5,
    random_state=42, verbosity=0, early_stopping_rounds=50, eval_metric="mae",
)
xgb_tuned.fit(Xcore, y_core, eval_set=[(Xearly_enc, y_early)], verbose=False)
print(f"  Best iteration (early stopping on {EARLY_STOP_SEASON}): {xgb_tuned.best_iteration}")
_eval("XGBoost (v1, tuned)", y_val.to_numpy(), xgb_tuned.predict(Xval_core))


# ── 6. Print results ──────────────────────────────────────────────────────────
CASCADE_NAME = "Cascade (expected_batters_faced)"
cascade_mae = results[CASCADE_NAME]["mae"]

# baseline/model/run.py's own numbers (re-run with tuned XGBoost, 2026-08-26),
# hardcoded here for a direct comparison printout — not re-derived, since
# baseline used a smaller, frozen feature set.
BASELINE_LR_MAE = 2.9215
BASELINE_XGB_DEFAULT_MAE = 3.4655
BASELINE_XGB_TUNED_MAE = 2.7411

print(f"\n{'=' * 72}")
print(f"EXPERIMENT RESULTS — {MODEL_NAME} v1 (opposing traffic + rest + home/away + pitch efficiency)")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} starts)  |  Test season {TEST_SEASON} locked")
print("Primary: MAE (lower=better)  |  Secondary: RMSE, Bias, Pearson r")
print("=" * 72)
print(f"{'Model':<32} {'MAE':>8} {'vs cascade':>12}  {'RMSE':>8} {'Bias':>8} {'Pearson r':>10}")
print("-" * 72)
for name, res in results.items():
    delta = f"{res['mae'] - cascade_mae:+.4f}" if name != CASCADE_NAME else "—"
    print(f"{name:<32} {res['mae']:>8.4f} {delta:>12}  {res['rmse']:>8.4f} {res['bias']:>+8.4f} {res['pearson_r']:>10.4f}")
print("-" * 72)
print(f"{'(baseline LR, for reference)':<32} {BASELINE_LR_MAE:>8.4f}")
print(f"{'(baseline XGB default, for reference)':<32} {BASELINE_XGB_DEFAULT_MAE:>8.4f}")
print(f"{'(baseline XGB tuned, for reference)':<32} {BASELINE_XGB_TUNED_MAE:>8.4f}")
print("=" * 72)

print("\nInterpretation:")
candidates = {n: r for n, r in results.items() if n != CASCADE_NAME}
beats_floor = {n: r for n, r in candidates.items() if r["mae"] < cascade_mae}
best_model_name, best_model = min(candidates.items(), key=lambda t: t[1]["mae"])

if not beats_floor:
    print(f"  No model beats the cascade floor ({CASCADE_NAME}, MAE {cascade_mae:.4f}).")
    print("  Opposing-lineup traffic, rest days, home/away, and pitch efficiency do not")
    print("  demonstrate real signal beyond the cascade's own point estimate this pass.")
else:
    print(f"  {best_model_name} beats the cascade floor ({CASCADE_NAME}, MAE {cascade_mae:.4f}) —")
    print(f"  {best_model_name} MAE {best_model['mae']:.4f}.")

BASELINE_BEST_MAE = min(BASELINE_LR_MAE, BASELINE_XGB_DEFAULT_MAE, BASELINE_XGB_TUNED_MAE)
vs_baseline = best_model["mae"] - BASELINE_BEST_MAE
if vs_baseline < -0.02:
    print(f"  vs baseline's best model (MAE {BASELINE_BEST_MAE:.4f}): {vs_baseline:+.4f} —")
    print("  the new features add real signal beyond the baseline (cascade-inputs-only) feature set.")
elif vs_baseline > 0.02:
    print(f"  vs baseline's best model (MAE {BASELINE_BEST_MAE:.4f}): {vs_baseline:+.4f} —")
    print("  worse than the smaller baseline feature set — check for a wiring bug before concluding")
    print("  the new features don't help.")
else:
    print(f"  vs baseline's best model (MAE {BASELINE_BEST_MAE:.4f}): {vs_baseline:+.4f} — flat,")
    print("  no demonstrated improvement from the new features this pass.")
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
# Same isolation method k_predictor's count_distribution_check/run.py uses
# (BENCHMARKS.md's "Isolating which input drives error in a compound
# prediction" section) — substitute the winning model's prediction for
# expected_batters_faced and see whether the closest-quartile floor MAE
# itself moves (it may not — a real, worthwhile win can still be a shrinking
# gap between quartiles rather than a lower floor, per the user's own caveat).
print(f"\n{'=' * 72}\nbf_gap-QUARTILE FLOOR RE-CHECK — {best_model_name} vs the original cascade\n"
      f"(same isolation method as k_predictor's count_distribution_check/run.py;\n"
      f"'bf_gap' = |realized_batters_faced - point estimate|)\n{'=' * 72}")

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
ax.set_title("Feature importance — XGBoost (v1, tuned)")
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

    log_evaluation_to_mlflow(
        metrics=metrics,
        params=params,
        tags={
            "stage": "v1_opposing_traffic_and_rest",
            "target": TARGET,
            "val_season": str(VAL_SEASON),
            "git_sha": get_git_sha() or "unknown",
        },
        artifact_paths=artifact_paths,
    )
print(f"\nLogged all runs to MLflow (experiment: {MODEL_NAME}).")


# ── 9. Write v1_results.md ────────────────────────────────────────────────────
md_lines = [
    f"# v1 Results — {MODEL_NAME} (opposing traffic + rest + home/away + pitch efficiency)",
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
    f"| (baseline LR, for reference) | {BASELINE_LR_MAE:.4f} | | | | |",
    f"| (baseline XGBoost default, for reference) | {BASELINE_XGB_DEFAULT_MAE:.4f} | | | | |",
    f"| (baseline XGBoost tuned, for reference) | {BASELINE_XGB_TUNED_MAE:.4f} | | | | |",
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
    "## Setup",
    "",
    f"- Features: {FEATURE_COLS}",
    "- New this pass: team_roll_season_walk_rate/_on_base_rate (opposing lineup,",
    "  rolling_stats.build_team_batter_onbase_rolling_feats, new shared TDD'd",
    "  function), pitcher_team_days_since_last_game (game_context.build_team_rest_days,",
    "  reused unmodified), is_home (derived from schedule), ",
    "  pitcher_roll_season_pitch_count_avg (pitch-efficiency proxy,",
    "  rolling_stats.build_pbp_pitcher_rolling_feats, reused unmodified).",
    "- XGBoost (v1, tuned) hyperparameters and its held-out-season",
    "  early-stopping setup are carried over unchanged from",
    "  experiments/xgb_vs_cascade_diagnostic/run.py and the re-run baseline —",
    "  see ROADMAP.md's 2026-08-26 follow-up entry for the overfitting",
    "  diagnosis that motivated tuning XGBoost in the first place.",
]
results_path = BASE_DIR / "v1_results.md"
results_path.write_text("\n".join(md_lines) + "\n")
print(f"\nSaved {results_path}")
