"""
Baseline batters_faced_predictor — expected starting-pitcher batters-faced
regression. Run from src/models/batters_faced_predictor/ with:
python baseline/model/run.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Split strategy (same convention as hit_predictor/k_predictor/n_pa_predictor/
bb_predictor/short_outing_predictor):
  train = all train_seasons except val_season and test_season
  val   = val_season  (iterate against this during development)
  test  = test_season (locked away — final eval only in a future train.py)

Grain: one row per (personId, gamepk) STARTING-PITCHER START — same grain as
short_outing_predictor, not per PA or per batter-game.

This is the FIRST REGRESSION model in this repo's model layer (every sibling
so far predicts a binary outcome). The floor to beat is NOT a global-mean
naive — it's an already-shipped shrinkage cascade,
hit_predictor.processing.features.game_context::build_expected_batters_faced
(pitcher -> team -> league last-season baseline, blended toward this
season's own emerging rolling average). Its own val-2024 accuracy (measured
in k_predictor's count_distribution_check diagnostic, recorded in
ROADMAP.md): MAE 2.861, RMSE 3.951, Bias +0.073, Pearson r 0.495.

Models compared, all fed the SAME inputs the cascade formula itself already
has access to (nothing new — new feature engineering is v1's job, not this
baseline's):
  1. The cascade's own expected_batters_faced prediction (a benchmark row,
     not a trained model).
  2. Linear Regression.
  3. XGBoost Regressor, default hyperparameters.
  4. XGBoost Regressor, TUNED hyperparameters (shallow trees, low learning
     rate, early stopping against a held-out fit season, regularization).
The question this baseline answers: does letting a regressor combine the
cascade's own inputs non-linearly already beat the hand-built shrinkage
formula, before any new feature engineering is tried?

Default XGBoost overfits badly at this data size (~18-19k training rows) —
diagnosed in experiments/xgb_vs_cascade_diagnostic/run.py (see ROADMAP.md's
2026-08-26 follow-up entry): train MAE far below val MAE, and the biggest
cascade-vs-XGBoost disagreements were all cases of XGBoost predicting an
implausibly low batters-faced count for an established starter. A properly
tuned XGBoost (same features, same data) beat the cascade in that
diagnostic — hyperparameters below are carried over from there unchanged.
"""
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

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context

import models.batters_faced_predictor.processing.pipeline as pipeline

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PA_SHRINKAGE_K = 5.0  # same default as game_context.build_expected_batters_faced / k_predictor's reuse of it


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
# Same reason as every sibling model: build_pitcher_start_pa_stats needs a
# prior season's pbp for the shift, which isn't loaded for 2016.
FIT_SEASONS.remove(2017)

# Internal early-stopping validation set for the tuned XGBoost, carried over
# from experiments/xgb_vs_cascade_diagnostic/run.py: hold out the most
# recent fit season so early stopping has a real held-out signal. NOT the
# same as VAL_SEASON (2024) — that stays untouched by any fitting decision.
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

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)


# ── 3. Build start-grain DataFrame ────────────────────────────────────────────
print("\nBuilding start-grain DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
pbp = hp_pipeline.build_pbp_features(pbp, schedule, player_info)
# build_pitcher_start_pa_this_season needs game_datetime to order same-date
# doubleheaders correctly (build_pbp_features only ever adds game_date) —
# same explicit merge k_predictor's count_distribution_check/run.py needs
# for the same reason (see that file's own comment).
pbp = pbp.merge(schedule[["gamepk", "game_datetime"]], on="gamepk", how="left")

start_outcome = pipeline.create_start_pa_outcome(pbp)

# ------------------------- BATTERS-FACED CASCADE (the floor) ------------------ #
# Same building blocks k_predictor's count_distribution_check already wires up —
# reused here as-is (not modified in place, per implementation_plan.md).
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

# ------------------------- HANDEDNESS ----------------------------------------- #
pitcher_hand = (
    pbp[["pitcher_id", "pitcher_throw_hand"]]
    .drop_duplicates(subset=["pitcher_id"])
    .rename(columns={"pitcher_id": "personId"})
)
start_outcome = start_outcome.merge(pitcher_hand, on="personId", how="left")


# ── 4. Season-based train / val / test split ─────────────────────────────────
FEATURE_COLS = [
    "pitcher_last_season_start_pa_avg_pa_per_start",
    "pitcher_last_season_start_pa_n_starts",
    "pitcher_this_season_start_pa_avg_pa_per_start",
    "pitcher_this_season_start_pa_starts_n",
    "team_last_season_avg_pa_per_start",
    "league_last_season_avg_pa_per_start",
    "expected_batters_faced",
    "expected_batters_faced_weight",
    "pitcher_throw_hand",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in start_outcome.columns]
CASCADE_COL = "expected_batters_faced"  # the benchmark "model" — cascade's own point estimate
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
_eval("Linear regression", y_val.to_numpy(), lr.predict(Xval_sc))

print("Training XGBoost (default)...")
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(Xtr, y_train)
_eval("XGBoost (default)", y_val.to_numpy(), xgb_model.predict(Xval))

print("Training XGBoost (tuned)...")
# Hyperparameters and early-stopping setup carried over unchanged from
# experiments/xgb_vs_cascade_diagnostic/run.py, which found this combination
# beats the cascade (val MAE 2.7491 vs. 2.8609) on this same feature set —
# see that script and ROADMAP.md's 2026-08-26 follow-up entry for the
# overfitting diagnosis that motivated it.
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
_eval("XGBoost (tuned)", y_val.to_numpy(), xgb_tuned.predict(Xval_core))


# ── 6. Print results ──────────────────────────────────────────────────────────
CASCADE_NAME = "Cascade (expected_batters_faced)"
cascade_mae = results[CASCADE_NAME]["mae"]

print(f"\n{'=' * 72}")
print(f"BASELINE RESULTS — {MODEL_NAME}")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} starts)  |  Test season {TEST_SEASON} locked")
print("Primary: MAE (lower=better)  |  Secondary: RMSE, Bias, Pearson r")
print("=" * 72)
print(f"{'Model':<32} {'MAE':>8} {'vs cascade':>12}  {'RMSE':>8} {'Bias':>8} {'Pearson r':>10}")
print("-" * 72)
for name, res in results.items():
    delta = f"{res['mae'] - cascade_mae:+.4f}" if name != CASCADE_NAME else "—"
    print(f"{name:<32} {res['mae']:>8.4f} {delta:>12}  {res['rmse']:>8.4f} {res['bias']:>+8.4f} {res['pearson_r']:>10.4f}")
print("=" * 72)

print("\nInterpretation:")
candidates = {n: r for n, r in results.items() if n != CASCADE_NAME}
beats_floor = {n: r for n, r in candidates.items() if r["mae"] < cascade_mae}
best_name, best = min(candidates.items(), key=lambda t: t[1]["mae"])

if not beats_floor:
    print(f"  No model beats the cascade floor ({CASCADE_NAME}, MAE {cascade_mae:.4f}).")
    print("  Combining the cascade's own inputs non-linearly doesn't improve on the")
    print("  hand-built shrinkage formula — new information (v1's candidate features)")
    print("  is needed, not a better combiner of what the cascade already sees.")
else:
    print(f"  {best_name} beats the cascade floor ({CASCADE_NAME}, MAE {cascade_mae:.4f}) —")
    print(f"  {best_name} MAE {best['mae']:.4f}. Worth carrying forward into a real experiment.")

lr_mae = results["Linear regression"]["mae"]
xgb_default_mae = results["XGBoost (default)"]["mae"]
xgb_tuned_mae = results["XGBoost (tuned)"]["mae"]
print(f"  XGBoost (default) vs (tuned): {xgb_default_mae:.4f} -> {xgb_tuned_mae:.4f}"
      f" ({xgb_tuned_mae - xgb_default_mae:+.4f}) — tuning alone, no new features.")
if abs(xgb_tuned_mae - lr_mae) / max(lr_mae, 1e-9) < 0.02:
    print("  Linear vs tuned XGBoost: within ~2% — relationship appears largely linear.")
else:
    print("  Linear vs tuned XGBoost: >2% gap — some nonlinear structure XGBoost is picking up.")
print("=" * 72)


# ── 6b. Quartile breakdown by expected_batters_faced_weight ──────────────────
# Same stratification the cascade's own Story-0 baseline capture used
# (ROADMAP.md) — Q1 thinnest-sample -> Q4 most-reliable this-season sample.
print(f"\n{'=' * 72}\nMAE/RMSE/Bias/Pearson r BY {WEIGHT_COL} QUARTILE (direct comparability with\nthe cascade's own already-published quartile breakdown in ROADMAP.md)\n{'=' * 72}")

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
quartile_df = pd.DataFrame(quartile_rows)


# ── 7. Plots ───────────────────────────────────────────────────────────────
PLOT_DIR = BASE_DIR / "plots" / "baseline-model"
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
ax.scatter(y_val, best["pred"], alpha=0.3, s=12, color="steelblue")
lims = [min(y_val.min(), best["pred"].min()), max(y_val.max(), best["pred"].max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("realized_batters_faced (actual)")
ax.set_ylabel("predicted")
ax.set_title(f"{best_name} predicted vs. actual — val ({VAL_SEASON})")
plt.tight_layout()
plt.savefig(PLOT_DIR / "residuals.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'residuals.png'}")

feature_names = num_cols + cat_cols
importances = pd.Series(xgb_tuned.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, max(4, len(importances) * 0.35)))
ax.barh(importances.index, importances.values, color="steelblue")
ax.set_title("Feature importance — XGBoost (tuned)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'feature_importance.png'}")


# ── 8. Write baseline_results.md ──────────────────────────────────────────────
md_lines = [
    f"# Baseline Results — {MODEL_NAME}",
    "",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
    "**Task:** Regression (starting-pitcher-start grain) — the first regression",
    "target in this repo's model layer (every sibling model predicts a binary",
    "outcome).  ",
    "**Target:** realized_batters_faced  ",
    "**Primary metric:** MAE (lower = better)  ",
    "**Diagnostics:** RMSE, Bias (signed mean error), Pearson r  ",
    f"**Data:** s3://{BUCKET}  ",
    "",
    "## Split",
    "",
    "| Split | Seasons | Rows | Mean realized_batters_faced |",
    "|-------|---------|------|------------------------------|",
    f"| Train | {FIT_SEASONS} | {len(train_df):,} | {y_train.mean():.2f} |",
    f"| Val   | {VAL_SEASON} | {len(val_df):,} | {y_val.mean():.2f} |",
    f"| Test  | {TEST_SEASON} | {len(test_df):,} | locked — not evaluated here |",
    "",
    "## Results (evaluated on val)",
    "",
    "The floor is NOT a global-mean naive — it's the already-shipped shrinkage",
    "cascade (`game_context.build_expected_batters_faced`), included here as a",
    "benchmark row rather than a trained model. Every candidate model is fed the",
    "SAME inputs the cascade formula already has access to — this baseline asks",
    "whether a regressor combining those inputs non-linearly beats the",
    "hand-built shrinkage formula, before any new feature engineering.",
    "",
    "| Model | MAE | Δ vs cascade | RMSE | Bias | Pearson r |",
    "|-------|-----|---------------|------|------|-----------|",
]
for name, res in results.items():
    delta = f"{res['mae'] - cascade_mae:+.4f}" if name != CASCADE_NAME else "—"
    md_lines.append(f"| {name} | {res['mae']:.4f} | {delta} | {res['rmse']:.4f} | {res['bias']:+.4f} | {res['pearson_r']:.4f} |")

if not beats_floor:
    interpretation = (
        f"No model beats the cascade floor (**{CASCADE_NAME}**, MAE {cascade_mae:.4f}). "
        "Combining the cascade's own inputs non-linearly doesn't improve on the hand-built "
        "shrinkage formula — real new information (opposing-lineup on-base rate, rest days, "
        "home/away, handedness matchup, pitch-efficiency trend — v1's candidate features) is "
        "needed next, not a better combiner of what the cascade already sees."
    )
else:
    interpretation = (
        f"**{best_name}** beats the cascade floor (**{CASCADE_NAME}**, MAE {cascade_mae:.4f}) "
        f"with MAE {best['mae']:.4f} — worth carrying forward into a real experiment."
    )

md_lines += [
    "",
    "## Interpretation",
    "",
    interpretation,
    "",
    f"## MAE/RMSE/Bias/Pearson r by {WEIGHT_COL} quartile",
    "",
    "Same stratification as the cascade's own Story-0 baseline capture (ROADMAP.md)",
    "— Q1 (thinnest this-season sample) through Q4 (most reliable).",
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
    "## Setup",
    "",
    f"- Features: {FEATURE_COLS}",
    "- No new feature engineering — every feature here is an input the",
    "  expected_batters_faced shrinkage cascade itself already uses",
    "  (pitcher/team/league last-season avg PA/start, this-season rolling avg",
    "  PA/start + starts_n, the cascade's own point estimate + weight,",
    "  handedness). New candidate features (opposing-lineup on-base rate, rest",
    "  days, home/away, pitch-efficiency trend) are deliberately deferred to v1.",
    "- Grain: one row per (personId, gamepk) starting-pitcher start — same as",
    "  short_outing_predictor, different from every PA/batter-game-grain",
    "  sibling. Scoped to REALIZED pitcher_role == 'sp' by construction.",
    "- Label: realized_batters_faced,",
    "  batters_faced_predictor.processing.pipeline.create_start_pa_outcome.",
    "- build_expected_batters_faced / build_pitcher_start_pa_this_season are",
    "  NOT modified — called as-is, same reasoning that already keeps",
    "  build_expected_start_innings frozen for its own dependents.",
    "- XGBoost (tuned) hyperparameters and its held-out-season early-stopping",
    "  setup are carried over unchanged from",
    "  `experiments/xgb_vs_cascade_diagnostic/run.py`, which diagnosed default",
    "  XGBoost as overfitting at this data size (train MAE far below val MAE;",
    "  worked examples showed it predicting implausibly low batters-faced",
    "  counts for established starters) — see ROADMAP.md's 2026-08-26",
    "  follow-up entry.",
    "",
    "## Plots",
    "",
    "- `plots/baseline-model/target_distribution.png`",
    "- `plots/baseline-model/residuals.png` — best model's predicted vs. actual",
    "- `plots/baseline-model/feature_importance.png` — XGBoost (tuned)",
    "",
    "## Next steps",
    "",
    "- If a model beats the cascade floor: move to a real v1 experiment, add",
    "  opposing-lineup on-base/walk rate (`rolling_stats.build_team_batter_onbase_rolling_feats`),",
    "  team rest days (`game_context.build_team_rest_days`), home/away, and a",
    "  pitch-efficiency proxy — and carry the tuned XGBoost hyperparameters",
    "  forward rather than reverting to defaults.",
    "- If not: the cascade's inputs alone, recombined, aren't enough — the same",
    "  new-feature set above is still the next step, just with lower prior odds",
    "  of a quick win from recombination alone.",
    "- Re-run the bf_gap-quartile floor-isolation check",
    "  (k_predictor/experiments/count_distribution_check/run.py's method,",
    "  documented in BENCHMARKS.md) with whichever estimate wins, substituted",
    "  for expected_batters_faced, to see whether the closest-quartile floor",
    "  MAE itself moves.",
    "- Final evaluation on test season (2025) only once in a real experiment.",
]

results_path = BASE_DIR / "baseline_results.md"
results_path.write_text("\n".join(md_lines) + "\n")
print(f"\nSaved {results_path}")
