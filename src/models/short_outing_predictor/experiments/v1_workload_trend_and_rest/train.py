"""
short_outing_predictor experiment v1: trailing-3-start IP workload trend +
team rest days.
Run from src/models/short_outing_predictor/ with:
  python experiments/v1_workload_trend_and_rest/train.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    v1 SUMMARY                                #
# ---------------------------------------------------------------------------- #
# Extends the baseline (baseline/model/run.py — last-season/this-season(season-
# to-date)/league IP-per-start + expected_start_innings blend + handedness;
# logistic regression beat naive, PR-AUC 0.4199 vs 0.3094). This experiment
# adds two of README's named mechanism pieces the baseline explicitly deferred:
#
#   1. TRAILING-3-START IP TREND (recent workload, not season-to-date):
#      pitcher_last3_start_ip_avg_ip_per_start / _starts_n, via
#      game_context.build_pitcher_start_ip_this_season(..., window=3) — that
#      function's window param didn't exist before this experiment; generalized
#      it from a hardcoded 'season' (expanding) rolling window to accept an int
#      (trailing N-start) window too, same window='season'-vs-int distinction
#      _rolling_sum's own docstring already draws for team stats. Backward
#      compatible: default window='season' preserves every existing caller's
#      column names/values (tests/hit_predictor/test_game_context.py's
#      test_build_pitcher_start_ip_this_season_default_window_is_season_and_
#      unchanged). A season-to-date average dilutes a real recent trend (e.g.
#      a pitcher who's been shortened his last 3 outings after a hot start)
#      — a trailing window is the more responsive signal README's "recent
#      workload/pitch-count trend" mechanism actually describes.
#   2. TEAM REST DAYS: team_days_since_last_game, via
#      game_context.build_team_rest_days(schedule) — already exists, joined
#      on the pitcher's own team_id + gamepk (pulled straight from
#      pitcher_boxscore, not re-derived). Approximates README's "bullpen rest
#      state" / "day after a doubleheader" mechanisms at team-wide grain (not
#      literally bullpen-specific — no per-reliever workload table exists
#      yet) — the closest available proxy without new feature engineering.
#
# Deliberately NOT added this pass (see baseline_results.md's Next steps and
# ROADMAP.md for why):
#   - Explicit doubleheader flag: build_doubleheader_flag exists but needs
#     schedule's raw 'doubleheader' code column, which isn't in
#     processed_data/games/schedule's SCHEDULE_COLUMNS yet (documented gap,
#     CLAUDE.md's Known Issues) — would need a schedule reprocessing backfill
#     first, out of scope here.
#   - Opponent platoon-advantage depth: needs new pre-game opposing-lineup-
#     handedness-composition engineering that doesn't exist anywhere in this
#     codebase yet (unlike the two features above, which were "wire up what
#     exists" work) — a real feature-engineering task for a v2, not this pass.
#   - Rotation/IL status: no data source in this pipeline at all.
#
# Evaluated the same way as the baseline (PR-AUC primary, ROC-AUC secondary)
# for a direct, apples-to-apples comparison. Unlike every sibling model, no
# PA->game aggregation check is needed here — this model already predicts at
# start grain, the same grain a short-outing/bullpen-day prop resolves on —
# so Section 7 instead runs the full reliability+resolution decision-metric
# verdict (utils/eval.py::evaluate_hit_predictor / summarize_verdict, reused
# from hit_predictor) directly against the naive floor at this final grain.
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
from models.short_outing_predictor.utils.mlflow_logging import log_evaluation_to_mlflow, get_git_sha

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.utils.eval import evaluate_hit_predictor, summarize_verdict, plot_calibration_curve

import models.short_outing_predictor.processing.pipeline as pipeline

STAGE = Path(__file__).parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STARTER_IP_SHRINKAGE_K = 5.0
TRAILING_WINDOW = 3  # last-3-starts trailing window for the new workload-trend feature

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
# Same reason as baseline/every sibling model: build_pitcher_start_ip_stats
# needs a prior season's pbp for the shift, which isn't loaded for 2016.
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

print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", all_boxscore_seasons,
)

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)


# ── 3. Build start-grain DataFrame ────────────────────────────────────────────
print("\nBuilding start-grain DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = hp_pipeline.build_pbp_features(pbp, schedule, player_info)

start_outcome = pipeline.create_start_outcome(pitcher_boxscore, pbp)

# ------------------------- 1. STARTER INNINGS ESTIMATE (baseline features) --- #
pitcher_start_ip_last_season = season_stats.build_pitcher_start_ip_stats(pitcher_boxscore, pbp)
league_avg_start_ip = season_stats.build_league_avg_start_ip(pitcher_start_ip_last_season)
pitcher_start_ip_this_season = game_context.build_pitcher_start_ip_this_season(pitcher_boxscore, pbp)
expected_start_innings = game_context.build_expected_start_innings(
    pitcher_start_ip_last_season, pitcher_start_ip_this_season, league_avg_start_ip,
    k=STARTER_IP_SHRINKAGE_K,
)
expected_start_innings["personId"] = expected_start_innings["personId"].astype(str)
expected_start_innings["gamepk"] = expected_start_innings["gamepk"].astype(str)

start_outcome = start_outcome.drop(columns=["game_date", "game_season"]).merge(
    expected_start_innings, on=["personId", "gamepk"], how="left",
)

# ------------------------- 2. TRAILING-3-START IP TREND (NEW) ---------------- #
pitcher_start_ip_trailing = game_context.build_pitcher_start_ip_this_season(
    pitcher_boxscore, pbp, window=TRAILING_WINDOW,
)[["personId", "gamepk", f"pitcher_last{TRAILING_WINDOW}_start_ip_starts_n",
   f"pitcher_last{TRAILING_WINDOW}_start_ip_avg_ip_per_start"]]
pitcher_start_ip_trailing["personId"] = pitcher_start_ip_trailing["personId"].astype(str)
pitcher_start_ip_trailing["gamepk"] = pitcher_start_ip_trailing["gamepk"].astype(str)
start_outcome = start_outcome.merge(pitcher_start_ip_trailing, on=["personId", "gamepk"], how="left")

# ------------------------- 3. TEAM REST DAYS (NEW) --------------------------- #
# Pitcher's own team's rest state — the closest available proxy for README's
# "bullpen rest state"/"day after a doubleheader" mechanisms without a
# per-reliever workload table or a working doubleheader flag (see header).
pitcher_team = pitcher_boxscore[["personId", "gamepk", "team_id"]].assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
    team_id=lambda x: x["team_id"].astype(str),
)
team_rest_days = game_context.build_team_rest_days(schedule)[["team_id", "gamepk", "team_days_since_last_game"]]
team_rest_days["team_id"] = team_rest_days["team_id"].astype(str)
team_rest_days["gamepk"] = team_rest_days["gamepk"].astype(str)

start_outcome = start_outcome.merge(pitcher_team, on=["personId", "gamepk"], how="left")
start_outcome = start_outcome.merge(team_rest_days, on=["team_id", "gamepk"], how="left")

# ------------------------- 4. HANDEDNESS (baseline feature) ------------------ #
pitcher_hand = (
    pbp[["pitcher_id", "pitcher_throw_hand"]]
    .drop_duplicates(subset=["pitcher_id"])
    .rename(columns={"pitcher_id": "personId"})
)
start_outcome = start_outcome.merge(pitcher_hand, on="personId", how="left")


# ── 4. Season-based train / val / test split ─────────────────────────────────
FEATURE_COLS = [
    # baseline features
    "pitcher_last_season_start_ip_avg_ip_per_start",
    "pitcher_last_season_start_ip_n_starts",
    "pitcher_this_season_start_ip_avg_ip_per_start",
    "pitcher_this_season_start_ip_starts_n",
    "league_last_season_avg_ip_per_start",
    "expected_start_innings",
    "expected_start_innings_weight",
    "pitcher_throw_hand",
    # 1. trailing-3-start IP trend (recent workload)
    f"pitcher_last{TRAILING_WINDOW}_start_ip_avg_ip_per_start",
    f"pitcher_last{TRAILING_WINDOW}_start_ip_starts_n",
    # 2. team rest days
    "team_days_since_last_game",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in start_outcome.columns]

NAIVE_BUCKET_COL = "expected_start_innings"

model_df = start_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"Short-outing rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

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

print("Evaluating naive (per expected-start-innings-bucket rate)...")
train_bucket = train_df[NAIVE_BUCKET_COL].round(0)
bucket_rate = train_df.assign(_bucket=train_bucket).groupby("_bucket")[TARGET].mean()
val_bucket = X_val[NAIVE_BUCKET_COL].round(0)
naive_bucket_pred = val_bucket.map(bucket_rate).fillna(y_train.mean())
_eval("Naive (per-innings-bucket rate)", y_val, naive_bucket_pred.to_numpy())

print("Training logistic regression...")
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xval_sc = scaler.transform(Xval)
lr = LogisticRegression(max_iter=1000)
lr.fit(Xtr_sc, y_train)
_eval("Logistic regression", y_val, lr.predict_proba(Xval_sc)[:, 1])

print("Training XGBoost...")
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss")
xgb_model.fit(Xtr, y_train)
_eval("XGBoost", y_val, xgb_model.predict_proba(Xval)[:, 1])


# ── 6. Print results ──────────────────────────────────────────────────────────
naive_pr_auc = results["Naive (most frequent)"]["pr_auc"]
bucket_pr_auc = results["Naive (per-innings-bucket rate)"]["pr_auc"]
NAIVE_NAMES = ("Naive (most frequent)", "Naive (per-innings-bucket rate)")
best_naive_name, best_naive_pr_auc = max(
    (("Naive (most frequent)", naive_pr_auc), ("Naive (per-innings-bucket rate)", bucket_pr_auc)),
    key=lambda t: t[1],
)

# baseline/model/run.py's own result, hardcoded here for a direct comparison
# printout — not re-derived, since the baseline used a smaller, frozen
# feature set (see that file for its own numbers).
BASELINE_LR_PR_AUC = 0.4199
BASELINE_LR_ROC_AUC = 0.6772

print(f"\n{'='*72}")
print(f"EXPERIMENT RESULTS — {MODEL_NAME} v1 (workload trend + team rest)")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} starts)  |  Test season {TEST_SEASON} locked")
print("Primary: PR-AUC (higher=better)  |  Secondary: ROC-AUC (higher=better)")
print("=" * 72)
print(f"{'Model':<32} {'PR-AUC':>8} {'vs best naive':>14}  {'ROC-AUC':>8}")
print("-" * 72)
for name, res in results.items():
    delta = f"{res['pr_auc'] - best_naive_pr_auc:+.4f}" if name not in NAIVE_NAMES else "—"
    print(f"{name:<32} {res['pr_auc']:>8.4f} {delta:>14}  {res['roc_auc']:>8.4f}")
print("-" * 72)
print(f"{'(baseline LR, for reference)':<32} {BASELINE_LR_PR_AUC:>8.4f} {'':>14}  {BASELINE_LR_ROC_AUC:>8.4f}")
print("=" * 72)

print("\nInterpretation:")
candidates = {n: r for n, r in results.items() if n not in NAIVE_NAMES}
beats_floor = {n: r for n, r in candidates.items() if r["pr_auc"] > best_naive_pr_auc}
best_model_name, best_model = max(candidates.items(), key=lambda t: t[1]["pr_auc"])

if not beats_floor:
    print(f"  No model beats the best naive floor ({best_naive_name}, PR-AUC {best_naive_pr_auc:.4f}).")
else:
    print(f"  {best_model_name} beats the best naive floor ({best_naive_name}, PR-AUC {best_naive_pr_auc:.4f}) —")
    print(f"  {best_model_name} PR-AUC {best_model['pr_auc']:.4f}.")

vs_baseline = best_model["pr_auc"] - BASELINE_LR_PR_AUC
if vs_baseline > 0.005:
    print(f"  vs baseline's logistic regression (PR-AUC {BASELINE_LR_PR_AUC:.4f}): {vs_baseline:+.4f} —")
    print("  workload trend / team rest add real signal beyond the baseline feature set.")
elif vs_baseline < -0.005:
    print(f"  vs baseline's logistic regression (PR-AUC {BASELINE_LR_PR_AUC:.4f}): {vs_baseline:+.4f} —")
    print("  worse than the smaller baseline feature set — check for a wiring bug before concluding")
    print("  the new features don't help.")
else:
    print(f"  vs baseline's logistic regression (PR-AUC {BASELINE_LR_PR_AUC:.4f}): {vs_baseline:+.4f} — flat,")
    print("  no demonstrated improvement from adding workload trend / team rest this pass.")
print("=" * 72)


# ── 7. Reliability + resolution verdict at start grain ───────────────────────
# No PA->game rollup needed here, unlike every sibling model — this model
# already predicts at start grain, the grain a short-outing/bullpen-day prop
# resolves on. Runs hit_predictor's own decision-metric framework
# (evaluate_hit_predictor/summarize_verdict, same convention as config.yaml's
# decision_metrics for hit_predictor and every sibling's game-grain check)
# directly against the naive-per-innings-bucket floor at this final grain.
print("\n" + "=" * 72)
print(f"RELIABILITY + RESOLUTION VERDICT — {best_model_name} vs naive, start grain")
print("=" * 72)

base_rate = float(y_val.mean())
print("\n--- Naive (per-innings-bucket rate) ---")
naive_metrics = evaluate_hit_predictor(
    y_true=y_val, y_prob=naive_bucket_pred.to_numpy(), base_rate=base_rate, n_bins=10, min_n=200,
)
print(f"\n--- {best_model_name} ---")
best_metrics = evaluate_hit_predictor(
    y_true=y_val, y_prob=best_model["prob"], base_rate=base_rate, n_bins=10, min_n=200,
)

verdict = summarize_verdict(naive_metrics, best_metrics)
print(f"\nVerdict vs naive at start grain: {verdict['verdict']}")
print(f"  reliability delta: {verdict['reliability_delta']:+.4f} (lower=better, negative=more honest)")
print(f"  resolution delta:  {verdict['resolution_delta']:+.4f} (higher=better, positive=more real spread)")
print("=" * 72)

PLOT_DIR = STAGE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
plot_calibration_curve(
    y_val,
    {
        "Naive (per-innings-bucket rate)": {"proba": naive_bucket_pred.to_numpy()},
        best_model_name: {"proba": best_model["prob"]},
    },
    n_bins=10, min_n=50,
    save_path=PLOT_DIR / "calibration.png",
)
print(f"Saved {PLOT_DIR / 'calibration.png'}")


# ── 8. Plots ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
train_df[TARGET].value_counts().sort_index().plot(kind="bar", color="steelblue", ax=ax)
ax.set_title(f"is_short_outing distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_xlabel("is_short_outing")
ax.set_ylabel("starts")
plt.tight_layout()
plt.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'target_distribution.png'}")

fig, ax = plt.subplots(figsize=(5, 5))
cm = confusion_matrix(y_val, results["XGBoost"]["pred"])
ConfusionMatrixDisplay(cm, display_labels=["Full start", "Short outing"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"XGBoost confusion matrix — val ({VAL_SEASON})")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'confusion_matrix.png'}")

feature_names = num_cols + cat_cols
importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, max(4, len(importances) * 0.35)))
ax.barh(importances.index, importances.values, color="steelblue")
ax.set_title("Feature importance — XGBoost")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'feature_importance.png'}")


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
        "fit_seasons": str(FIT_SEASONS),
    }
    if name == best_model_name:
        metrics.update({
            "start_reliability": best_metrics["reliability"],
            "start_resolution": best_metrics["resolution"],
            "start_roc_auc": best_metrics["roc_auc"],
        })
        params["start_grain_verdict"] = verdict["verdict"]
        artifact_paths.append(PLOT_DIR / "calibration.png")

    log_evaluation_to_mlflow(
        metrics=metrics,
        params=params,
        tags={
            "stage": "v1_workload_trend_and_rest",
            "target": TARGET,
            "val_season": str(VAL_SEASON),
            "git_sha": get_git_sha() or "unknown",
        },
        artifact_paths=artifact_paths,
    )
print(f"\nLogged all runs to MLflow (experiment: {MODEL_NAME}).")
