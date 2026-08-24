"""
k_predictor experiment v1: pitcher workload rolling features.
Run from src/models/k_predictor/ with: python experiments/v1_pitcher_workload/train.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    v1 SUMMARY                                #
# ---------------------------------------------------------------------------- #
# Extends the baseline (baseline/model/run.py — expected_pitcher_role +
# season-level pitcher/batter strikeout_rate + handedness only; logistic
# regression beat naive, PR-AUC 0.2698 vs 0.2257). This experiment adds the
# pitcher-side boxscore workload features hit_predictor already uses (WHIP,
# innings pitched) in three forms, per the standing convention of exposing
# raw sample-size denominators alongside rates rather than only a single
# blended number:
#
#   1. SEASON (last season): pitcher_last_season_whip — already computed by
#      hit_predictor's season_stats.build_pitcher_stats_all_roles, just not
#      wired into the baseline (which only used the pbp-based K-rate table).
#   2. ROLLING RAW (this season, in-season, no shrinkage): pitcher_roll_season_ip,
#      _whip, _k_rate, _bb_rate, _hr_rate, _games_n (boxscore-based, via
#      rolling_stats.build_pitcher_rolling_stats_all_roles) + pitcher_roll_season_pa_total
#      ("plate appearances seen" — batters faced) and pitcher_roll_season_pa_strikeout_rate
#      (pbp-based, via rolling_stats.build_pbp_pitcher_rolling_feats_all_roles).
#      games_n and pa_total are both already-existing sample-size denominators
#      in hit_predictor's rolling tables — games_n needed one small addition
#      (see rolling_stats.build_pitcher_rolling_stats, added this session).
#   3. ROLLING SHRUNK TO LAST SEASON: pitcher_shrunk_whip — a new blend
#      (k_predictor.processing.features.pitcher_workload.build_pitcher_shrunk_whip),
#      same empirical-Bayes shrinkage-weight pattern as hit_predictor's
#      game_context.build_expected_start_innings (games_n / (games_n + k)),
#      generalized to WHIP and to both pitcher roles (not SP-only). This
#      guards against an early-season fluke rolling WHIP (thin sample)
#      dominating the raw rolling feature above.
#
# Deliberately basic for this pass, per the user's own framing — no batter-
# side rolling features, no pitch-level (stuff/command) features, no
# times_through_order. Evaluated the same way as the baseline (PR-AUC
# primary, ROC-AUC secondary) for a direct, apples-to-apples comparison.
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
from models.hit_predictor.utils.eval import (
    run_pa_vs_game_grain_check, aggregate_pa_predictions_to_game,
    evaluate_hit_predictor, summarize_verdict, plot_calibration_curve,
)

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip

STAGE = Path(__file__).parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent

WHIP_SHRINKAGE_K = 20.0

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

# ------------------------- 1. SEASON (LAST SEASON) --------------------------- #
pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_pa_strikeout_rate"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})

pitcher_box_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_whip"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})

batter_season_stats = season_stats.build_pbp_batter_feats(pbp)[
    ["batter_id", "game_season", "batter_last_season_pa_strikeout_rate"]
]

pa_outcome = pa_outcome.merge(
    pitcher_role_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome = pa_outcome.merge(
    pitcher_box_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome = pa_outcome.merge(batter_season_stats, on=["game_season", "batter_id"], how="left")

# ------------------------- 2. ROLLING RAW (THIS SEASON) ---------------------- #
# Boxscore-based: IP, WHIP, K/BB/HR rate, games pitched — per-game grain,
# joins on gamepk (not game_season) since it updates every game.
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

# pbp-based: plate appearances seen (batters faced) + rolling K rate — richer,
# updates every game rather than once a season.
pitcher_pbp_rolling = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window="season")
pbp_rolling_cols = ["pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate"]
pa_outcome = pa_outcome.merge(
    pitcher_pbp_rolling.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + pbp_rolling_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

# "average innings pitched" — a direct ratio of the two raw rolling counts above.
pa_outcome["pitcher_roll_season_avg_ip_per_game"] = (
    pa_outcome["pitcher_roll_season_ip"] / pa_outcome["pitcher_roll_season_games_n"].replace(0, np.nan)
)

# ------------------------- 3. ROLLING SHRUNK TO LAST SEASON ------------------ #
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


# ── 4. Season-based train / val / test split ─────────────────────────────────
FEATURE_COLS = [
    "expected_pitcher_role",
    "pitcher_throw_hand",
    "batter_bat_side",
    # 1. season (last season)
    "pitcher_last_season_pa_strikeout_rate",
    "pitcher_last_season_whip",
    "batter_last_season_pa_strikeout_rate",
    # 2. rolling raw (this season)
    "pitcher_roll_season_ip",
    "pitcher_roll_season_whip",
    "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate",
    "pitcher_roll_season_hr_rate",
    "pitcher_roll_season_games_n",
    "pitcher_roll_season_avg_ip_per_game",
    "pitcher_roll_season_pa_total",
    "pitcher_roll_season_pa_strikeout_rate",
    # 3. rolling shrunk to last season
    "pitcher_shrunk_whip",
    "pitcher_shrunk_whip_weight",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in pa_outcome.columns]

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
print(f"Strikeout rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

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
        # .astype(object) alone leaves pandas' nullable-string missing marker
        # (pd.NA) in place, which SimpleImputer can't handle — fillna(np.nan)
        # normalizes to the plain float NaN sklearn expects.
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

print("Evaluating naive (per expected-pitcher-role K rate)...")
role_rate = train_df.groupby(NAIVE_ROLE_COL)[TARGET].mean()
naive_role_pred = X_val[NAIVE_ROLE_COL].map(role_rate).fillna(y_train.mean())
_eval("Naive (per-role K rate)", y_val, naive_role_pred.to_numpy())

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
role_pr_auc = results["Naive (per-role K rate)"]["pr_auc"]
best_naive_name, best_naive_pr_auc = max(
    (("Naive (most frequent)", naive_pr_auc), ("Naive (per-role K rate)", role_pr_auc)),
    key=lambda t: t[1],
)

# baseline/model/run.py's own result, hardcoded here for a direct comparison
# printout — not re-derived, since the baseline used a smaller, frozen
# feature set (see that file for its own numbers).
BASELINE_LR_PR_AUC = 0.2698
BASELINE_LR_ROC_AUC = 0.5758

print(f"\n{'='*72}")
print(f"EXPERIMENT RESULTS — {MODEL_NAME} v1 (pitcher workload rolling features)")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} PAs)  |  Test season {TEST_SEASON} locked")
print("Primary: PR-AUC (higher=better)  |  Secondary: ROC-AUC (higher=better)")
print("=" * 72)
print(f"{'Model':<28} {'PR-AUC':>8} {'vs best naive':>14}  {'ROC-AUC':>8}")
print("-" * 72)
for name, res in results.items():
    delta = f"{res['pr_auc'] - best_naive_pr_auc:+.4f}" if name not in ("Naive (most frequent)", "Naive (per-role K rate)") else "—"
    print(f"{name:<28} {res['pr_auc']:>8.4f} {delta:>14}  {res['roc_auc']:>8.4f}")
print("-" * 72)
print(f"{'(baseline LR, for reference)':<28} {BASELINE_LR_PR_AUC:>8.4f} {'':>14}  {BASELINE_LR_ROC_AUC:>8.4f}")
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

vs_baseline = best["pr_auc"] - BASELINE_LR_PR_AUC
if vs_baseline > 0.005:
    print(f"  vs baseline's logistic regression (PR-AUC {BASELINE_LR_PR_AUC:.4f}): {vs_baseline:+.4f} —")
    print("  pitcher workload rolling features add real signal beyond season-level rate alone.")
elif vs_baseline < -0.005:
    print(f"  vs baseline's logistic regression (PR-AUC {BASELINE_LR_PR_AUC:.4f}): {vs_baseline:+.4f} —")
    print("  worse than the smaller baseline feature set — check for a wiring bug before concluding")
    print("  workload features don't help.")
else:
    print(f"  vs baseline's logistic regression (PR-AUC {BASELINE_LR_PR_AUC:.4f}): {vs_baseline:+.4f} — flat,")
    print("  no demonstrated improvement from adding pitcher workload rolling features this pass.")
print("=" * 72)


# ── 7. Game-grain aggregation check ───────────────────────────────────────────
# Rolls PA-grain predictions up to batter-game grain — "will this batter
# strike out at least once in this game," the grain a DK strikeout prop
# actually resolves on — same approach hit_predictor uses for its own "1+
# hits" game-grain check (BENCHMARKS.md §4.5): game_pred_prob = 1 - prod(1-p)
# over that batter's PAs that game, compared against the properly-aggregated
# naive floor (not just a raw uncertainty term), per the same correction
# hit_predictor's own check needed. Reused directly from hit_predictor's
# utils/eval.py — it's generic on any 0/1 target despite "is_hit"/"hit"
# naming internally; those are cosmetic leftovers, not hit-specific logic.
print("\n" + "=" * 72)
print("GAME-GRAIN CHECK — batter-game \"1+ strikeout\" (same approach as hit_predictor)")
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
PLOT_DIR = STAGE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

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
cm = confusion_matrix(y_val, results["XGBoost"]["pred"])
ConfusionMatrixDisplay(cm, display_labels=["No K", "K"]).plot(ax=ax, cmap="Blues", colorbar=False)
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
        "whip_shrinkage_k": WHIP_SHRINKAGE_K,
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
            "stage": "v1_pitcher_workload",
            "target": TARGET,
            "val_season": str(VAL_SEASON),
            "git_sha": get_git_sha() or "unknown",
        },
        artifact_paths=artifact_paths,
    )
print("\nLogged all runs to MLflow (experiment: k_predictor).")
