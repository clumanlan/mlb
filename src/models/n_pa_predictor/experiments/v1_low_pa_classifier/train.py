"""
n_pa_predictor experiment v1: low-PA-tail classifier.
Run from src/models/n_pa_predictor/ with: python experiments/v1_low_pa_classifier/train.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    v1 SUMMARY                                #
# ---------------------------------------------------------------------------- #
# Pivots away from the baseline (baseline/model/run.py — regression on the full
# n_pa count, all batting orders, evaluated by MAE). That baseline came back
# negative twice: first with (batting_order, home/away, expected_start_innings,
# batter's own rolling PA rate, team win_pct), then again after adding the
# opposing starter's season WHIP — global-mean naive (MAE 0.4955) beat every
# model both times. XGBoost's own feature_importance showed WHIP WAS being used
# (~0.035 importance, not zero), just swamped by batting_order (~0.50) — the
# effect is real but MAE on the full n_pa distribution is structurally the wrong
# lens to see it: 94.7% of starter-games land in {3,4,5} PA, so a constant
# prediction is already close almost everywhere, and WHIP's effect is a
# probability shift at the margin (3-vs-4 PA), not a count shift big enough to
# move whole-population MAE.
#
# This experiment reframes the problem entirely, per a live investigation
# (2026-08-23) into WHY n_pa<=3 happens: pulled real 2024 batter-games and found
# (a) batting_order alone is a huge, monotonic driver (5.3% low-tail rate at
# leadoff -> 49.4% at 9-hole), and (b) controlling for batting_order>=6, the
# opposing starter's WHIP correlates with the TEAM's total PA that game at
# +0.47 (strong) while K-rate correlates at ~0.00 (essentially nothing) — the
# mechanism compressing a lineup's PA count is traffic allowed (walks+hits),
# not swing-and-miss stuff.
#
# So: TARGET changes from n_pa (regression) to low_pa = n_pa <= LOW_PA_THRESHOLD
# (binary classification, LOW_PA_THRESHOLD=3). FEATURE_COLS are UNCHANGED from
# the baseline (same batting_order / is_home / expected_start_innings /
# opp_starter_whip_season / batter's own rolling PA rate / team win_pct/
# runs_scored) — this experiment tests whether the SAME signal, correctly
# framed and evaluated, is usable, not whether new features help.
#
# Full dataset, NOT filtered to batting_order>=6 — deliberate choice over a
# bottom-of-order-only submodel: a tree model can learn the batting_order x
# WHIP interaction on its own (it already had both features candidate; this
# experiment is the first time it's asked the right QUESTION of them), and one
# general model handles order 1-5 too (predicting low probability there, which
# is simply correct — their low-tail rate is 5-14%, not because the model is
# broken but because the phenomenon barely exists for top-of-order hitters).
#
# EVALUATION PRIORITIZES PRECISION, not accuracy or recall — per the stated use
# case, a "predict low PA" call feeds a betting decision, so a false positive
# (predicted low_pa=1, batter actually gets 4+) is the costly error, and a
# missed low-tail case (false negative) just means passing on a bet, not
# taking a bad one. Primary reporting is therefore a PRECISION-RECALL curve
# and a confidence-threshold sweep (precision + coverage at each threshold),
# not a single accuracy/F1 number at the default 0.5 cut. Two naive floors:
# global base rate, and a per-batting-order-slot historical rate (the harder,
# non-ML floor — same "harder floor" discipline as the baseline's rolling-avg
# naive). Note the slot-only naive's own ceiling: its highest possible
# predicted probability is 49.4% (9-hole's historical rate) — it can NEVER
# produce a 60%+ confidence prediction. Any real high-confidence "bet with
# certainty" signal has to come from combining batting_order with something
# else (WHIP), which is exactly what this experiment tests for.
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
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
from models.n_pa_predictor.utils.mlflow_logging import (
    create_run_id, log_evaluation_to_mlflow, get_git_sha,
)

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.processing.features import rolling_stats

import models.n_pa_predictor.processing.pipeline as pipeline
from models.n_pa_predictor.processing.features.batter_playing_time import build_batter_pa_rolling_stats

STAGE = Path(__file__).parent.name
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PLOT_DIR = BASE_DIR / "plots" / "v1_low_pa_classifier"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LOW_PA_THRESHOLD = 3
STARTER_IP_SHRINKAGE_K = 5.0
# Focused on the 0.6-0.9 operating region, not the whole probability range —
# this is a selective/low-volume betting use case (per user direction
# 2026-08-23), so the low-threshold/high-recall end isn't the relevant
# decision surface. Finer granularity (0.05 steps) from 0.7 up, since that's
# specifically where an apparent LR-vs-XGBoost "crossover" turned out (see
# WILSON_Z below) to be sample-size noise, not real — worth resolving finely.
CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
WILSON_Z = 1.96  # 95% CI

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

TARGET = "low_pa"  # overrides config.yaml's n_pa (regression) — this experiment's own target

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)  # same reason as baseline — needs a prior season's pbp for the shift

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
# Identical assembly to baseline/model/run.py — see that file for the join-by-
# join rationale. Only the label (step 4) and everything downstream differs.
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

# ------------------------- OPPOSING STARTER'S EXPECTED INNINGS -------------- #
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

# ------------------------- OPPOSING STARTER'S SEASON WHIP -------------------- #
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

# ------------------------- BATTER'S OWN ROLLING PA RATE ---------------------- #
batter_pa_rolling = build_batter_pa_rolling_stats(batter_game, window="season")
batter_game = batter_game.merge(
    batter_pa_rolling.drop(columns=["game_date", "game_season"]),
    on=["batter_id", "gamepk"], how="left",
)

# ------------------------- BATTING TEAM'S ROLLING RECORD --------------------- #
team_win_loss_season = game_context.build_team_win_loss_record(schedule, window="season")
team_win_loss_season["team_id"] = team_win_loss_season["team_id"].astype(str)
batter_game = batter_game.merge(
    team_win_loss_season.drop(columns=["game_date", "game_datetime", "game_season"]).rename(
        columns={"team_id": "batter_team_id"}
    ),
    on=["batter_team_id", "gamepk"], how="left",
)


# ── 4. Season-based train / val / test split ─────────────────────────────────
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
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"low_pa (n_pa<={LOW_PA_THRESHOLD}) rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

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
        X_tr[cat_cols] = X_tr[cat_cols].astype(object)
        X_ev[cat_cols] = X_ev[cat_cols].astype(object)

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


# ── 5. Naive floors ────────────────────────────────────────────────────────────
# (a) global base rate — weakest floor, one number for every row.
# (b) per-batting-order-slot historical rate — the harder, non-ML floor. Its
#     own ceiling matters: the highest slot rate (9-hole) tops out well under
#     50%, so this floor can NEVER produce a high-confidence prediction —
#     tracked explicitly in the threshold sweep below.
slot_rate = train_df.groupby("batting_order")[TARGET].mean()
print("\nPer-slot low_pa rate (train, the harder naive floor):")
print(slot_rate.round(3))

naive_global_prob = pd.Series(y_train.mean(), index=y_val.index)
naive_slot_prob = val_df["batting_order"].map(slot_rate).fillna(y_train.mean())


# ── 6. Train models ────────────────────────────────────────────────────────────
print("\nTraining logistic regression...")
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xval_sc = scaler.transform(Xval)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(Xtr_sc, y_train)
lr_prob = lr.predict_proba(Xval_sc)[:, 1]

print("Training XGBoost...")
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss",
)
xgb_model.fit(Xtr, y_train)
xgb_prob = xgb_model.predict_proba(Xval)[:, 1]

probs = {
    "Naive (global rate)": naive_global_prob.values,
    "Naive (per-slot rate)": naive_slot_prob.values,
    "Logistic regression": lr_prob,
    "XGBoost": xgb_prob,
}


# ── 7. Diagnostic metrics (log loss, Brier, ROC-AUC, PR-AUC) ─────────────────
diag = {}
for name, p in probs.items():
    diag[name] = {
        "log_loss": log_loss(y_val, p, labels=[0, 1]),
        "brier": brier_score_loss(y_val, p),
        "roc_auc": roc_auc_score(y_val, p),
        "pr_auc": average_precision_score(y_val, p),
    }

print(f"\n{'='*76}")
print("DIAGNOSTIC METRICS (val, n=%s, base rate=%.3f)" % (f"{len(val_df):,}", y_val.mean()))
print("=" * 76)
print(f"{'Model':<24} {'LogLoss':>9} {'Brier':>8} {'ROC-AUC':>9} {'PR-AUC':>8}")
print("-" * 76)
for name, m in diag.items():
    print(f"{name:<24} {m['log_loss']:>9.4f} {m['brier']:>8.4f} {m['roc_auc']:>9.4f} {m['pr_auc']:>8.4f}")
print("=" * 76)


# ── 8. Precision-focused: confidence-threshold sweep ─────────────────────────
# This is the number that matters for a betting decision: "if I only act when
# the model says >= X% confident, what's my real hit rate (precision), and how
# many opportunities does that give me (coverage)?"
#
# Wilson score interval on precision, not just the point estimate — at high
# thresholds N gets thin fast (val 2024 alone gives n=60-237 above 0.7), and a
# raw precision number with no uncertainty band invites over-trusting exactly
# the kind of small-sample noise that produced an apparent LR-vs-XGBoost
# "crossover" around 0.8 that a 95% CI shows is not statistically real (CIs
# overlap almost completely there) — vs. the 0.6 threshold, where LR's and
# XGBoost's CIs barely overlap, a real (if modest) difference.
def wilson_ci(p, n, z=WILSON_Z):
    if n == 0:
        return (np.nan, np.nan)
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


sweep_rows = []
for name, p in probs.items():
    for t in CONFIDENCE_THRESHOLDS:
        pred = (p >= t).astype(int)
        n_pred = int(pred.sum())
        if n_pred == 0:
            sweep_rows.append({
                "model": name, "threshold": t, "n": 0,
                "precision": np.nan, "recall": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            })
            continue
        precision = precision_score(y_val, pred, zero_division=np.nan)
        ci_low, ci_high = wilson_ci(precision, n_pred)
        sweep_rows.append({
            "model": name, "threshold": t, "n": n_pred,
            "precision": precision,
            "recall": recall_score(y_val, pred, zero_division=np.nan),
            "ci_low": ci_low, "ci_high": ci_high,
        })
sweep_df = pd.DataFrame(sweep_rows)

print("\nCONFIDENCE-THRESHOLD SWEEP (precision + 95% Wilson CI + coverage)")
print("=" * 76)
for name in probs:
    print(f"\n{name}:")
    sub = sweep_df[sweep_df["model"] == name]
    print(f"  {'Threshold':>10} {'N':>8} {'Precision':>10} {'95% CI':>18} {'Recall':>8}")
    for _, r in sub.iterrows():
        if pd.isna(r["precision"]):
            print(f"  {r['threshold']:>10.2f} {r['n']:>8,} {'—':>10} {'—':>18} {'—':>8}")
            continue
        ci_str = f"[{r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}]"
        print(f"  {r['threshold']:>10.2f} {r['n']:>8,} {r['precision']:>10.3f} {ci_str:>18} {r['recall']:>8.3f}")
print("=" * 76)


# ── 9. Interpretation ──────────────────────────────────────────────────────────
xgb_diag = diag["XGBoost"]
best_naive_pr_auc = max(diag["Naive (global rate)"]["pr_auc"], diag["Naive (per-slot rate)"]["pr_auc"])

print("\nInterpretation:")
if xgb_diag["pr_auc"] <= best_naive_pr_auc:
    print(f"  XGBoost PR-AUC ({xgb_diag['pr_auc']:.4f}) does not beat the best naive floor")
    print(f"  ({best_naive_pr_auc:.4f}). Reframing to classification did not, on its own,")
    print("  surface usable signal — see the threshold sweep above for whether ANY")
    print("  model reaches a usable precision at a non-trivial threshold regardless.")
else:
    print(f"  XGBoost PR-AUC ({xgb_diag['pr_auc']:.4f}) beats the best naive floor")
    print(f"  ({best_naive_pr_auc:.4f}) — real signal. Check the threshold sweep above for")
    print("  the highest threshold with both acceptable precision AND non-trivial N —")
    print("  that's the actual usable operating point, not the PR-AUC number alone.")
xgb_sub = sweep_df[(sweep_df["model"] == "XGBoost") & sweep_df["precision"].notna()]
naive_slot_sub = sweep_df[(sweep_df["model"] == "Naive (per-slot rate)") & sweep_df["precision"].notna()]
naive_slot_ceiling = slot_rate.max()
print(f"\n  Naive (per-slot rate)'s own ceiling: max threshold it ever clears is "
      f"{naive_slot_sub['threshold'].max() if len(naive_slot_sub) else float('nan')}"
      f" (highest slot's historical rate is {naive_slot_ceiling:.3f}) — it CANNOT reach")
print("  confidence higher than that ceiling by construction. Any prediction above it has to")
print("  come from XGBoost/LR")
print("  combining batting_order with WHIP (or other features), not from the lookup table alone.")
print("=" * 76)


# ── 10. Plots ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = y_train.value_counts().sort_index()
ax.bar([f"n_pa > {LOW_PA_THRESHOLD}", f"n_pa ≤ {LOW_PA_THRESHOLD}"], counts.values, color=["steelblue", "tomato"])
for i, v in enumerate(counts.values):
    ax.text(i, v + counts.max() * 0.01, f"{v:,}\n({v/len(y_train)*100:.1f}%)", ha="center", fontsize=9)
ax.set_title(f"low_pa target distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_ylabel("Batter-games")
plt.tight_layout()
plt.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'target_distribution.png'}")

fig, ax = plt.subplots(figsize=(7, 6))
colors = {"Naive (global rate)": "gray", "Naive (per-slot rate)": "silver", "Logistic regression": "steelblue", "XGBoost": "tomato"}
for name, p in probs.items():
    precision, recall, _ = precision_recall_curve(y_val, p)
    ax.plot(recall, precision, label=f"{name} (PR-AUC={diag[name]['pr_auc']:.3f})", color=colors.get(name))
ax.axhline(y_val.mean(), color="black", linestyle="--", linewidth=1, label=f"Base rate ({y_val.mean():.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall curve — val ({VAL_SEASON})\nlow_pa = n_pa ≤ {LOW_PA_THRESHOLD}")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(PLOT_DIR / "precision_recall_curve.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'precision_recall_curve.png'}")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
for name, p in probs.items():
    frac_pos, mean_pred = calibration_curve(y_val, p, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=name, color=colors.get(name))
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives (actual low_pa rate)")
ax.set_title(f"Calibration curve — val ({VAL_SEASON})")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "calibration_curve.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'calibration_curve.png'}")

feature_names = num_cols + cat_cols
importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, max(4, len(importances) * 0.35)))
ax.barh(importances.index, importances.values, color="steelblue")
ax.set_title("Feature importance — XGBoost (low_pa classifier)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'feature_importance.png'}")


# ── 11. MLflow logging ────────────────────────────────────────────────────────
for name, model, p in (("Logistic regression", lr, lr_prob), ("XGBoost", xgb_model, xgb_prob)):
    metrics = {
        **diag[name],
        **{
            f"precision_at_{t}": sweep_df[(sweep_df["model"] == name) & (sweep_df["threshold"] == t)]["precision"].iloc[0]
            for t in CONFIDENCE_THRESHOLDS
        },
        **{
            f"n_at_{t}": sweep_df[(sweep_df["model"] == name) & (sweep_df["threshold"] == t)]["n"].iloc[0]
            for t in CONFIDENCE_THRESHOLDS
        },
    }
    metrics = {k: v for k, v in metrics.items() if pd.notna(v)}

    log_evaluation_to_mlflow(
        metrics=metrics,
        params={
            **{k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None},
            "FIT_SEASONS": str(FIT_SEASONS),
            "VAL_SEASON": VAL_SEASON,
            "TEST_SEASON": TEST_SEASON,
            "LOW_PA_THRESHOLD": LOW_PA_THRESHOLD,
        },
        tags={
            "model_type": name,
            "stage": STAGE,
            "target": TARGET,
            "val_season": str(VAL_SEASON),
            "git_sha": get_git_sha(),
        },
        artifact_paths=[
            str(PLOT_DIR / "precision_recall_curve.png"),
            str(PLOT_DIR / "calibration_curve.png"),
            str(PLOT_DIR / "feature_importance.png"),
        ],
    )
print("\nLogged to MLflow (experiment: n_pa_predictor)")
