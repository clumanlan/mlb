"""
Baseline hit predictor.
Run from models/hit_predictor/ with: python run_baseline.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Split strategy:
  train = all train_seasons except val_season and test_season
  val   = val_season  (iterate against this during development)
  test  = test_season (locked away — final eval only in train.py)
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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score, ConfusionMatrixDisplay
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent

# ── 1. Config ────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET           = cfg["bucket"]
REGION           = cfg["region"]
TRAIN_SEASONS    = cfg["train_seasons"]
FEATURE_SEASONS  = cfg["feature_seasons"]
TARGET           = cfg["target_column"]
DATE_COL         = cfg["date_column"]
TEST_SEASON      = cfg["test_season"]
VAL_SEASON       = cfg["val_season"]
MODEL_NAME       = cfg["model_name"]

# Seasons used for model fitting (everything that isn't val or test)
FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]

HITS = {"Single", "Double", "Triple", "Home Run"}
PA_OUTCOMES = HITS | {
    "Strikeout", "Groundout", "Flyout", "Lineout", "Pop Out",
    "Forceout", "Grounded Into DP", "Double Play", "Triple Play",
    "Fielders Choice", "Fielders Choice Out", "Field Error",
    "Bunt Groundout", "Bunt Pop Out", "Bunt Lineout",
    "Strikeout Double Play", "Field Out", "Batter Out",
    "Walk", "Intent Walk", "Hit By Pitch", "Catcher Interference",
}


boto_session = boto3.Session(region_name=REGION)


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
    "s3://{bucket}/processed_data/prepared/playbyplay/{season}/",
    TRAIN_SEASONS,
    chunked=True,
)

print("\nLoading schedule...")
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/",
    TRAIN_SEASONS,
)

print("\nLoading game info...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/",
    TRAIN_SEASONS,
)

all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))
print("\nLoading batter boxscore...")
boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/",
    all_boxscore_seasons,
)

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/",
    boto3_session=boto_session,
)


# ── 3. Build PA-level DataFrame ───────────────────────────────────────────────
print("\nBuilding PA-level DataFrame...")

pa_cols = [c for c in ["gamepk", "play_id", "play_result", "batter_id", "pitcher_id",
                        "batter_team_id", "pitcher_team_id"] if c in pbp.columns]
pa = (
    pbp[pa_cols]
    .drop_duplicates(subset=["gamepk", "play_id"], keep="first")
    .reset_index(drop=True)
)
pa = pa[pa["play_result"].isin(PA_OUTCOMES)].copy()
pa[TARGET]    = pa["play_result"].isin(HITS).astype(int)
pa["gamepk"]  = pa["gamepk"].astype(str)
pa["batter_id"]  = pa["batter_id"].astype(str)
pa["pitcher_id"] = pa["pitcher_id"].astype(str)

# Join game_date from schedule
schedule["gamepk"]   = schedule["gamepk"].astype(str)
schedule["game_date"] = pd.to_datetime(schedule["game_date"])
pa = pa.merge(
    schedule[["gamepk", "game_date"]].drop_duplicates("gamepk"),
    on="gamepk", how="left",
)

# figure this out later why some game play by plays are missing from game info 
# Join game_season + weather from game_info
game_info["gamepk"] = game_info["gamepk"].astype(str)
pa = pa.merge(
    game_info[["gamepk", "game_season", "weather_condition", "weather_temp"]].drop_duplicates("gamepk"),
    on="gamepk", how="left",
)
pa["weather_temp"] = pa["weather_temp"].astype(str).str.extract(r"(\d+)")[0].astype(float)

# Join batting_order from batter boxscore
boxscore["gamepk"]   = boxscore["gamepk"].astype(str)
boxscore["personId"] = boxscore["personId"].astype(str)
batting_order = (
    boxscore[["gamepk", "personId", "batting_order"]]
    .drop_duplicates(subset=["gamepk", "personId"])
    .rename(columns={"personId": "batter_id"})
)
batting_order["batting_order"] = pd.to_numeric(batting_order["batting_order"], errors="coerce")
pa = pa.merge(batting_order, on=["gamepk", "batter_id"], how="left")

# Compute last_season_ba
# For each training season Y, use BA from the most recent available feature season < Y.
# This correctly bridges the covid gap: 2022 → 2019 BA.
boxscore["game_season"] = pd.to_numeric(boxscore.get("game_season", np.nan), errors="coerce")
if boxscore["game_season"].isna().all() and "game_date" in boxscore.columns:
    boxscore["game_season"] = pd.to_datetime(boxscore["game_date"]).dt.year

season_ba = (
    boxscore[boxscore["game_season"].isin(FEATURE_SEASONS)]
    .groupby(["personId", "game_season"])
    .agg(hits=("h", "sum"), abs=("ab", "sum"))
    .reset_index()
)
season_ba["ba"] = (season_ba["hits"] / season_ba["abs"].replace(0, np.nan)).round(3)

available_feature_seasons = sorted(season_ba["game_season"].unique())
lsba_rows = []
for target_season in TRAIN_SEASONS:
    prior = [s for s in available_feature_seasons if s < target_season]
    if not prior:
        continue
    subset = (
        season_ba[season_ba["game_season"] == max(prior)][["personId", "ba"]]
        .copy()
        .rename(columns={"ba": "last_season_ba"})
    )
    subset["game_season"] = target_season
    lsba_rows.append(subset)

last_season_ba = pd.concat(lsba_rows, ignore_index=True)
last_season_ba["personId"]    = last_season_ba["personId"].astype(str)
last_season_ba["game_season"] = last_season_ba["game_season"].astype(int)

pa["game_season"] = pd.to_numeric(pa["game_season"], errors="coerce").astype("Int64")
pa = pa.merge(
    last_season_ba.rename(columns={"personId": "batter_id"}),
    on=["batter_id", "game_season"], how="left",
)

# Join player attributes
player_info["person_id"] = player_info["person_id"].astype(str)

def height_to_inches(height_str):
    feet, inches = height_str.replace('"', '').split("'")
    return int(feet.strip()) * 12 + int(inches.strip())

player_info['height_in_inches'] = player_info['height'].apply(height_to_inches)

batter_attrs = (
    player_info[["person_id", "batSide", "weight", "height_in_inches", "strikeZoneTop", "strikeZoneBottom"]]
    .drop_duplicates("person_id")
    .rename(columns={"person_id": "batter_id"})
)
pitcher_attrs = (
    player_info[["person_id", "pitchHand"]]
    .drop_duplicates("person_id")
    .rename(columns={"person_id": "pitcher_id", "pitchHand": "pitcher_hand"})
)
pa = pa.merge(batter_attrs, on="batter_id", how="left")
pa = pa.merge(pitcher_attrs, on="pitcher_id", how="left")


# ── 4. Season-based train / val / test split ─────────────────────────────────
FEATURE_COLS = [
    "batting_order",     # lineup slot (1-9), known pre-game
    "batSide",           # batter handedness (L/R/S)
    "pitcher_hand",      # pitcher handedness (L/R) — key platoon split driver
    "game_season",       # year
    "weather_condition", # Sunny/Cloudy/etc.
    "weather_temp",      # temperature (°F)
    "weight",            # batter weight
    "height_in_inches",  # batter height
    "strikeZoneTop",     # batter-specific strike zone boundaries
    "strikeZoneBottom",
    "last_season_ba",    # prior-season batting average (main signal)
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in pa.columns]

na_subset_to_check_later = ['batSide', 'pitcher_hand', 'game_season']

model_df = pa[FEATURE_COLS + [TARGET, DATE_COL]].dropna(subset=na_subset_to_check_later).copy()
model_df["game_season"] = model_df["game_season"].astype(int)


train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"Hit rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_val   = val_df[FEATURE_COLS]
y_val   = val_df[TARGET]

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(X_train[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

def encode(X_tr, X_ev, cat_cols, num_cols):
    """Impute + encode; fit on train only, transform both."""
    X_tr = X_tr.copy()
    X_ev = X_ev.copy()

    # Coerce to numpy-compatible types
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
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        Xtr_cat = enc.fit_transform(Xtr_cat_imp)
        Xev_cat = enc.transform(Xev_cat_imp)
    else:
        Xtr_cat = np.empty((len(X_tr), 0))
        Xev_cat = np.empty((len(X_ev), 0))

    return np.hstack([Xtr_num, Xtr_cat]), np.hstack([Xev_num, Xev_cat])

Xtr, Xval = encode(X_train, X_val, cat_cols, num_cols)




# ── 6. Train three models, evaluate on val ───────────────────────────────────
results = {}

print("\nTraining naive baseline...")
naive = DummyClassifier(strategy="prior")
naive.fit(Xtr, y_train)
naive_prob = naive.predict_proba(Xval)[:, 1]
results["Naive baseline"] = {
    "log_loss": log_loss(y_val, naive_prob),
    "brier":  brier_score_loss(y_val, naive_prob),
    "proba":    naive_prob,
}

print("Training logistic regression...")
scaler   = StandardScaler()
Xtr_sc   = scaler.fit_transform(Xtr)
Xval_sc  = scaler.transform(Xval)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(Xtr_sc, y_train)
lr_prob = lr.predict_proba(Xval_sc)[:, 1]
results["Logistic regression"] = {
    "log_loss": log_loss(y_val, lr_prob),
    "brier":  brier_score_loss(y_val, lr_prob),
    "proba":    lr_prob,
}

print("Training XGBoost...")
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    n_estimators=100, random_state=42, verbosity=0,
    eval_metric="logloss", use_label_encoder=False,
)
xgb_model.fit(Xtr, y_train)
xgb_prob = xgb_model.predict_proba(Xval)[:, 1]
results["XGBoost"] = {
    "log_loss": log_loss(y_val, xgb_prob),
    "brier":    brier_score_loss(y_val, xgb_prob),
    "proba":    xgb_prob,
}


# ── 7. Print results ─────────────────────────────────────────────────────────
naive_ll    = results["Naive baseline"]["log_loss"]
naive_brier = results["Naive baseline"]["brier"]

print(f"\n{'='*64}")
print(f"BASELINE RESULTS — {MODEL_NAME}")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} PAs)  |  Test season {TEST_SEASON} locked")
print(f"Primary: log_loss (lower=better)  |  Secondary: brier (lower=better)")
print("="*64)
print(f"{'Model':<22} {'LogLoss':>9} {'vs Naive':>10}  {'Brier':>9} {'vs Naive':>10}")
print("-"*64)
for name, res in results.items():
    print(name)
    ll_delta    = res["log_loss"] - naive_ll
    brier_delta = res["brier"]    - naive_brier
    ll_str      = f"{ll_delta:+.4f}"    if name != "Naive baseline" else "—"
    brier_str   = f"{brier_delta:+.4f}" if name != "Naive baseline" else "—"
    print(f"{name:<22} {res['log_loss']:>9.4f} {ll_str:>10}  {res['brier']:>9.4f} {brier_str:>10}")


print("="*64)

xgb_ll = results["XGBoost"]["log_loss"]
lr_ll  = results["Logistic regression"]["log_loss"]
ll_nonlinear = (lr_ll - xgb_ll) / lr_ll * 100

print("\nInterpretation:")
if xgb_ll >= naive_ll and lr_ll >= naive_ll:
    print("  No model beats naive. Check target distribution, leakage, or data quality.")
elif ll_nonlinear > 3:
    print(f"  XGBoost meaningfully better than linear ({ll_nonlinear:.1f}% log_loss reduction).")
    print("  Nonlinear structure present — tree models warranted in train.py.")
elif ll_nonlinear < -1:
    print("  Linear model matches or beats XGBoost. Relationship appears largely linear.")
    print("  Consider calibrated logistic regression as the production baseline.")
else:
    print("  XGBoost ≈ linear model (within noise). Linear is a reasonable starting point.")

if (naive_ll - xgb_ll) / naive_ll * 100 < 1:
    print("  Both models show weak improvement over naive — features may need enrichment.")
print("="*64)


# ── 8. Plots ─────────────────────────────────────────────────────────────────

# Target distribution (train set)
fig, ax = plt.subplots(figsize=(6, 4))
counts = y_train.value_counts().sort_index()
ax.bar(["No hit (0)", "Hit (1)"], counts.values, color=["steelblue", "tomato"])
for i, v in enumerate(counts.values):
    ax.text(i, v + counts.max() * 0.01, f"{v:,}\n({v/len(y_train)*100:.1f}%)", ha="center", fontsize=9)
ax.set_title(f"Target distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_ylabel("Plate appearances")
plt.tight_layout()
plt.savefig(BASE_DIR / "plots/baseline/target_distribution.png", dpi=120)
plt.close()
print("\nSaved plots/baseline/target_distribution.png")

# Target distribution (train set)
fig, ax = plt.subplots(figsize=(6, 4))
counts = y_train.value_counts().sort_index()
ax.bar(["No hit (0)", "Hit (1)"], counts.values, color=["steelblue", "tomato"])
for i, v in enumerate(counts.values):
    ax.text(i, v + counts.max() * 0.01, f"{v:,}\n({v/len(y_train)*100:.1f}%)", ha="center", fontsize=9)
ax.set_title(f"Target distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_ylabel("Plate appearances")
plt.tight_layout()
plt.savefig(BASE_DIR / "plots/baseline/target_distribution.png", dpi=120)
plt.close()
print("\nSaved plots/baseline/target_distribution.png")

# Calibration curves — how honest are the probability estimates?
# For each model, bucket predicted probabilities and compare to actual hit rates.
# A perfectly calibrated model follows the diagonal: if it says 30%, hits happen 30% of the time.
# This is the key diagnostic before comparing against book implied probabilities.
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

PLOT_DIR = BASE_DIR / "plots" / "baseline"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

calibration_models = {
    "Naive baseline": naive_prob,
    "Logistic regression": lr_prob,
    "XGBoost": xgb_prob,
}
colors = ["gray", "steelblue", "tomato"]

print("\nBrier scores (lower = better, naive sets the floor):")
print(f"{'Model':<22} {'Brier':>8} {'vs Naive':>10}")
print("-" * 42)

naive_brier = brier_score_loss(y_val, naive_prob)

for (name, probs), color in zip(calibration_models.items(), colors):
    # Calibration curve — strategy="quantile" gives equal-count bins, which is
    # critical at the PA level: per-PA hit rate is ~0.22, so uniform bins above
    # 0.5 would be nearly empty. Quantile bins give reliable estimates across
    # the full predicted range, including your BTS decision range.
    fraction_of_positives, mean_predicted = calibration_curve(
        y_val, probs, n_bins=10, strategy="quantile"
    )
    ax.plot(mean_predicted, fraction_of_positives, marker="o", linewidth=2,
            color=color, label=name)

    # Brier score — mean squared error of probability predictions.
    # Companion to log loss: less sensitive to extreme confident errors,
    # more interpretable as "average squared distance from the truth".
    brier = brier_score_loss(y_val, probs)
    delta = f"{brier - naive_brier:+.4f}" if name != "Naive baseline" else "—"
    print(f"{name:<22} {brier:>8.4f} {delta:>10}")

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives (actual hit rate)")
ax.set_title(f"Calibration curves — val season {VAL_SEASON}\n"
             f"Closer to diagonal = more trustworthy probabilities")
ax.legend(loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(PLOT_DIR / "calibration_curve.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'calibration_curve.png'}")

# Log loss broken down by predicted-probability bucket — aggregate log loss
# can hide problems in your decision-relevant range. For BTS specifically,
# you care about calibration at the top end (predicted hit prob ≥ 0.30 at the
# PA level, which rolls up to the high-confidence game-level picks).
print("\nXGBoost log loss by predicted probability bucket (val):")
print(f"{'Bucket':<14} {'N':>8} {'Mean pred':>11} {'Hit rate':>10} {'LogLoss':>9}")
print("-" * 54)
bucket_edges = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 1.01]
for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
    mask = (xgb_prob >= lo) & (xgb_prob < hi)
    n = int(mask.sum())
    if n == 0:
        continue
    bucket_ll = log_loss(y_val[mask], xgb_prob[mask], labels=[0, 1])
    print(f"[{lo:.2f}, {hi:.2f})   {n:>8,} {xgb_prob[mask].mean():>11.3f} "
          f"{y_val[mask].mean():>10.3f} {bucket_ll:>9.4f}")

# Confusion matrix (XGBoost on val, threshold=0.5)
# Note: threshold=0.5 is mostly informational at PA level since hit rate ~0.22;
# the model will predict "no hit" for almost everything. Look at the
# probability distribution and calibration curve for the real story.
xgb_pred = (xgb_prob >= 0.5).astype(int)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_val, xgb_pred, ax=ax, colorbar=False)
ax.set_title(f"Confusion matrix — XGBoost on val ({VAL_SEASON})")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'confusion_matrix.png'}")

# Predicted probability distribution (XGBoost on val)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(xgb_prob[y_val == 0], bins=50, alpha=0.6, label="No hit", color="steelblue", density=True)
ax.hist(xgb_prob[y_val == 1], bins=50, alpha=0.6, label="Hit",    color="tomato",    density=True)
ax.axvline(y_train.mean(), color="black", linestyle="--", linewidth=1,
           label=f"Train hit rate ({y_train.mean():.3f})")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Density")
ax.set_title(f"XGBoost predicted probability distribution — val ({VAL_SEASON})")
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "predicted_proba_distribution.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'predicted_proba_distribution.png'}")

# Feature importance (XGBoost, top 20)
feature_names = num_cols + cat_cols


importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
top = importances.tail(20)
fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.35)))
ax.barh(top.index, top.values, color="steelblue")
ax.set_title("Feature importance — XGBoost (top 20)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'feature_importance.png'}")

# ── 9. Write baseline_results.md ─────────────────────────────────────────────
naive_ll = results["Naive baseline"]["log_loss"]

md_lines = [
    f"# Baseline Results — {MODEL_NAME}",
    "",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
    f"**Task:** Binary classification (per plate appearance)  ",
    f"**Target:** Did the batter record a hit in this PA?  ",
    f"**Primary metric:** log_loss (lower = better)  ",
    f"**Diagnostics:** brier score + calibration plot  ",
    f"**Data:** s3://{BUCKET}  ",
    "",
    "## Split",
    "",
    f"| Split | Seasons | Rows | Hit rate |",
    f"|-------|---------|------|----------|",
    f"| Train | {FIT_SEASONS} | {len(train_df):,} | {y_train.mean():.3f} |",
    f"| Val   | {VAL_SEASON} | {len(val_df):,} | {y_val.mean():.3f} |",
    f"| Test  | {TEST_SEASON} | {len(test_df):,} | locked — not evaluated here |",
    "",
    "## Results (evaluated on val)",
    "",
    "Naive baseline predicts the train hit rate for every PA — it sets the floor.",
    "Improvement vs naive is the real signal that the model is learning something.",
    "",
    "| Model | LogLoss | Δ vs Naive | % improvement | Brier | Δ vs Naive |",
    "|-------|---------|------------|---------------|-------|------------|",
]
for name, res in results.items():
    if name == "Naive baseline":
        ll_d = "—"
        pct  = "—"
        br_d = "—"
    else:
        ll_d = f"{res['log_loss'] - naive_ll:+.4f}"
        pct  = f"{(naive_ll - res['log_loss']) / naive_ll * 100:+.2f}%"
        br_d = f"{res['brier']    - naive_brier:+.4f}"
    md_lines.append(
        f"| {name} | {res['log_loss']:.4f} | {ll_d} | {pct} | {res['brier']:.4f} | {br_d} |"
    )

md_lines += [
    "",
    "## Setup",
    "",
    f"- Features: {FEATURE_COLS}",
    "- last_season_ba: 2016→2017, 2019→2022 (covid gap bridge), then year-over-year",
    "- Excluded: inning, half_inning, pitch_number (in-game); batter_id/pitcher_id (IDs)",
    "",
    "## Plots",
    "",
    "- `plots/baseline/calibration_curve.png` — key diagnostic, quantile bins",
    "- `plots/baseline/predicted_proba_distribution.png` — sanity check on output spread",
    "- `plots/baseline/confusion_matrix.png` — informational only at PA level",
    "- `plots/baseline/feature_importance.png`",
    "",
    "## Next steps",
    "",
    "- Add pitcher ERA / WHIP / K-rate features",
    "- Add rolling batting average windows (7/14/30 day) from feature store",
    "- Add ballpark and batter×pitcher handedness interaction",
    "- Aggregate PA-level predictions to game-level for BTS decision evaluation",
    "- Final evaluation on test season (2025) only once in train.py",
]

with open(BASE_DIR / "baseline_results.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"\nSaved {BASE_DIR / 'baseline_results.md'}")
print("\nDone.")