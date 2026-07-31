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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

BASE_DIR = Path(__file__).resolve().parent.parent



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



# batting average previous year (if none use median?) + 
# + batting rolling this season weighted?
# + average for batting order



"""
Rules-based expected batting average (xBA) baseline.

Three interpretable components, combined as a weighted average:
  1. comp_prev_season   -> last_season_ba, median-filled for rookies/missing
  2. comp_rolling_shrunk -> this-season, play-by-play cumulative BA,
                             shrunk toward season league average, computed
                             WITHOUT lookahead (only uses PAs strictly
                             before the current play)
  3. comp_order_slot    -> previous season's league-wide BA by batting
                             order slot (coach's-judgment proxy)

Design decisions baked in (change these if your intuition disagrees --
that's the point of this baseline):
  - last_season_ba is NOT shrunk (no PA count available for it, and it's
    already a full-season stat where shrinkage matters less)
  - rolling BA shrinkage uses empirical Bayes: (hits + k*lg_avg) / (PA + k)
  - order-slot table is computed per season from THIS data, then shifted
    forward one season so 2024 rows use 2023's order-slot table
  - all three components fall back to a league median when unavailable
    (rookie with no last_season_ba, first PA of the season, new order
    slot never seen last season)
"""


def add_last_season_component(df: pd.DataFrame) -> pd.DataFrame:
    """Component 1: last season's BA, median-filled."""
    df = df.copy()
    league_median = df['last_season_ba'].median()
    df['comp_prev_season'] = df['last_season_ba'].fillna(league_median)
    return df


def add_rolling_season_component(df: pd.DataFrame, k: float = 100.0) -> pd.DataFrame:
    """
    Component 2: this-season cumulative BA per batter, shrunk toward the
    season's league average, computed WITHOUT lookahead.

    Sort order matters here -- we sort by season/date/game/play so the
    cumulative sum for a given play only reflects PAs that happened
    strictly earlier. Ties within the same game_date are broken by
    gamepk/play_id, which is an approximation (true at-bat sequence
    would be better if you have a more granular ordering key).
    """
    df = df.copy()
    df = df.sort_values(['batter_id', 'game_season', 'game_date', 'gamepk', 'play_id'])

    grp = df.groupby(['batter_id', 'game_season'])
    # cumulative hits/PA BEFORE this play (subtract current row's own outcome)
    df['_cum_hits_before'] = grp['is_hit'].cumsum() - df['is_hit']
    df['_cum_pa_before'] = grp.cumcount()

    # season league average, used as the shrinkage target
    season_league_avg = df.groupby('game_season')['is_hit'].transform('mean')

    df['comp_rolling_shrunk'] = (
        (df['_cum_hits_before'] + k * season_league_avg) /
        (df['_cum_pa_before'] + k)
    )

    df = df.drop(columns=['_cum_hits_before', '_cum_pa_before'])
    return df


def add_batting_order_component(df: pd.DataFrame) -> pd.DataFrame:
    """
    Component 3: previous season's league-wide BA by batting order slot.
    A 2024 play looks up the order-slot table built from 2023 data.
    """
    df = df.copy()

    order_stats = (
        df.groupby(['game_season', 'batting_order'])['is_hit']
        .mean()
        .reset_index()
        .rename(columns={'is_hit': 'order_slot_ba'})
    )
    # shift forward: stats computed in season S apply as a lookup for season S+1
    order_stats['game_season'] = order_stats['game_season'] + 1

    df = df.merge(
        order_stats,
        on=['game_season', 'batting_order'],
        how='left'
    )

    league_median = df['order_slot_ba'].median()
    df['comp_order_slot'] = df['order_slot_ba'].fillna(league_median)
    df = df.drop(columns=['order_slot_ba'])
    return df


def combine_xba(
    df: pd.DataFrame,
    w_prev: float = 0.4,
    w_roll: float = 0.4,
    w_order: float = 0.2,
) -> pd.DataFrame:
    """Weighted average of the three components. Weights should sum to 1."""
    assert abs((w_prev + w_roll + w_order) - 1.0) < 1e-9, "weights must sum to 1"
    df = df.copy()
    df['xba_pred'] = (
        w_prev * df['comp_prev_season'] +
        w_roll * df['comp_rolling_shrunk'] +
        w_order * df['comp_order_slot']
    )
    return df


def build_rules_baseline(
    df: pd.DataFrame,
    k: float = 100.0,
    w_prev: float = 0.4,
    w_roll: float = 0.4,
    w_order: float = 0.2,
) -> pd.DataFrame:
    """
    Full pipeline: returns the input dataframe with four new columns:
      comp_prev_season, comp_rolling_shrunk, comp_order_slot, xba_pred
    """
    out = df.copy()
    out = add_last_season_component(out)
    out = add_rolling_season_component(out, k=k)
    out = add_batting_order_component(out)
    out = combine_xba(out, w_prev=w_prev, w_roll=w_roll, w_order=w_order)
    return out




# CREATE SOME BASELINE FUNCTION HERE 

# we're going to build a utility function taht runs the same metrics on each output:
# - one summary performance state 
# - calibration in different bins: ideally we would like the tails optimized for,
# - so we're worreried more about precision on both ends
# - let's vverify with claude that this is the case



# ── 4. Season-based train / val / test split ─────────────────────────────────

na_subset_to_check_later = ['batSide', 'pitcher_hand', 'game_season']

pa = build_rules_baseline(pa, k=100, w_prev=0.4, w_roll=0.4, w_order=0.2)
model_df = pa.dropna(subset=na_subset_to_check_later).copy()
model_df = model_df[~model_df['batting_order'].isnull()]  # remove substitute batters

model_df = model_df[['game_season','comp_rolling_shrunk', 'comp_order_slot', 'xba_pred'] + [TARGET, DATE_COL]]

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"Hit rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

y_train = train_df[TARGET]
y_val   = val_df[TARGET]

eps = 1e-6  # log_loss is undefined at exact 0/1

# ── 5. Naive floor — predict the train hit rate for every PA ─────────────────

naive_prob = np.full(len(y_val), y_train.mean())
naive_prob = np.clip(naive_prob, eps, 1 - eps)
naive_brier = brier_score_loss(y_val, naive_prob)
naive_ll = log_loss(y_val, naive_prob, labels=[0, 1])

results = {"Naive baseline": {"log_loss": naive_ll, "brier": naive_brier}}

# ── 6. Rules-based baseline ───────────────────────────────────────────────────
# Runs on the FULL `pa` frame (not model_df/val_df) because the rolling/
# shrinkage component needs the complete within-season play sequence to
# stay leakage-safe. We slice down to val_df's exact rows afterward.

rules_df = build_rules_baseline(pa, k=100, w_prev=0.4, w_roll=0.4, w_order=0.2)

rules_prob = rules_df.loc[val_df.index, 'xba_pred'].to_numpy()
n_missing = np.isnan(rules_prob).sum()
if n_missing:
    print(f"Warning: {n_missing} rules_prob NaNs, filling with train hit rate")
    rules_prob = np.nan_to_num(rules_prob, nan=y_train.mean())
rules_prob = np.clip(rules_prob, eps, 1 - eps)

rules_brier = brier_score_loss(y_val, rules_prob)
rules_ll = log_loss(y_val, rules_prob, labels=[0, 1])

print(f"\nRules-based baseline — LogLoss: {rules_ll:.4f}  Brier: {rules_brier:.4f}")
print(f"  Δ LogLoss vs naive: {rules_ll - naive_ll:+.4f}")
print(f"  Δ Brier   vs naive: {rules_brier - naive_brier:+.4f}")

results["Rules-based baseline"] = {"log_loss": rules_ll, "brier": rules_brier}

# per-component breakdown — which single rule is doing the most work?
print("\nRules-based components, evaluated individually (val):")
print(f"{'Component':<28} {'LogLoss':>9} {'Brier':>9}")
print("-" * 48)
component_cols = {
    'comp_prev_season': 'Rules: prev season only',
    'comp_rolling_shrunk': 'Rules: rolling shrunk only',
    'comp_order_slot': 'Rules: order slot only',
}
for col, label in component_cols.items():
    p = rules_df.loc[val_df.index, col].to_numpy()
    p = np.nan_to_num(p, nan=y_train.mean())
    p = np.clip(p, eps, 1 - eps)
    ll = log_loss(y_val, p, labels=[0, 1])
    br = brier_score_loss(y_val, p)
    print(f"{label:<28} {ll:>9.4f} {br:>9.4f}")

calibration_models = {
    "Naive baseline": naive_prob,
    "Rules-based baseline": rules_prob,
}
colors = ["gray", "seagreen"]

# ── 7. Plots ─────────────────────────────────────────────────────────────────

PLOT_DIR = BASE_DIR / "plots" / "baseline-model"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Target distribution (train set)
fig, ax = plt.subplots(figsize=(6, 4))
counts = y_train.value_counts().sort_index()
ax.bar(["No hit (0)", "Hit (1)"], counts.values, color=["steelblue", "tomato"])
for i, v in enumerate(counts.values):
    ax.text(i, v + counts.max() * 0.01, f"{v:,}\n({v/len(y_train)*100:.1f}%)", ha="center", fontsize=9)
ax.set_title(f"Target distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_ylabel("Plate appearances")
plt.tight_layout()
plt.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
plt.close()
print("\nSaved plots/baseline-model/target_distribution.png")

# Calibration curves — how honest are the probability estimates?
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

print("\nBrier scores (lower = better, naive sets the floor):")
print(f"{'Model':<22} {'Brier':>8} {'vs Naive':>10}")
print("-" * 42)

for (name, probs), color in zip(calibration_models.items(), colors):
    fraction_of_positives, mean_predicted = calibration_curve(
        y_val, probs, n_bins=10, strategy="quantile"
    )
    ax.plot(mean_predicted, fraction_of_positives, marker="o", linewidth=2,
            color=color, label=name)

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

# Log loss by predicted-probability bucket — tails matter most to you.
bucket_edges = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 1.01]

print("\nRules-based baseline log loss by predicted probability bucket (val):")
print(f"{'Bucket':<14} {'N':>8} {'Mean pred':>11} {'Hit rate':>10} {'LogLoss':>9}")
print("-" * 54)
for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
    mask = (rules_prob >= lo) & (rules_prob < hi)
    n = int(mask.sum())
    if n == 0:
        continue
    bucket_ll = log_loss(y_val[mask], rules_prob[mask], labels=[0, 1])
    print(f"[{lo:.2f}, {hi:.2f})   {n:>8,} {rules_prob[mask].mean():>11.3f} "
          f"{y_val[mask].mean():>10.3f} {bucket_ll:>9.4f}")

# Predicted probability distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(rules_prob[y_val == 0], bins=50, alpha=0.6, label="No hit", color="steelblue", density=True)
ax.hist(rules_prob[y_val == 1], bins=50, alpha=0.6, label="Hit",    color="tomato",    density=True)
ax.axvline(y_train.mean(), color="black", linestyle="--", linewidth=1,
           label=f"Train hit rate ({y_train.mean():.3f})")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Density")
ax.set_title(f"Rules-based predicted probability distribution — val ({VAL_SEASON})")
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "predicted_proba_distribution.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'predicted_proba_distribution.png'}")

# ── 8. Write baseline_results.md ─────────────────────────────────────────────

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
    "Rules-based baseline is a hand-weighted blend of prev-season BA, this-season",
    "shrunk rolling BA (k=100), and previous-season BA-by-batting-order-slot.",
    "",
    "| Model | LogLoss | Δ vs Naive | % improvement | Brier | Δ vs Naive |",
    "|-------|---------|------------|---------------|-------|------------|",
]
for name, res in results.items():
    if name == "Naive baseline":
        ll_d = "—"
        pct = "—"
        br_d = "—"
    else:
        ll_d = f"{res['log_loss'] - naive_ll:+.4f}"
        pct = f"{(naive_ll - res['log_loss']) / naive_ll * 100:+.2f}%"
        br_d = f"{res['brier'] - naive_brier:+.4f}"
    md_lines.append(
        f"| {name} | {res['log_loss']:.4f} | {ll_d} | {pct} | {res['brier']:.4f} | {br_d} |"
    )

md_lines += [
    "",
    "## Rules-based baseline — component breakdown (val)",
    "",
    "| Component | LogLoss | Brier |",
    "|-----------|---------|-------|",
]
for col, label in component_cols.items():
    p = rules_df.loc[val_df.index, col].to_numpy()
    p = np.nan_to_num(p, nan=y_train.mean())
    p = np.clip(p, eps, 1 - eps)
    ll = log_loss(y_val, p, labels=[0, 1])
    br = brier_score_loss(y_val, p)
    md_lines.append(f"| {label} | {ll:.4f} | {br:.4f} |")

md_lines += [
    "",
    "## Setup",
    "",
    "- last_season_ba: 2016→2017, 2019→2022 (covid gap bridge), then year-over-year",
    "- Rules-based baseline weights: w_prev=0.4, w_roll=0.4, w_order=0.2, k=100",
    "",
    "## Plots",
    "",
    "- `plots/baseline-model/calibration_curve.png` — key diagnostic, quantile bins",
    "- `plots/baseline-model/predicted_proba_distribution.png`",
    "- `plots/baseline-model/target_distribution.png`",
    "",
    "## Next steps",
    "",
    "- Look at where the component breakdown is weakest and revisit that rule",
    "- Tune w_prev/w_roll/w_order and k against val, not by hand-guessing",
    "- Check bucket-level calibration at the tails specifically",
    "- Final evaluation on test season (2025) only once, in train.py",
]

with open(BASE_DIR / "baseline_results.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"\nSaved {BASE_DIR / 'baseline_results.md'}")
print("\nDone.")