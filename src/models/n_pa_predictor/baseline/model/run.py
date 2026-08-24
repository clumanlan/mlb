"""
Baseline n_pa_predictor.
Run from src/models/n_pa_predictor/ with: python baseline/model/run.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Split strategy (same convention as hit_predictor):
  train = all train_seasons except val_season and test_season
  val   = val_season  (iterate against this during development)
  test  = test_season (locked away — final eval only in train.py)

Three baselines, in increasing sophistication:
  1. Naive (global): predict the train-set mean n_pa for every row.
  2. Naive (rolling): predict the batter's own trailing rolling avg_n_pa_per_game —
     the harder floor. Per ROADMAP/BENCHMARKS, this is the bar this model actually
     needs to clear before it's worth using over just reading the rolling average
     directly as a hit_predictor feature.
  3. Linear regression / XGBoost regressor: the actual candidate models.
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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.processing.features import rolling_stats

import models.n_pa_predictor.processing.pipeline as pipeline
from models.n_pa_predictor.processing.features.batter_playing_time import build_batter_pa_rolling_stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent


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
# Same reason as hit_predictor: build_pitcher_start_ip_stats needs a prior
# season's pbp for the shift, which isn't loaded for 2016.
FIT_SEASONS.remove(2017)

STARTER_IP_SHRINKAGE_K = 5.0

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
print("\nBuilding batter-game DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = hp_pipeline.build_pbp_features(pbp, schedule, player_info)

batter_game = pipeline.build_batter_game_frame(pbp, batter_boxscore, schedule)
batter_game["game_season"] = batter_game["game_date"].dt.year

# batter's own team + the opposing team's starter for this game — pbp already
# carries both (via build_pbp_features' _add_pbp_starting_pitcher) at PA grain;
# collapse to one row per (gamepk, team) for the join.
batter_team = (
    pbp[["gamepk", "batter_id", "batter_team_id"]].drop_duplicates(["gamepk", "batter_id"])
)
starter_by_team = (
    pbp[["gamepk", "pitcher_team_id", "starting_pitcher_id"]]
    .drop_duplicates(["gamepk", "pitcher_team_id"])
)
home = schedule[["gamepk", "home_id", "away_id"]].rename(
    columns={"home_id": "team_id", "away_id": "opp_team_id"}
)
away = schedule[["gamepk", "home_id", "away_id"]].rename(
    columns={"away_id": "team_id", "home_id": "opp_team_id"}
)
opp_team = pd.concat([home, away], ignore_index=True)
opp_team["gamepk"] = opp_team["gamepk"].astype(str)
opp_team["team_id"] = opp_team["team_id"].astype(str)
opp_team["opp_team_id"] = opp_team["opp_team_id"].astype(str)

batter_game = batter_game.merge(batter_team, on=["gamepk", "batter_id"], how="left")
batter_game = batter_game.merge(
    opp_team.rename(columns={"team_id": "batter_team_id"}), on=["gamepk", "batter_team_id"], how="left",
)
batter_game = batter_game.merge(
    starter_by_team.rename(
        columns={"pitcher_team_id": "opp_team_id", "starting_pitcher_id": "opp_starting_pitcher_id"}
    ),
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
# Traffic allowed (walks + hits), not strikeouts, is what compresses a team's total
# PA count that game — confirmed on real 2024 data (WHIP corr +0.47 with team PA
# count, K-rate corr ~0.00) before adding this. SP-role-only, this-season rolling,
# expanding + shift(1) — same point-in-time-safe engine as everything else here.
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

# Naive-rolling floor needs the batter's own rolling avg even where it's dropped
# from FEATURE_COLS for other models — it's already in FEATURE_COLS above.
NAIVE_ROLLING_COL = "batter_n_pa_roll_season_avg_n_pa_per_game"

model_df = batter_game[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
test_df  = model_df[model_df["game_season"] == TEST_SEASON].copy()  # never evaluated here

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  ← iterate against this")
print(f"Test season:  {TEST_SEASON} ← locked away, not evaluated here")
print(f"Mean n_pa — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

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


# ── 5. Train models, evaluate on val ─────────────────────────────────────────
results = {}


def _eval(name, y_true, y_pred):
    results[name] = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "pred": y_pred,
    }


print("\nEvaluating naive (global mean)...")
naive_global = DummyRegressor(strategy="mean")
naive_global.fit(Xtr, y_train)
_eval("Naive (global mean)", y_val, naive_global.predict(Xval))

print("Evaluating naive (batter's own rolling avg)...")
# Falls back to the train-set global mean wherever the batter has no prior
# games this season (first game of the year) — same fallback shape as
# build_expected_start_innings' baseline chain.
naive_rolling_pred = X_val[NAIVE_ROLLING_COL].fillna(y_train.mean()) if NAIVE_ROLLING_COL in X_val else pd.Series(
    y_train.mean(), index=X_val.index
)
_eval("Naive (rolling avg)", y_val, naive_rolling_pred)

print("Training linear regression...")
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xval_sc = scaler.transform(Xval)
lr = LinearRegression()
lr.fit(Xtr_sc, y_train)
_eval("Linear regression", y_val, lr.predict(Xval_sc))

print("Training XGBoost...")
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(Xtr, y_train)
_eval("XGBoost", y_val, xgb_model.predict(Xval))


# ── 6. Print results ──────────────────────────────────────────────────────────
naive_mae = results["Naive (global mean)"]["mae"]
rolling_mae = results["Naive (rolling avg)"]["mae"]

print(f"\n{'='*64}")
print(f"BASELINE RESULTS — {MODEL_NAME}")
print(f"Evaluated on val season {VAL_SEASON} ({len(val_df):,} batter-games)  |  Test season {TEST_SEASON} locked")
print("Primary: MAE (lower=better)  |  Secondary: RMSE (lower=better)")
print("=" * 64)
print(f"{'Model':<24} {'MAE':>8} {'vs Naive':>10}  {'RMSE':>8}")
print("-" * 64)
for name, res in results.items():
    delta = f"{res['mae'] - naive_mae:+.4f}" if name != "Naive (global mean)" else "—"
    print(f"{name:<24} {res['mae']:>8.4f} {delta:>10}  {res['rmse']:>8.4f}")
print("=" * 64)

print("\nInterpretation:")
# The real floor is the BETTER of the two naives, not just the rolling one —
# a model that beats the rolling avg but loses to the dumber global mean has
# not demonstrated signal.
best_naive_name, best_naive_mae = min(
    ((n, r["mae"]) for n, r in results.items() if n in ("Naive (global mean)", "Naive (rolling avg)")),
    key=lambda t: t[1],
)
candidates = {n: r for n, r in results.items() if n not in ("Naive (global mean)", "Naive (rolling avg)")}
beats_floor = {n: r for n, r in candidates.items() if r["mae"] < best_naive_mae}

if not beats_floor:
    print(f"  No model beats the best naive floor ({best_naive_name}, MAE {best_naive_mae:.4f}).")
    print("  Current feature set (batting_order, home/away, opposing starter's")
    print("  expected_start_innings + season WHIP, batter's own rolling PA rate, team win_pct) has")
    print("  no demonstrated signal beyond the constant-mean prediction — do not carry")
    print("  this feature set into train.py as-is. Investigate: is n_pa's real")
    print("  variance mostly driven by within-game events (extra innings, blowouts,")
    print("  injury subs) that no pre-game feature can see, before adding more features.")
else:
    best_name, best = min(beats_floor.items(), key=lambda t: t[1]["mae"])
    print(f"  {best_name} beats the best naive floor ({best_naive_name}, MAE {best_naive_mae:.4f}) —")
    print(f"  {best_name} MAE {best['mae']:.4f}. Worth carrying forward into train.py.")
print("=" * 64)


# ── 7. Plots ───────────────────────────────────────────────────────────────
PLOT_DIR = BASE_DIR / "plots" / "baseline-model"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(y_train, bins=range(0, int(y_train.max()) + 2), color="steelblue", align="left")
ax.set_title(f"n_pa distribution — train ({FIT_SEASONS[0]}–{FIT_SEASONS[-1]})")
ax.set_xlabel("Plate appearances")
ax.set_ylabel("Batter-games")
plt.tight_layout()
plt.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'target_distribution.png'}")

fig, ax = plt.subplots(figsize=(6, 6))
xgb_pred = results["XGBoost"]["pred"]
ax.scatter(y_val, xgb_pred, alpha=0.15, s=10, color="steelblue")
lims = [0, max(y_val.max(), xgb_pred.max()) + 1]
ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
ax.set_xlabel("Actual n_pa")
ax.set_ylabel("Predicted n_pa")
ax.set_title(f"XGBoost predicted vs actual — val ({VAL_SEASON})")
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "residuals.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'residuals.png'}")

feature_names = num_cols + cat_cols
importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
top = importances.tail(20)
fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.35)))
ax.barh(top.index, top.values, color="steelblue")
ax.set_title("Feature importance — XGBoost")
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
    "**Task:** Regression (batter-game grain)  ",
    "**Target:** How many plate appearances will this batter get in this game?  ",
    "**Primary metric:** MAE (lower = better)  ",
    "**Diagnostics:** RMSE, predicted-vs-actual scatter  ",
    f"**Data:** s3://{BUCKET}  ",
    "",
    "## Split",
    "",
    "| Split | Seasons | Rows | Mean n_pa |",
    "|-------|---------|------|-----------|",
    f"| Train | {FIT_SEASONS} | {len(train_df):,} | {y_train.mean():.3f} |",
    f"| Val   | {VAL_SEASON} | {len(val_df):,} | {y_val.mean():.3f} |",
    f"| Test  | {TEST_SEASON} | {len(test_df):,} | locked — not evaluated here |",
    "",
    "## Results (evaluated on val)",
    "",
    "Two naive floors are reported: global mean, and the batter's own trailing",
    "rolling avg_n_pa_per_game. The real bar a model must clear is the BETTER of",
    "the two — not just the global mean (see Interpretation below for which one",
    "won this run).",
    "",
    "| Model | MAE | Δ vs global naive | RMSE |",
    "|-------|-----|--------------------|------|",
]
for name, res in results.items():
    delta = f"{res['mae'] - naive_mae:+.4f}" if name != "Naive (global mean)" else "—"
    md_lines.append(f"| {name} | {res['mae']:.4f} | {delta} | {res['rmse']:.4f} |")

if not beats_floor:
    interpretation = (
        f"No model beats the best naive floor (**{best_naive_name}**, MAE {best_naive_mae:.4f}). "
        "The current feature set (batting_order, home/away, opposing starter's expected_start_innings + "
        "season WHIP, batter's own rolling PA rate, team win_pct) has no demonstrated signal beyond the constant-mean "
        "prediction — do not carry this feature set into train.py as-is. n_pa's real per-game variance "
        "may be dominated by within-game events (extra innings, blowouts, injury subs) that no pre-game "
        "feature can see; investigate that before adding more features."
    )
else:
    best_name, best = min(beats_floor.items(), key=lambda t: t[1]["mae"])
    interpretation = (
        f"**{best_name}** beats the best naive floor (**{best_naive_name}**, MAE {best_naive_mae:.4f}) "
        f"with MAE {best['mae']:.4f} — worth carrying forward into train.py."
    )

md_lines += [
    "",
    "## Interpretation",
    "",
    interpretation,
    "",
    "## Setup",
    "",
    f"- Features: {FEATURE_COLS}",
    "- Scope: starters only (has a batting_order) — subs/pinch-hitters excluded",
    "- Label: build_n_pa_label, filtered to PBP.PA_OUTCOMES (not boxscore's ab+bb, which undercounts)",
    "",
    "## Plots",
    "",
    "- `plots/baseline-model/target_distribution.png`",
    "- `plots/baseline-model/residuals.png` — predicted vs actual, XGBoost",
    "- `plots/baseline-model/feature_importance.png`",
    "",
    "## Next steps",
    "",
    "- If a model beats the best naive floor: move to train.py, consider more",
    "  opponent-pitcher signal (bullpen quality/rest), team implied run total.",
    "- If not: this feature set isn't ready for train.py. Investigate whether n_pa",
    "  variance is dominated by within-game events pre-game features can't see,",
    "  before adding more features — do not use an unproven model as a",
    "  hit_predictor feature, and prefer the global mean over a losing rolling-avg",
    "  baseline if one must be picked today.",
    "- Final evaluation on test season (2025) only once in train.py.",
]

with open(BASE_DIR / "baseline_results.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"\nSaved {BASE_DIR / 'baseline_results.md'}")
