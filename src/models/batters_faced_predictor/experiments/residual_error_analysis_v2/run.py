"""
Diagnostic (not a versioned experiment, same convention as
experiments/residual_error_analysis/run.py, xgb_vs_cascade_diagnostic/run.py,
and k_predictor/experiments/count_distribution_check/run.py): "look at where
the model is worst" error analysis, this time on v2's model (the current
strongest, tuned XGBoost MAE 2.6471) instead of v1's — the original
residual_error_analysis/run.py predates v2/v3 and never got re-run against the
model that's actually the standing candidate.

Trains the exact same model (same features, same hyperparameters, same
train/early-stop/val split) as experiments/v2_workload_and_rest/train.py, then
pulls the val-season (2024) predictions apart:

  1. Worst OVER-predictions and worst UNDER-predictions (top 25 each),
     enriched with context the model does NOT currently see: actual boxscore
     line, weather, game duration, doubleheader flag, score margin, opponent,
     venue, decision.
  2. Residual broken out by categorical/derived buckets not yet in
     FEATURE_COLS (weather_condition, month, day_of_week, doubleheader game,
     blowout margin, venue, decision, opponent team) to check for a
     systematic (not just noisy) miss.
  3. The cold-start bucket (0-2 starts) is shown for completeness but called
     out separately — v3 already investigated it (multi-season lookback,
     flat result, genuine small-sample-variance ceiling) — the point of this
     pass is to find a DIFFERENT lever, so worked examples below are also
     shown with cold-start pitchers excluded.

Goal: read the worked examples and bucket table to find a candidate feature
hypothesis for a v4, not to ship anything here. Any feature that looks
promising goes through impl-planning/TDD in a real experiment, per CLAUDE.md.

Run from src/models/batters_faced_predictor/ with:
  python experiments/residual_error_analysis_v2/run.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.processing.features import rolling_stats

import models.batters_faced_predictor.processing.pipeline as pipeline

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

STAGE = Path(__file__).parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PA_SHRINKAGE_K = 5.0

with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET, REGION = cfg["bucket"], cfg["region"]
TRAIN_SEASONS, FEATURE_SEASONS = cfg["train_seasons"], cfg["feature_seasons"]
TARGET, DATE_COL = cfg["target_column"], cfg["date_column"]
TEST_SEASON, VAL_SEASON = cfg["test_season"], cfg["val_season"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)
EARLY_STOP_SEASON = 2023
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]

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


print("Loading play-by-play...")
pbp = read_parquet_seasons("s3://{bucket}/processed_data/prepared/playbyplay/{season}/", TRAIN_SEASONS, chunked=True)
print("Loading schedule...")
schedule = read_parquet_seasons("s3://{bucket}/processed_data/games/schedule/{season}/", all_boxscore_seasons)
print("Loading batter boxscore...")
batter_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", all_boxscore_seasons)
print("Loading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", TRAIN_SEASONS)
pitcher_boxscore_val = pitcher_boxscore[pitcher_boxscore["game_season"] == VAL_SEASON].copy()
print("Loading game_info (val season only, for worked-example context)...")
game_info_val = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/processed_data/games/game_info/{VAL_SEASON}/", boto3_session=boto_session,
)
print("Loading player info...")
player_info = wr.s3.read_parquet(path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session)

print("\nBuilding start-grain DataFrame (v2 feature set)...")
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

# ------------------------- 1. OPPOSING-LINEUP ON-BASE/WALK RATE (v1) --------- #
home_away = schedule[["gamepk", "home_id", "away_id"]].drop_duplicates("gamepk").assign(
    gamepk=lambda x: x["gamepk"].astype(str), home_id=lambda x: x["home_id"].astype(str),
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

# ------------------------- 2. TEAM REST DAYS (v1, pitcher's own team) -------- #
team_rest_days = game_context.build_team_rest_days(schedule)[["team_id", "gamepk", "team_days_since_last_game"]]
team_rest_days = team_rest_days.assign(
    team_id=lambda x: x["team_id"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
).rename(columns={"team_id": "pitcher_team_id", "team_days_since_last_game": "pitcher_team_days_since_last_game"})
start_outcome = start_outcome.merge(team_rest_days, on=["pitcher_team_id", "gamepk"], how="left")

# ------------------------- 3. PITCH-EFFICIENCY PROXY (v1) -------------------- #
sp_pbp = pbp[pbp["pitcher_role"] == "sp"]
pitcher_pitch_efficiency = rolling_stats.build_pbp_pitcher_rolling_feats(
    sp_pbp, window="season", pitcher_role="sp", entity_col="pitcher_id",
)[["pitcher_id", "gamepk", "pitcher_roll_season_pitch_count_avg"]].rename(columns={"pitcher_id": "personId"})
pitcher_pitch_efficiency = pitcher_pitch_efficiency.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)
start_outcome = start_outcome.merge(pitcher_pitch_efficiency, on=["personId", "gamepk"], how="left")

# ------------------------- 4. HANDEDNESS (baseline feature) ------------------ #
pitcher_hand = pbp[["pitcher_id", "pitcher_throw_hand"]].drop_duplicates(subset=["pitcher_id"]).rename(
    columns={"pitcher_id": "personId"}
)
start_outcome = start_outcome.merge(pitcher_hand, on="personId", how="left")

# ------------------------- 5. TRAILING-3-START PA TREND (v2) ----------------- #
pitcher_start_pa_last3 = game_context.build_pitcher_start_pa_this_season(pbp, window=3)[
    ["personId", "gamepk", "pitcher_last3_start_pa_avg_pa_per_start", "pitcher_last3_start_pa_starts_n"]
].assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str))
start_outcome = start_outcome.merge(pitcher_start_pa_last3, on=["personId", "gamepk"], how="left")

start_outcome["pa_trend_ratio"] = start_outcome["pitcher_last3_start_pa_avg_pa_per_start"] / (
    start_outcome["pitcher_this_season_start_pa_avg_pa_per_start"].replace(0, np.nan)
)
start_outcome["pa_trend_direction"] = np.sign(
    start_outcome["pitcher_last3_start_pa_avg_pa_per_start"] - start_outcome["pitcher_this_season_start_pa_avg_pa_per_start"]
)

# ------------------------- 6. PITCHER'S OWN REST DAYS (v2) ------------------- #
rest_days_raw = game_context.build_pitcher_rest_days(pbp)
rest_days_for_merge = rest_days_raw[["personId", "gamepk", "pitcher_days_since_last_start"]].assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)
start_outcome = start_outcome.merge(rest_days_for_merge, on=["personId", "gamepk"], how="left")

# ------------------------- 7. WORKLOAD DENSITY (v2) -------------------------- #
workload = game_context.build_pitcher_workload_density(pitcher_boxscore, pbp, rest_days_raw)[
    ["personId", "gamepk", "pitcher_last_start_pitches", "pitcher_workload_density"]
].assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str))
start_outcome = start_outcome.merge(workload, on=["personId", "gamepk"], how="left")

starts_n_safe = start_outcome["pitcher_this_season_start_pa_starts_n"].fillna(0)
start_outcome["pitcher_workload_density_shrunk"] = start_outcome["pitcher_workload_density"] * (
    starts_n_safe / (starts_n_safe + PA_SHRINKAGE_K)
)

FEATURE_COLS = [
    "pitcher_last_season_start_pa_avg_pa_per_start", "pitcher_last_season_start_pa_n_starts",
    "pitcher_this_season_start_pa_avg_pa_per_start", "pitcher_this_season_start_pa_starts_n",
    "team_last_season_avg_pa_per_start", "league_last_season_avg_pa_per_start",
    "expected_batters_faced", "expected_batters_faced_weight", "pitcher_throw_hand",
    "team_roll_season_walk_rate", "team_roll_season_on_base_rate",
    "pitcher_team_days_since_last_game", "is_home", "pitcher_roll_season_pitch_count_avg",
    "pitcher_last3_start_pa_avg_pa_per_start", "pitcher_last3_start_pa_starts_n",
    "pa_trend_ratio", "pa_trend_direction", "pitcher_days_since_last_start",
    "pitcher_last_start_pitches", "pitcher_workload_density", "pitcher_workload_density_shrunk",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in start_outcome.columns]

model_df = start_outcome[
    ["personId", "gamepk", "pitcher_team_id"] + FEATURE_COLS + [TARGET, DATE_COL, "game_season"]
].dropna(subset=[TARGET]).copy()
model_df["game_season"] = model_df["game_season"].astype(int)

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()
core_df  = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_df = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()

print(f"\nCore fit seasons: {CORE_FIT_SEASONS} ({len(core_df):,} rows)")
print(f"Early-stop season: {EARLY_STOP_SEASON} ({len(early_df):,} rows)")
print(f"Val season: {VAL_SEASON} ({len(val_df):,} rows)")

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(train_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]


def encode_multi(X_tr, X_evs, cat_cols, num_cols):
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


Xcore, [Xearly, Xval] = encode_multi(core_df[FEATURE_COLS], [early_df[FEATURE_COLS], val_df[FEATURE_COLS]], cat_cols, num_cols)
y_core, y_early, y_val = core_df[TARGET], early_df[TARGET], val_df[TARGET]

print("\nTraining tuned XGBoost (same hyperparameters as v2_workload_and_rest)...")
xgb_tuned = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.02, max_depth=3, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5,
    random_state=42, verbosity=0, early_stopping_rounds=50, eval_metric="mae",
)
xgb_tuned.fit(Xcore, y_core, eval_set=[(Xearly, y_early)], verbose=False)
print(f"Best iteration: {xgb_tuned.best_iteration}")

val_df = val_df.copy()
val_df["predicted"] = xgb_tuned.predict(Xval)
val_df["residual"] = val_df[TARGET] - val_df["predicted"]          # + = model UNDER-predicted, - = OVER-predicted
val_df["abs_residual"] = val_df["residual"].abs()

mae = val_df["abs_residual"].mean()
print(f"\nSanity check — val MAE {mae:.4f} (should match v2_results.md's XGBoost (v2, tuned) row, 2.6471)")

# ── Enrich val predictions with context NOT in FEATURE_COLS, for worked examples ──
schedule_ctx = schedule[[
    "gamepk", "home_id", "away_id", "home_name", "away_name", "game_num",
    "home_score", "away_score", "game_datetime", "venue_name",
    "winning_pitcher", "losing_pitcher",
]].drop_duplicates("gamepk").assign(gamepk=lambda x: x["gamepk"].astype(str))

pitcher_boxscore_val = pitcher_boxscore_val.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)[["personId", "gamepk", "ip", "h", "r", "er", "bb", "k", "hr", "p", "s", "outcome", "player_name"]]

game_info_val = game_info_val.assign(gamepk=lambda x: x["gamepk"].astype(str))[
    ["gamepk", "weather_condition", "weather_temp", "weather_wind", "game_duration_minutes"]
]

examples = val_df.merge(schedule_ctx, on="gamepk", how="left")
examples = examples.merge(pitcher_boxscore_val, on=["personId", "gamepk"], how="left")
examples = examples.merge(game_info_val, on="gamepk", how="left")

examples["opp_team_name"] = np.where(
    examples["pitcher_team_id"].astype(str) == examples["home_id"].astype(str),
    examples["away_name"], examples["home_name"],
)
examples["pitcher_team_score"] = np.where(
    examples["pitcher_team_id"].astype(str) == examples["home_id"].astype(str),
    examples["home_score"], examples["away_score"],
)
examples["opp_team_score"] = np.where(
    examples["pitcher_team_id"].astype(str) == examples["home_id"].astype(str),
    examples["away_score"], examples["home_score"],
)
examples["score_margin"] = (examples["pitcher_team_score"] - examples["opp_team_score"]).abs()
examples["is_doubleheader_g2"] = (examples["game_num"] == 2).astype(int)
examples["month"] = pd.to_datetime(examples["game_datetime"]).dt.month
examples["day_of_week"] = pd.to_datetime(examples["game_datetime"]).dt.day_name()
examples["decision"] = np.select(
    [examples["personId"] == examples["winning_pitcher"].astype(str),
     examples["personId"] == examples["losing_pitcher"].astype(str)],
    ["win", "loss"], default="no_decision",
)
examples["is_cold_start"] = (examples["pitcher_this_season_start_pa_starts_n"] <= 2)

WORKED_EXAMPLE_COLS = [
    "player_name", "personId", "gamepk", TARGET, "predicted", "residual",
    "expected_batters_faced", "expected_batters_faced_weight",
    "pitcher_this_season_start_pa_starts_n", "pitcher_last3_start_pa_avg_pa_per_start",
    "pa_trend_direction", "pitcher_days_since_last_start", "pitcher_workload_density",
    "ip", "h", "r", "er", "bb", "k", "hr", "p", "s", "outcome", "decision",
    "is_home", "opp_team_name", "venue_name", "score_margin", "is_doubleheader_g2",
    "weather_condition", "weather_temp", "game_duration_minutes", "month", "day_of_week",
]
WORKED_EXAMPLE_COLS = [c for c in WORKED_EXAMPLE_COLS if c in examples.columns]

print(f"\n{'=' * 100}\nTOP 25 UNDER-PREDICTIONS, ALL PITCHERS (model said LOW, pitcher actually faced MANY)\n{'=' * 100}")
under_all = examples.sort_values("residual", ascending=False).head(25)
print(under_all[WORKED_EXAMPLE_COLS].round(2).to_string(index=False))

print(f"\n{'=' * 100}\nTOP 25 OVER-PREDICTIONS, ALL PITCHERS (model said HIGH, pitcher actually faced FEW)\n{'=' * 100}")
over_all = examples.sort_values("residual", ascending=True).head(25)
print(over_all[WORKED_EXAMPLE_COLS].round(2).to_string(index=False))

# ── Excluding cold-start (0-2 starts) pitchers — the specific ask: find a
# DIFFERENT lever than v3 already investigated ──
established = examples[~examples["is_cold_start"]]
print(f"\n{'=' * 100}\nTOP 25 UNDER-PREDICTIONS, EXCLUDING COLD-START (3+ starts this season, n={len(established):,})\n{'=' * 100}")
under_est = established.sort_values("residual", ascending=False).head(25)
print(under_est[WORKED_EXAMPLE_COLS].round(2).to_string(index=False))

print(f"\n{'=' * 100}\nTOP 25 OVER-PREDICTIONS, EXCLUDING COLD-START (3+ starts this season, n={len(established):,})\n{'=' * 100}")
over_est = established.sort_values("residual", ascending=True).head(25)
print(over_est[WORKED_EXAMPLE_COLS].round(2).to_string(index=False))

# ── Bucketed mean residual/MAE by categorical/derived cuts not in FEATURE_COLS ──
print(f"\n{'=' * 100}\nMEAN RESIDUAL / MAE BY BUCKET, EXCLUDING COLD-START (cuts NOT currently in FEATURE_COLS)\n{'=' * 100}")


def bucket_report(df, col, label=None):
    label = label or col
    g = df.groupby(col, observed=True)["residual"].agg(["mean", lambda s: s.abs().mean(), "count"])
    g.columns = ["mean_residual (bias)", "MAE", "n"]
    g = g.sort_values("MAE", ascending=False)
    print(f"\n--- {label} ---")
    print(g.round(3).to_string())
    return g


bucket_reports = {}
bucket_reports["weather_condition"] = bucket_report(established, "weather_condition")
bucket_reports["month"] = bucket_report(established, "month")
bucket_reports["day_of_week"] = bucket_report(established, "day_of_week")
bucket_reports["is_doubleheader_g2"] = bucket_report(established, "is_doubleheader_g2")
bucket_reports["decision"] = bucket_report(established, "decision")
bucket_reports["venue_name"] = bucket_report(established, "venue_name")

established = established.copy()
established["score_margin_bucket"] = pd.cut(
    established["score_margin"], [-0.1, 1, 3, 6, 100], labels=["<=1 (close)", "2-3", "4-6", "7+ (blowout)"],
)
bucket_reports["score_margin_bucket"] = bucket_report(established, "score_margin_bucket", "final score margin")

established["opp_onbase_bucket"] = pd.qcut(
    established["team_roll_season_on_base_rate"], 4,
    labels=["Q1 (weakest lineup)", "Q2", "Q3", "Q4 (strongest lineup)"], duplicates="drop",
)
bucket_reports["opp_onbase_bucket"] = bucket_report(established, "opp_onbase_bucket", "opposing lineup on-base rate quartile")

established["rest_days_bucket"] = pd.cut(
    established["pitcher_days_since_last_start"], [-0.1, 4, 5, 6, 100],
    labels=["<=4 (short rest)", "5 (normal)", "6", "7+ (extra rest)"],
)
bucket_reports["rest_days_bucket"] = bucket_report(established, "rest_days_bucket", "pitcher's own rest days")

established["workload_density_bucket"] = pd.qcut(
    established["pitcher_workload_density"], 4,
    labels=["Q1 (lightest recent load)", "Q2", "Q3", "Q4 (heaviest recent load)"], duplicates="drop",
)
bucket_reports["workload_density_bucket"] = bucket_report(established, "workload_density_bucket", "workload density quartile")

bucket_reports["pa_trend_direction"] = bucket_report(established, "pa_trend_direction", "PA trend direction (recent vs season)")

# ── Cold-start bucket shown separately, for completeness only (already
# investigated by v3 — not the target of this pass) ──
print(f"\n{'=' * 100}\nCOLD-START BUCKET (0-2 starts), SHOWN FOR REFERENCE ONLY — already investigated by v3\n{'=' * 100}")
cold_mae = examples.loc[examples["is_cold_start"], "abs_residual"].mean()
est_mae = established["abs_residual"].mean()
print(f"n={examples['is_cold_start'].sum()}  cold-start MAE {cold_mae:.4f}  |  n={len(established):,}  established MAE {est_mae:.4f}")

# ── Plots ──────────────────────────────────────────────────────────────────
PLOT_DIR = STAGE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(established["residual"], bins=40, color="steelblue")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_title(f"Residual distribution, excl. cold-start (realized - predicted) — val {VAL_SEASON}")
ax.set_xlabel("residual (batters faced)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "residual_hist_excl_cold_start.png", dpi=120)
plt.close()
print(f"\nSaved {PLOT_DIR / 'residual_hist_excl_cold_start.png'}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(established["team_roll_season_on_base_rate"], established["residual"], alpha=0.3, s=12, color="steelblue")
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("opposing lineup rolling on-base rate")
ax.set_ylabel("residual (realized - predicted)")
ax.set_title("Residual vs. opposing lineup on-base rate, excl. cold-start")
plt.tight_layout()
plt.savefig(PLOT_DIR / "residual_vs_opp_onbase.png", dpi=120)
plt.close()
print(f"Saved {PLOT_DIR / 'residual_vs_opp_onbase.png'}")

# ── Save worked examples + bucket tables to markdown for review ─────────────
md_lines = [
    "# Residual error analysis v2 — batters_faced_predictor tuned XGBoost (v2 features)",
    "",
    f"Val season {VAL_SEASON}, MAE {mae:.4f} (sanity check vs. v2_results.md's 2.6471)",
    "",
    "Cold-start (0-2 starts) is already investigated (v3, flat result, genuine",
    "small-sample-variance ceiling) — this pass excludes it from the worked",
    "examples/buckets below to look for a DIFFERENT lever.",
    "",
    f"Cold-start MAE {cold_mae:.4f} (n={examples['is_cold_start'].sum()}) vs. "
    f"established MAE {est_mae:.4f} (n={len(established):,}), shown for reference only.",
    "",
    "## Top 25 under-predictions, excluding cold-start",
    "",
    under_est[WORKED_EXAMPLE_COLS].round(2).to_markdown(index=False),
    "",
    "## Top 25 over-predictions, excluding cold-start",
    "",
    over_est[WORKED_EXAMPLE_COLS].round(2).to_markdown(index=False),
    "",
]
for label, g in bucket_reports.items():
    md_lines += [f"## Bucket: {label}", "", g.round(3).to_markdown(), ""]

(STAGE / "error_analysis_v2.md").write_text("\n".join(md_lines) + "\n")
print(f"\nSaved {STAGE / 'error_analysis_v2.md'}")
