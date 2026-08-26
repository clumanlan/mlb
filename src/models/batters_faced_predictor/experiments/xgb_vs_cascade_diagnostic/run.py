"""
Diagnostic (not a versioned experiment, same convention as
k_predictor/experiments/count_distribution_check/run.py): why does the
expected_batters_faced shrinkage cascade beat XGBoost fed the exact same
inputs? Answers four questions on real 2024 val-season data:

  1. Is default-hyperparameter XGBoost overfitting? (train vs val MAE gap)
  2. Does tuning close the gap? (shallow trees, low learning rate, early
     stopping, regularization, subsample/colsample)
  3. Is the cascade-vs-XGBoost MAE gap distinguishable from sampling noise?
     (bootstrap CI on the val set, 4,786 starts)
  4. What do the biggest cascade-vs-XGBoost disagreements actually look
     like? (worked examples, real features)

Run from src/models/batters_faced_predictor/ with:
  python experiments/xgb_vs_cascade_diagnostic/run.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context

import models.batters_faced_predictor.processing.pipeline as pipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

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
# Internal early-stopping validation set: hold out the most recent fit
# season (2023) from XGBoost's own training rows so early stopping has a
# real held-out signal, same "most recent slice as dev set" convention
# time-based splits normally use. NOT the same as VAL_SEASON (2024) — that
# stays untouched by any fitting decision, including early-stopping.
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
print("Loading player info...")
player_info = wr.s3.read_parquet(path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session)

print("\nBuilding start-grain DataFrame...")
schedule = hp_pipeline.process_schedule(schedule)
pbp = hp_pipeline.build_pbp_features(pbp, schedule, player_info)
pbp = pbp.merge(schedule[["gamepk", "game_datetime"]], on="gamepk", how="left")

start_outcome = pipeline.create_start_pa_outcome(pbp)

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

pitcher_hand = pbp[["pitcher_id", "pitcher_throw_hand"]].drop_duplicates(subset=["pitcher_id"]).rename(
    columns={"pitcher_id": "personId"}
)
start_outcome = start_outcome.merge(pitcher_hand, on="personId", how="left")

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
CASCADE_ONLY_COLS = ["expected_batters_faced", "expected_batters_faced_weight"]

model_df = start_outcome[["personId", "gamepk"] + FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].dropna(
    subset=[TARGET, "expected_batters_faced"]
).copy()
# expected_batters_faced can still be NaN for a true first-ever start with no
# league fallback yet for that season (documented in
# build_batters_faced_residual_bins' own docstring, game_context.py) — rare,
# concentrated in the earliest fit season(s). Dropped here, same precedent.
model_df["game_season"] = model_df["game_season"].astype(int)

train_df  = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
core_df   = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_df  = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()
val_df    = model_df[model_df["game_season"] == VAL_SEASON].copy()

print(f"\nCore fit seasons (train): {CORE_FIT_SEASONS} ({len(core_df):,} rows)")
print(f"Early-stopping season:    {EARLY_STOP_SEASON} ({len(early_df):,} rows)")
print(f"Full fit seasons (default-XGB train): {FIT_SEASONS} ({len(train_df):,} rows)")
print(f"Val season:               {VAL_SEASON} ({len(val_df):,} rows)")


def encode(X_tr, X_evs, cat_cols, num_cols):
    """Fit imputers/encoder on X_tr, transform X_tr plus any number of eval frames."""
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


num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(train_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

# ── Encodings for each training regime ────────────────────────────────────────
# (a) full FIT_SEASONS train -> val (matches baseline/model/run.py exactly)
Xtr_full, [Xval_full] = encode(train_df[FEATURE_COLS], [val_df[FEATURE_COLS]], cat_cols, num_cols)
# (b) core (excl. early-stop season) train -> early-stop val, val
Xtr_core, [Xearly, Xval_core] = encode(
    core_df[FEATURE_COLS], [early_df[FEATURE_COLS], val_df[FEATURE_COLS]], cat_cols, num_cols
)

y_tr_full, y_val = train_df[TARGET], val_df[TARGET]
y_tr_core, y_early = core_df[TARGET], early_df[TARGET]


def mae_rmse(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))


results = {}

# ── 1. Reproduce the baseline exactly (sanity check) ──────────────────────────
print(f"\n{'=' * 78}\n1. REPRODUCING BASELINE NUMBERS (sanity check)\n{'=' * 78}")

cascade_train_mae, cascade_train_rmse = mae_rmse(y_tr_full, train_df["expected_batters_faced"])
cascade_val_mae, cascade_val_rmse = mae_rmse(y_val, val_df["expected_batters_faced"])
print(f"Cascade            train MAE {cascade_train_mae:.4f}  val MAE {cascade_val_mae:.4f}")

lr = LinearRegression()
scaler = StandardScaler()
lr.fit(scaler.fit_transform(Xtr_full), y_tr_full)
lr_train_pred = lr.predict(scaler.transform(Xtr_full))
lr_val_pred = lr.predict(scaler.transform(Xval_full))
lr_train_mae, _ = mae_rmse(y_tr_full, lr_train_pred)
lr_val_mae, _ = mae_rmse(y_val, lr_val_pred)
print(f"LinearRegression   train MAE {lr_train_mae:.4f}  val MAE {lr_val_mae:.4f}")

xgb_default = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_default.fit(Xtr_full, y_tr_full)
xgb_def_train_pred = xgb_default.predict(Xtr_full)
xgb_def_val_pred = xgb_default.predict(Xval_full)
xgb_def_train_mae, _ = mae_rmse(y_tr_full, xgb_def_train_pred)
xgb_def_val_mae, _ = mae_rmse(y_val, xgb_def_val_pred)
print(f"XGBoost (default)  train MAE {xgb_def_train_mae:.4f}  val MAE {xgb_def_val_mae:.4f}"
      f"   <-- train/val gap: {xgb_def_val_mae - xgb_def_train_mae:+.4f}")
print(f"(for reference, LR train/val gap: {lr_val_mae - lr_train_mae:+.4f}, cascade has no 'train' fit)")
results["Cascade"] = {"val_pred": val_df["expected_batters_faced"].to_numpy(), "val_mae": cascade_val_mae}
results["Linear regression"] = {"val_pred": lr_val_pred, "val_mae": lr_val_mae}
results["XGBoost (default)"] = {"val_pred": xgb_def_val_pred, "val_mae": xgb_def_val_mae}

# ── 2. Does tuning close the gap? ──────────────────────────────────────────────
print(f"\n{'=' * 78}\n2. TUNED XGBOOST (shallow trees, low LR, early stopping, regularization)\n{'=' * 78}")

xgb_tuned = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.02, max_depth=3, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5,
    random_state=42, verbosity=0, early_stopping_rounds=50, eval_metric="mae",
)
xgb_tuned.fit(Xtr_core, y_tr_core, eval_set=[(Xearly, y_early)], verbose=False)
print(f"Best iteration (early stopping on {EARLY_STOP_SEASON}): {xgb_tuned.best_iteration}")

xgb_tuned_train_pred = xgb_tuned.predict(Xtr_core)
xgb_tuned_val_pred = xgb_tuned.predict(Xval_core)
xgb_tuned_train_mae, _ = mae_rmse(y_tr_core, xgb_tuned_train_pred)
xgb_tuned_val_mae, xgb_tuned_val_rmse = mae_rmse(y_val, xgb_tuned_val_pred)
print(f"XGBoost (tuned)    train MAE {xgb_tuned_train_mae:.4f}  val MAE {xgb_tuned_val_mae:.4f}"
      f"   <-- train/val gap: {xgb_tuned_val_mae - xgb_tuned_train_mae:+.4f}")
print(f"vs. cascade val MAE {cascade_val_mae:.4f}: {xgb_tuned_val_mae - cascade_val_mae:+.4f}")
print(f"vs. XGBoost default val MAE {xgb_def_val_mae:.4f}: {xgb_tuned_val_mae - xgb_def_val_mae:+.4f}")
results["XGBoost (tuned)"] = {"val_pred": xgb_tuned_val_pred, "val_mae": xgb_tuned_val_mae}

# ── 3. Cascade-only features (2 cols) — does XGBoost need the raw components at all? ──
print(f"\n{'=' * 78}\n3. XGBOOST GIVEN ONLY THE CASCADE'S OWN OUTPUT (2 features: point estimate + weight)\n{'=' * 78}")
print("If this ~matches the cascade, XGBoost CAN use the pre-computed interaction fine —")
print("it's rediscovering it from raw components that's the hard part.")

Xtr_co, [Xval_co] = encode(train_df[CASCADE_ONLY_COLS], [val_df[CASCADE_ONLY_COLS]], [], CASCADE_ONLY_COLS)
xgb_cascade_only = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=10,
    random_state=42, verbosity=0,
)
xgb_cascade_only.fit(Xtr_co, y_tr_full)
co_val_pred = xgb_cascade_only.predict(Xval_co)
co_val_mae, _ = mae_rmse(y_val, co_val_pred)
print(f"XGBoost (cascade-only features) val MAE {co_val_mae:.4f}  (cascade itself: {cascade_val_mae:.4f})")
results["XGBoost (cascade-only feats)"] = {"val_pred": co_val_pred, "val_mae": co_val_mae}

# ── 4. Raw components ONLY (no expected_batters_faced/_weight at all) ─────────
print(f"\n{'=' * 78}\n4. XGBOOST GIVEN ONLY THE RAW COMPONENTS (no pre-computed interaction at all)\n{'=' * 78}")
raw_cols = [c for c in FEATURE_COLS if c not in CASCADE_ONLY_COLS]
Xtr_raw, [Xval_raw] = encode(
    train_df[raw_cols], [val_df[raw_cols]],
    [c for c in raw_cols if c not in num_cols], [c for c in raw_cols if c in num_cols],
)
xgb_raw_only = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=10,
    random_state=42, verbosity=0,
)
xgb_raw_only.fit(Xtr_raw, y_tr_full)
raw_val_pred = xgb_raw_only.predict(Xval_raw)
raw_val_mae, _ = mae_rmse(y_val, raw_val_pred)
print(f"XGBoost (raw components only, must rediscover the shrinkage interaction) val MAE {raw_val_mae:.4f}")
results["XGBoost (raw components only)"] = {"val_pred": raw_val_pred, "val_mae": raw_val_mae}

# ── 5. Bootstrap CI — is the gap distinguishable from sampling noise? ─────────
print(f"\n{'=' * 78}\n5. BOOTSTRAP CI (2,000 resamples of the {len(val_df):,}-start val set)\n{'=' * 78}")
rng = np.random.default_rng(42)
y_val_arr = y_val.to_numpy()
n = len(y_val_arr)
N_BOOT = 2000

boot_deltas = {}
for name in ["Linear regression", "XGBoost (default)", "XGBoost (tuned)"]:
    cascade_pred = results["Cascade"]["val_pred"]
    model_pred = results[name]["val_pred"]
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        cascade_mae_b = np.abs(y_val_arr[idx] - cascade_pred[idx]).mean()
        model_mae_b = np.abs(y_val_arr[idx] - model_pred[idx]).mean()
        deltas[i] = model_mae_b - cascade_mae_b
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    boot_deltas[name] = (lo, hi, deltas.mean())
    sig = "does NOT cross 0 -> gap is real, not noise" if lo > 0 or hi < 0 else "CROSSES 0 -> can't rule out noise at 95%"
    print(f"{name:<28} MAE delta vs cascade: mean {deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({sig})")

# ── 6. Worked examples — biggest cascade vs. XGBoost(default) disagreements ───
print(f"\n{'=' * 78}\n6. WORKED EXAMPLES — biggest |cascade - XGBoost(default)| disagreements\n{'=' * 78}")
examples = val_df[["personId", "gamepk", TARGET] + FEATURE_COLS].copy()
examples["cascade_pred"] = results["Cascade"]["val_pred"]
examples["xgb_default_pred"] = results["XGBoost (default)"]["val_pred"]
examples["disagreement"] = (examples["cascade_pred"] - examples["xgb_default_pred"]).abs()
examples["cascade_err"] = (examples[TARGET] - examples["cascade_pred"]).abs()
examples["xgb_err"] = (examples[TARGET] - examples["xgb_default_pred"]).abs()
top = examples.sort_values("disagreement", ascending=False).head(10)
cols_to_show = [
    "personId", "gamepk", TARGET, "cascade_pred", "xgb_default_pred", "disagreement",
    "cascade_err", "xgb_err", "expected_batters_faced_weight",
    "pitcher_this_season_start_pa_starts_n", "pitcher_last_season_start_pa_avg_pa_per_start",
]
print(top[cols_to_show].round(2).to_string(index=False))
print(f"\nOf these 10 biggest disagreements: cascade was closer to realized in "
      f"{(top['cascade_err'] < top['xgb_err']).sum()}/10 cases.")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
print(f"{'Model':<32} {'Val MAE':>10} {'vs cascade':>12}")
for name, res in results.items():
    delta = "—" if name == "Cascade" else f"{res['val_mae'] - cascade_val_mae:+.4f}"
    print(f"{name:<32} {res['val_mae']:>10.4f} {delta:>12}")
