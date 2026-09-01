"""
k_predictor experiment: direct count model (architecture spike, not a v-numbered
feature pass -- see ROADMAP.md's "Near-term backlog (k_predictor)").
Run from src/models/k_predictor/ with: python experiments/direct_count_model/train.py
No AWS credentials needed -- reads the local v6 feature-set cache built by
count_distribution_check/within_game_correlation_check.py (see ROADMAP.md
2026-08-29: "any future k_predictor-v6-feature-set diagnostic should load
that cache instead of rebuilding from S3").
"""
# ---------------------------------------------------------------------------- #
#                            DIRECT COUNT MODEL SPIKE                          #
# ---------------------------------------------------------------------------- #
# v6 (the standing production candidate) never predicts total strikeouts
# directly -- it classifies each PA independently, then assembles a start's
# total as a Poisson-binomial sum of those PA-level probabilities. Two
# findings already in this project say that architecture may be leaving
# something on the table:
#
#   1. icc_check.py found a real, statistically bulletproof within-start
#      correlation between PA-level residuals (ICC=0.0100, p vanishingly
#      small) -- pitchers really do run hot or cold across a whole start,
#      beyond what pre-game features predict. The independent-slot assembly
#      structurally cannot use this; a model of the start's TOTAL can.
#   2. run_xgboost_uncertainty.py's worst misses are all large
#      under-predictions on unusually dominant starts (Luis Gil 14K vs. 4.7
#      predicted, etc.) -- exactly the failure mode a shared per-start
#      "quality today" signal would help with, if a model can learn it.
#
# This script tests the direct alternative: model total_strikeouts per SP
# start as a Poisson count target, on v6's own frozen feature set (aggregated
# to start grain), and compares it head-to-head against "sum of v6's PA
# classifier probabilities" (the assembled baseline -- mathematically v6's
# own point estimate, without the Poisson-binomial uncertainty machinery on
# top) on the same val-season starts.
#
# Time-boxed spike. The count model gets the same max_depth x learning_rate
# grid v6's own classifier tune used (9 configs, early-stopped) -- if the
# best of 9 still doesn't beat the assembled baseline, close the thread
# rather than sinking further tuning time into it. The higher-priority open
# question is still ROADMAP.md item 6(c) (score 2026 against the real
# market) -- this spike is secondary to that.
# ---------------------------------------------------------------------------- #
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

pd.set_option("display.max_columns", None)

CACHE_DIR = Path(__file__).resolve().parent.parent / "count_distribution_check" / "_model_cache"
TARGET = "is_strikeout"
GROUP_KEYS = ["gamepk", "pitcher_id"]

# Same split v6 / icc_check.py use -- keeps this spike directly comparable.
CORE_FIT_SEASONS = [2018, 2019, 2022]
EARLY_STOP_SEASON = 2023
VAL_SEASON = 2024
# TEST_SEASON (2025) stays untouched, same convention as every other version.

MIN_PA_PER_START = 10  # drop aborted/injury-shortened starts, same threshold icc_check.py uses

# These vary PA-to-PA within a start (opposing batter, times-through-order
# position) -- aggregate to start grain, don't take first(). Every other
# FEATURE_COLS entry is pitcher/team/weather-level and constant within a start.
BATTER_LEVEL_COLS = [
    "batter_last_season_pa_strikeout_rate",
    "batter_roll_season_pa_strikeout_rate",
    "expected_times_through_order",
]


# ── 1. Load v6's frozen PA-grain feature cache ──────────────────────────────
model_df = pd.read_parquet(CACHE_DIR / "model_df_v6.parquet")
with open(CACHE_DIR / "feature_cols_v6.json") as f:
    FEATURE_COLS = json.load(f)

start_sizes = model_df.groupby(GROUP_KEYS).size()
eligible_starts = start_sizes[start_sizes >= MIN_PA_PER_START].index
model_df = model_df.set_index(GROUP_KEYS)
model_df = model_df[model_df.index.isin(eligible_starts)].reset_index()


def season_split(seasons):
    return model_df[model_df["game_season"].isin(seasons)].copy()


core_pa = season_split(CORE_FIT_SEASONS)
early_pa = season_split([EARLY_STOP_SEASON])
val_pa = season_split([VAL_SEASON])


# ── 2. Assembled baseline: refit v6's exact PA-level classifier ────────────
num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(model_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]


def prep(df, cols):
    df = df[cols].copy()
    n_cols = [c for c in cols if c in num_cols]
    c_cols = [c for c in cols if c in cat_cols]
    if n_cols:
        df[n_cols] = df[n_cols].apply(pd.to_numeric, errors="coerce")
    if c_cols:
        df[c_cols] = df[c_cols].astype(object).fillna(np.nan)
    return df


num_imp = SimpleImputer(strategy="median")
Xcore_num = num_imp.fit_transform(prep(core_pa, num_cols)) if num_cols else np.empty((len(core_pa), 0))
if cat_cols:
    cat_imp = SimpleImputer(strategy="most_frequent")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xcore_cat = enc.fit_transform(cat_imp.fit_transform(prep(core_pa, cat_cols)))
else:
    cat_imp, enc = None, None
    Xcore_cat = np.empty((len(core_pa), 0))
Xcore = np.hstack([Xcore_num, Xcore_cat])


def transform(df):
    x_num = num_imp.transform(prep(df, num_cols)) if num_cols else np.empty((len(df), 0))
    if cat_cols:
        x_cat = enc.transform(cat_imp.transform(prep(df, cat_cols)))
    else:
        x_cat = np.empty((len(df), 0))
    return np.hstack([x_num, x_cat])


print("Refitting v6's exact PA-level classifier (assembled baseline)...")
pa_clf = xgb.XGBClassifier(
    n_estimators=2000, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, verbosity=0, eval_metric="logloss", early_stopping_rounds=30,
)
pa_clf.fit(Xcore, core_pa[TARGET], eval_set=[(transform(early_pa), early_pa[TARGET])], verbose=False)

val_pa = val_pa.copy()
val_pa["pred_prob"] = pa_clf.predict_proba(transform(val_pa))[:, 1]
assembled = val_pa.groupby(GROUP_KEYS)["pred_prob"].sum().rename("assembled_expected_k")


# ── 3. Start-grain frame: sum(is_strikeout) as a direct regression target ──
CONSTANT_COLS = [c for c in FEATURE_COLS if c not in BATTER_LEVEL_COLS and c != "batter_bat_side"]
START_FEATURE_COLS = (
    CONSTANT_COLS
    + [f"{c}_lineup_mean" for c in BATTER_LEVEL_COLS]
    + ["frac_batter_left"]
)


def to_start_grain(df):
    agg = df.groupby(GROUP_KEYS).agg(
        **{c: (c, "first") for c in CONSTANT_COLS},
        **{f"{c}_lineup_mean": (c, "mean") for c in BATTER_LEVEL_COLS},
        frac_batter_left=("batter_bat_side", lambda s: (s == "L").mean()),
        total_strikeouts=(TARGET, "sum"),
        game_season=("game_season", "first"),
    )
    return agg.reset_index()


core_start = to_start_grain(core_pa)
early_start = to_start_grain(early_pa)
val_start = to_start_grain(val_pa)

start_num_cols = [c for c in START_FEATURE_COLS if pd.api.types.is_numeric_dtype(core_start[c])]
start_cat_cols = [c for c in START_FEATURE_COLS if c not in start_num_cols]


def prep_start(df, cols):
    df = df[cols].copy()
    n_cols = [c for c in cols if c in start_num_cols]
    c_cols = [c for c in cols if c in start_cat_cols]
    if n_cols:
        df[n_cols] = df[n_cols].apply(pd.to_numeric, errors="coerce")
    if c_cols:
        df[c_cols] = df[c_cols].astype(object).fillna(np.nan)
    return df


start_num_imp = SimpleImputer(strategy="median")
Xcore_start_num = (
    start_num_imp.fit_transform(prep_start(core_start, start_num_cols))
    if start_num_cols else np.empty((len(core_start), 0))
)
if start_cat_cols:
    start_cat_imp = SimpleImputer(strategy="most_frequent")
    start_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xcore_start_cat = start_enc.fit_transform(start_cat_imp.fit_transform(prep_start(core_start, start_cat_cols)))
else:
    start_cat_imp, start_enc = None, None
    Xcore_start_cat = np.empty((len(core_start), 0))
Xcore_start = np.hstack([Xcore_start_num, Xcore_start_cat])


def transform_start(df):
    x_num = start_num_imp.transform(prep_start(df, start_num_cols)) if start_num_cols else np.empty((len(df), 0))
    if start_cat_cols:
        x_cat = start_enc.transform(start_cat_imp.transform(prep_start(df, start_cat_cols)))
    else:
        x_cat = np.empty((len(df), 0))
    return np.hstack([x_num, x_cat])


# ── 4. Direct count model -- Poisson objective (total_strikeouts is a count) ─
# Same grid shape as v6's own classifier tune (max_depth x learning_rate, 9
# configs, early-stopped against EARLY_STOP_SEASON, final selection on the
# untouched VAL_SEASON) -- picking the best of 9 rather than a single
# hand-picked config before deciding whether direct-vs-assembled is settled.
print("Tuning direct count model (XGBoost, count:poisson, max_depth x learning_rate)...")
Xearly_start = transform_start(early_start)
Xval_start = transform_start(val_start)
y_core_start, y_early_start, y_val_start = (
    core_start["total_strikeouts"], early_start["total_strikeouts"], val_start["total_strikeouts"],
)

COUNT_GRID = {"max_depth": [2, 3, 4], "learning_rate": [0.01, 0.03, 0.1]}
count_search_results = []
best_count_model, best_count_mae, best_count_config = None, np.inf, None
for max_depth in COUNT_GRID["max_depth"]:
    for learning_rate in COUNT_GRID["learning_rate"]:
        model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=2000, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
            random_state=42, verbosity=0, eval_metric="poisson-nloglik", early_stopping_rounds=30,
        )
        model.fit(Xcore_start, y_core_start, eval_set=[(Xearly_start, y_early_start)], verbose=False)
        pred = model.predict(Xval_start)
        mae = mean_absolute_error(y_val_start, pred)
        count_search_results.append({
            "max_depth": max_depth, "learning_rate": learning_rate,
            "best_iteration": model.best_iteration, "mae": mae,
        })
        if mae < best_count_mae:
            best_count_model, best_count_mae = model, mae
            best_count_config = {"max_depth": max_depth, "learning_rate": learning_rate, "best_iteration": model.best_iteration}

count_search_df = pd.DataFrame(count_search_results).sort_values("mae")
print(f"All {len(count_search_df)} count-model configs by val MAE:")
print(count_search_df.to_string(index=False))
print(f"Best count-model config: {best_count_config} -> MAE {best_count_mae:.4f}")

count_model = best_count_model
val_start["direct_pred_k"] = count_model.predict(Xval_start)
val_start = val_start.set_index(GROUP_KEYS).join(assembled).reset_index()


# ── 5. Head-to-head: direct count model vs. today's assembled PA-classifier ─
naive_pred = core_start["total_strikeouts"].mean()

mae_naive = mean_absolute_error(val_start["total_strikeouts"], [naive_pred] * len(val_start))
mae_assembled = mean_absolute_error(val_start["total_strikeouts"], val_start["assembled_expected_k"])
mae_direct = mean_absolute_error(val_start["total_strikeouts"], val_start["direct_pred_k"])

print(f"\n{'=' * 72}\nRESULTS -- {VAL_SEASON} val season, {len(val_start):,} SP starts (>= {MIN_PA_PER_START} PA)\n{'=' * 72}")
print(f"Naive (predict CORE_FIT mean, {naive_pred:.2f} K/start):     MAE = {mae_naive:.4f}")
print(f"Assembled (sum of v6 PA-classifier pred_prob):     MAE = {mae_assembled:.4f}")
print(f"Direct count model (count:poisson on start grain): MAE = {mae_direct:.4f}")

print(f"\ncorr(actual, assembled) = {val_start['total_strikeouts'].corr(val_start['assembled_expected_k']):.4f}")
print(f"corr(actual, direct)    = {val_start['total_strikeouts'].corr(val_start['direct_pred_k']):.4f}")

importances = pd.Series(count_model.feature_importances_, index=START_FEATURE_COLS).sort_values(ascending=False)
print(f"\nTop 10 direct-model feature importances:\n{importances.head(10)}")

# ── Next steps (not built here) ─────────────────────────────────────────────
# - If direct beats assembled: check whether it specifically closes the
#   dominant-start misses (Luis Gil-style) that motivated this spike, not
#   just the aggregate MAE -- pull the same worst-residual starts
#   investigate_worst_residuals.py already identified and compare.
# - If flat/worse: close the thread. The ICC-implied variance inflation
#   (~1.21x, see icc_check.py) is small enough that this was always a
#   long-shot bet, not a confident hypothesis.
# - Lineup aggregation here is mean-only (frac_batter_left, batter K-rate
#   lineup mean) -- loses the "one elite masher vs. a lineup of easy outs"
#   distinction v7's toughest-out/star-power features targeted at PA grain.
#   A max/min alongside the mean is the natural follow-up if this thread
#   stays open.
