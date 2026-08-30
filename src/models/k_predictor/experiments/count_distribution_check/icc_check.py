"""
k_predictor: how big is the within-game 'quality today' effect, actually?

within_game_correlation_check.py found r=0.105 (p=8.8e-13) between the SUM of
odd-PA residuals and the SUM of even-PA residuals within a start. That's a
correlation between two ~10-PA GROUP SUMS, not the pairwise correlation
between any two individual PA residuals -- those are different numbers, and
conflating them overstates the effect. This computes the actual quantity that
determines how much the total-K distribution's variance should widen: the
intraclass correlation (ICC) of individual PA-level residuals within a start,
via a standard one-way random-effects ANOVA decomposition (unbalanced groups).

Uses the cache from within_game_correlation_check.py -- no S3 read.
Run from src/models/k_predictor/ with: python experiments/count_distribution_check/icc_check.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

CACHE_DIR = Path(__file__).parent / "_model_cache"
model_df = pd.read_parquet(CACHE_DIR / "model_df_v6.parquet")
with open(CACHE_DIR / "feature_cols_v6.json") as f:
    FEATURE_COLS = json.load(f)

TARGET = "is_strikeout"
VAL_SEASON = 2024
FIT_SEASONS = [2018, 2019, 2022, 2023]
EARLY_STOP_SEASON = 2023
CORE_FIT_SEASONS = [2018, 2019, 2022]
MIN_PA_PER_START = 10

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(model_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

core_train_df = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_stop_df = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()
val_df = model_df[model_df["game_season"] == VAL_SEASON].copy()


def fit_transform(fit_df, cols):
    fit_df = fit_df[cols].copy()
    n_cols = [c for c in cols if c in num_cols]
    c_cols = [c for c in cols if c in cat_cols]
    if n_cols:
        fit_df[n_cols] = fit_df[n_cols].apply(pd.to_numeric, errors="coerce")
    if c_cols:
        fit_df[c_cols] = fit_df[c_cols].astype(object).fillna(np.nan)
    return fit_df


num_imp = SimpleImputer(strategy="median")
Xcore_num = num_imp.fit_transform(fit_transform(core_train_df, num_cols)) if num_cols else np.empty((len(core_train_df), 0))
if cat_cols:
    cat_imp = SimpleImputer(strategy="most_frequent")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xcore_cat = enc.fit_transform(cat_imp.fit_transform(fit_transform(core_train_df, cat_cols)))
else:
    cat_imp, enc = None, None
    Xcore_cat = np.empty((len(core_train_df), 0))
Xcore = np.hstack([Xcore_num, Xcore_cat])
y_core = core_train_df[TARGET]


def transform(df):
    df = df[FEATURE_COLS].copy()
    x_num = num_imp.transform(fit_transform(df, num_cols)) if num_cols else np.empty((len(df), 0))
    if cat_cols:
        x_cat = enc.transform(cat_imp.transform(fit_transform(df, cat_cols)))
    else:
        x_cat = np.empty((len(df), 0))
    return np.hstack([x_num, x_cat])


Xearly = transform(early_stop_df)
y_early = early_stop_df[TARGET]

print("Fitting XGBoost (max_depth=2, learning_rate=0.03, early-stopped)...")
best_xgb = xgb.XGBClassifier(
    n_estimators=2000, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, verbosity=0, eval_metric="logloss", early_stopping_rounds=30,
)
best_xgb.fit(Xcore, y_core, eval_set=[(Xearly, y_early)], verbose=False)
print(f"best_iteration={best_xgb.best_iteration}")

val_df["pred_prob"] = best_xgb.predict_proba(transform(val_df))[:, 1]
val_df["residual"] = val_df[TARGET].astype(float) - val_df["pred_prob"]

start_counts = val_df.groupby(["gamepk", "pitcher_id"]).size()
eligible = start_counts[start_counts >= MIN_PA_PER_START].index
val_idx = val_df.set_index(["gamepk", "pitcher_id"])
val_df = val_idx[val_idx.index.isin(eligible)].reset_index()

# ── One-way random-effects ANOVA, unbalanced groups (Fisher's classic ICC(1)) ─
g = val_df.groupby(["gamepk", "pitcher_id"])
group_sizes = g["residual"].size()
group_means = g["residual"].mean()
grand_mean = val_df["residual"].mean()
N = len(val_df)
G = len(group_sizes)

ssb = (group_sizes * (group_means - grand_mean) ** 2).sum()
ssw = ((val_df.groupby(["gamepk", "pitcher_id"])["residual"].transform(lambda x: (x - x.mean()) ** 2))).sum()
msb = ssb / (G - 1)
msw = ssw / (N - G)
n0 = (N - (group_sizes ** 2).sum() / N) / (G - 1)  # average adjusted group size, unbalanced case

icc = (msb - msw) / (msb + (n0 - 1) * msw)

print(f"\n{'=' * 72}\nINDIVIDUAL-PA-LEVEL ICC (one-way random-effects ANOVA)\n{'=' * 72}")
print(f"n PAs: {N:,}   n starts: {G:,}   avg PAs/start: {N/G:.1f}   n0 (adj.): {n0:.2f}")
print(f"MSB: {msb:.5f}   MSW: {msw:.5f}")
print(f"ICC (pairwise correlation between two individual PA residuals, same start): {icc:.4f}")

# ── What this actually does to the total-K distribution's spread ──────────────
avg_n_slots = 22  # matches run_xgboost_uncertainty.py's observed mean
multiplier = 1 + (avg_n_slots - 1) * icc
print(f"\nAt ~{avg_n_slots} PAs/start, variance inflation factor over independence: {multiplier:.3f}x")
print(f"=> SD inflation factor: {np.sqrt(multiplier):.3f}x")
print(f"(compare to the r=0.105 half-sum correlation this project's first pass reported --")
print(f" that number describes correlation between two ~11-PA GROUP SUMS, not the individual-")
print(f" PA pairwise correlation that actually drives total-K variance)")
