"""
k_predictor experiment v17: does a pretrained tabular foundation model
(TabPFN) beat v6's tuned XGBoost on the same pre-game PA-grain feature
matrix?

Background (see this session's research, not re-derived here): the
well-established finding (Grinsztajn et al., NeurIPS 2022) is that
gradient-boosted trees still beat from-scratch neural nets on tabular data
at k_predictor's scale (~15-20k-100k+ rows). TabPFN is a different kind of
comparison -- an in-context-learning transformer PRETRAINED on millions of
synthetic tabular tasks, not trained from scratch on this data -- and it
specifically targets the small-to-medium regime (official ceiling: 10,000
training rows, 500 features) where it has shown real wins in published
benchmarks. This experiment is the cheap, honest way to find out if that
transfers to a real PA-grain strikeout target: near-zero feature-engineering
cost, run on the *exact same* pre-game feature matrix v6 already uses.

Design -- apples-to-apples is the whole point:

  TabPFN's row ceiling (10,000) is far below k_predictor's full training
  pool (~630k PA rows across FIT_SEASONS). Comparing TabPFN-on-a-subsample
  against v6's full-data XGBoost fit would confound "different model" with
  "different training set size" -- not a fair test of the model itself. So
  this experiment trains TWO XGBoost variants:

    1. xgb_subsample -- v6's exact tuned hyperparameters (max_depth=2,
       learning_rate=0.03), fit on the SAME stratified 8,000-row subsample
       TabPFN gets. This is the fair, apples-to-apples comparison.
    2. xgb_v6_full (reference only) -- v6's actual cached fit on the full
       ~630k-row pool, loaded as-is. Included to show what XGBoost gives up
       (if anything) by being restricted to TabPFN's row ceiling, but NOT
       the primary verdict -- it differs in training set size, not just
       model family.

  All three are evaluated on the identical, untouched VAL_SEASON=2024
  population (105,265 PA rows) -- same protocol every version since v7 has
  used. TEST_SEASON (2025) stays untouched per config.yaml's own comment.

Subsampling uses the new, TDD'd stratified_subsample helper
(k_predictor/utils/sampling.py, tests/k_predictor/test_sampling.py) --
preserves the ~22% strikeout base rate, reproducible with a fixed seed.

TabPFN needs a one-time Prior Labs license acceptance + API token before it
will download model weights: see https://ux.priorlabs.ai (Licenses tab),
then `export TABPFN_TOKEN="<key>"` before running this script.

Run from src/models/k_predictor/ with:
    python experiments/v17_tabpfn_comparison/train.py
No AWS credentials needed -- loads the same cached model_df_v6.parquet
(733,275 PA rows, v6's 42 pre-game features already built) that v16 used,
per this repo's own "any future k_predictor-v6-feature-set diagnostic
should load that cache instead of rebuilding from S3" note.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

from models.k_predictor.utils.sampling import stratified_subsample

STAGE = Path(__file__).parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
V6_CACHE_DIR = BASE_DIR / "backtest" / "_model_cache" / "v6"
MODEL_DF_CACHE = BASE_DIR / "experiments" / "count_distribution_check" / "_model_cache" / "model_df_v6.parquet"

# TabPFN's official pretraining ceiling is 10,000 training rows / 500
# features. Stay comfortably under it rather than override with
# ignore_pretraining_limits -- the point is to test TabPFN in its real,
# supported regime, not a stretched one.
TABPFN_SUBSAMPLE_N = 8000
SHORT_PITCHER_WINDOW = 3
SHORT_TEAM_WINDOW = 5
short_team_prefix = f"team_roll_last{SHORT_TEAM_WINDOW}g_pa_strikeout_rate"


# ── 1. Config (unchanged shape from v6_tuned/train.py) ────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

TARGET, DATE_COL = cfg["target_column"], cfg["date_column"]
TEST_SEASON, VAL_SEASON = cfg["test_season"], cfg["val_season"]
TRAIN_SEASONS = cfg["train_seasons"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)


# ── 2. Load the cached PA-grain frame; v6's 42 pre-game features only --
# deliberately excludes v16's in-game context features, since those don't
# exist before first pitch and this experiment targets the same pre-game
# prop use case v6 does. ───────────────────────────────────────────────────
print(f"\nLoading cached PA-grain frame from {MODEL_DF_CACHE}...")
pa_outcome = pd.read_parquet(MODEL_DF_CACHE)
print(f"Loaded {len(pa_outcome):,} rows, {pa_outcome.shape[1]} columns, seasons {sorted(pa_outcome['game_season'].unique())}")

V6_FEATURE_COLS = [
    "expected_pitcher_role", "pitcher_throw_hand", "batter_bat_side",
    "pitcher_last_season_pa_strikeout_rate", "pitcher_last_season_whip", "batter_last_season_pa_strikeout_rate",
    "pitcher_roll_season_ip", "pitcher_roll_season_whip", "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate", "pitcher_roll_season_hr_rate", "pitcher_roll_season_games_n",
    "pitcher_roll_season_avg_ip_per_game", "pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate",
    "batter_roll_season_pa_strikeout_rate", "opp_team_roll_season_pa_strikeout_rate", "pitching_team_roll_season_pa_strikeout_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_whip",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_k_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_bb_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_hr_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_avg_ip_per_game", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_total",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_strikeout_rate",
    "opp_team_roll_season_pa_strikeout_rate_mean", "opp_team_roll_season_pa_strikeout_rate_std",
    "opp_team_roll_season_pa_strikeout_rate_max",
    f"opp_{short_team_prefix}_mean", f"opp_{short_team_prefix}_std", f"opp_{short_team_prefix}_max",
    "pitcher_roll_season_command_swinging_strike_rate", "pitcher_roll_season_pa_pitch_count_mean",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_command_swinging_strike_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean",
    "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight",
    "weather_condition", "weather_temp", "expected_times_through_order",
]
V6_FEATURE_COLS = [c for c in V6_FEATURE_COLS if c in pa_outcome.columns]
FEATURE_COLS = V6_FEATURE_COLS
print(f"Feature count: {len(FEATURE_COLS)} (v6 pre-game features only, no in-game context)")

model_df = pa_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(model_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

fit_pool_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df = model_df[model_df["game_season"] == VAL_SEASON].copy()
print(f"Fit pool (all FIT_SEASONS): {len(fit_pool_df):,} rows  |  VAL_SEASON={VAL_SEASON}: {len(val_df):,} rows")

subsample_df = stratified_subsample(fit_pool_df, target_col=TARGET, n=TABPFN_SUBSAMPLE_N, random_state=42)
print(f"Stratified subsample for TabPFN + xgb_subsample: {len(subsample_df):,} rows "
      f"(target rate {subsample_df[TARGET].mean():.4f} vs full-pool {fit_pool_df[TARGET].mean():.4f})")


# ── 3. Shared preprocessing -- fit ONLY on the subsample, since that's the
# actual training data both xgb_subsample and TabPFN see. ─────────────────────
def clean_cols(df, cols_num, cols_cat):
    df = df[cols_num + cols_cat].copy()
    if cols_num:
        df[cols_num] = df[cols_num].apply(pd.to_numeric, errors="coerce")
    if cols_cat:
        df[cols_cat] = df[cols_cat].astype(object).fillna(np.nan)
    return df


num_imp = SimpleImputer(strategy="median")
Xsub_num = num_imp.fit_transform(clean_cols(subsample_df, num_cols, [])[num_cols]) if num_cols else np.empty((len(subsample_df), 0))
if cat_cols:
    cat_imp = SimpleImputer(strategy="most_frequent")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xsub_cat = enc.fit_transform(cat_imp.fit_transform(clean_cols(subsample_df, [], cat_cols)[cat_cols]))
else:
    cat_imp, enc = None, None
    Xsub_cat = np.empty((len(subsample_df), 0))
X_sub = np.hstack([Xsub_num, Xsub_cat])
y_sub = subsample_df[TARGET].to_numpy()


def transform(df):
    df_num = clean_cols(df, num_cols, [])[num_cols] if num_cols else df
    x_num = num_imp.transform(df_num[num_cols]) if num_cols else np.empty((len(df), 0))
    if cat_cols:
        df_cat = clean_cols(df, [], cat_cols)[cat_cols]
        x_cat = enc.transform(cat_imp.transform(df_cat))
    else:
        x_cat = np.empty((len(df), 0))
    return np.hstack([x_num, x_cat])


X_val = transform(val_df)
y_val = val_df[TARGET].to_numpy()


# ── 4. Fit xgb_subsample -- v6's exact tuned hyperparameters, same 8,000-row
# subsample TabPFN gets. A 90/10 stratified split of the subsample itself
# provides the early-stopping monitor (v6/v16 used a held-out season instead,
# but the subsample pools all fit seasons together, so a random split is the
# closest equivalent at this size). ─────────────────────────────────────────
X_core, X_es, y_core, y_es = train_test_split(X_sub, y_sub, test_size=0.10, stratify=y_sub, random_state=42)

print("\nFitting xgb_subsample (max_depth=2, learning_rate=0.03, early-stopped on 10% of the subsample)...")
xgb_subsample = xgb.XGBClassifier(
    n_estimators=2000, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, verbosity=0, eval_metric="logloss", early_stopping_rounds=30,
)
xgb_subsample.fit(X_core, y_core, eval_set=[(X_es, y_es)], verbose=False)
print(f"best_iteration={xgb_subsample.best_iteration}")


# ── 5. Fit TabPFN on the identical 8,000-row subsample. ───────────────────────
print("\nFitting TabPFN on the same subsample...")
from tabpfn import TabPFNClassifier

t0 = time.time()
tabpfn_clf = TabPFNClassifier(device="auto", random_state=42)
tabpfn_clf.fit(X_sub, y_sub)
print(f"TabPFN fit time: {time.time() - t0:.1f}s")


# ── 6. Load v6's actual cached full-data fit -- reference only, NOT the
# primary comparison (different training set size). ───────────────────────────
v6_cached = joblib.load(V6_CACHE_DIR / "preprocessing.pkl")
v6_num_imp, v6_cat_imp, v6_enc = v6_cached["num_imp"], v6_cached["cat_imp"], v6_cached["enc"]
v6_feature_cols, v6_num_cols, v6_cat_cols = v6_cached["feature_cols"], v6_cached["num_cols"], v6_cached["cat_cols"]
xgb_v6_full = xgb.XGBClassifier()
xgb_v6_full.load_model(V6_CACHE_DIR / "xgb_model.json")


def v6_clean_cols(df, cols_num, cols_cat):
    df = df[cols_num + cols_cat].copy()
    if cols_num:
        df[cols_num] = df[cols_num].apply(pd.to_numeric, errors="coerce")
    if cols_cat:
        df[cols_cat] = df[cols_cat].astype(object).fillna(np.nan)
    return df


def v6_transform(df):
    df = df[v6_feature_cols].copy()
    x_num = v6_num_imp.transform(v6_clean_cols(df, v6_num_cols, [])[v6_num_cols]) if v6_num_cols else np.empty((len(df), 0))
    if v6_cat_cols:
        x_cat = v6_enc.transform(v6_cat_imp.transform(v6_clean_cols(df, [], v6_cat_cols)[v6_cat_cols]))
    else:
        x_cat = np.empty((len(df), 0))
    return np.hstack([x_num, x_cat])


# ── 7. Evaluate all three on the identical, untouched VAL_SEASON population.
# TabPFN inference is batched -- its cost scales with train_size x test_size,
# and 8,000 x 105,265 in one call risks the same memory-pressure failure mode
# this repo has hit before (see v16/v6's changelog notes). ────────────────────
def pr_roc(y_true, y_prob):
    return average_precision_score(y_true, y_prob), roc_auc_score(y_true, y_prob)


print(f"\nScoring TabPFN on VAL_SEASON ({len(val_df):,} rows) in batches...")
BATCH = 2000
tabpfn_probs = np.empty(len(X_val))
t0 = time.time()
for start in range(0, len(X_val), BATCH):
    end = min(start + BATCH, len(X_val))
    tabpfn_probs[start:end] = tabpfn_clf.predict_proba(X_val[start:end])[:, 1]
    if start % (BATCH * 5) == 0:
        print(f"  {end:,}/{len(X_val):,} rows scored ({time.time() - t0:.1f}s elapsed)")
print(f"TabPFN inference total time: {time.time() - t0:.1f}s")

pred_xgb_sub = xgb_subsample.predict_proba(X_val)[:, 1]
pred_xgb_full = xgb_v6_full.predict_proba(v6_transform(val_df))[:, 1]

pr_tabpfn, roc_tabpfn = pr_roc(y_val, tabpfn_probs)
pr_xgb_sub, roc_xgb_sub = pr_roc(y_val, pred_xgb_sub)
pr_xgb_full, roc_xgb_full = pr_roc(y_val, pred_xgb_full)

print(f"\n{'=' * 78}\nRESULT -- VAL_SEASON={VAL_SEASON}, {len(val_df):,} PA rows, {len(FEATURE_COLS)} pre-game features\n{'=' * 78}")
print(f"{'Model':<40}{'Train rows':>12}{'PR-AUC':>10}{'ROC-AUC':>10}")
print(f"{'TabPFN (subsample)':<40}{len(subsample_df):>12,}{pr_tabpfn:>10.4f}{roc_tabpfn:>10.4f}")
print(f"{'XGBoost (same subsample)':<40}{len(subsample_df):>12,}{pr_xgb_sub:>10.4f}{roc_xgb_sub:>10.4f}")
print(f"{'XGBoost v6 (full pool, reference)':<40}{len(fit_pool_df):>12,}{pr_xgb_full:>10.4f}{roc_xgb_full:>10.4f}")
print(f"\nApples-to-apples delta (TabPFN - XGBoost, same {len(subsample_df):,}-row subsample): "
      f"PR-AUC {pr_tabpfn - pr_xgb_sub:+.4f}  ROC-AUC {roc_tabpfn - roc_xgb_sub:+.4f}")
