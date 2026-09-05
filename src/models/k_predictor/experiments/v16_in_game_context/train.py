"""
k_predictor experiment v16: in-game context -- does knowing how THIS start
has gone so far (realized within-game state, not a pre-game projection)
improve predicting the NEXT plate appearance?

Prompted directly by the already-established "hot/cold tonight" finding in
experiments/count_distribution_check/within_game_correlation_check.py +
icc_check.py: splitting each start's PAs into odd/even halves by real
chronological order found a real, statistically significant residual
correlation (r=0.105, p=8.8e-13, n=4,629 starts) and a real per-PA ICC
(0.0100) -- small, but never real-checked or exploited: nobody has fed the
pitcher's actual within-game running state to the model as literal features
to see if a shallow tree can use it PA-by-PA. This is that check.

Deliberately scoped as a DIAGNOSTIC, not a production-candidate pass:

  1. NOT a pre-game feature. Every prior k_predictor version (v1-v15) builds
     features knowable before first pitch, because the real-money use case
     (a total-strikeouts prop) needs a prediction BEFORE the game starts. In-
     game running state literally doesn't exist yet at that point -- these
     features cannot go into any pre-game total-K prediction. This
     experiment answers a narrower, purely diagnostic question instead: once
     a game IS in progress, does knowing what's happened so far sharpen the
     prediction for the very next batter?
  2. No game-grain aggregation check (run_pa_vs_game_grain_check, the usual
     second half of every version's verdict) -- that check assumes a single
     PRE-GAME probability per slot, aggregated across a whole start. There's
     no single "pre-game" probability here to aggregate; it's evaluated
     PA-by-PA only.
  3. New features come from processing/features/in_game_context.py
     (TDD'd, 7 tests: point-in-time-within-game correctness, no
     cross-contamination between two pitchers in the same gamepk, order-by-
     play_id not input-row-order, reset at the start of a new game). The
     "hot/cold" feature deliberately uses the pitcher's own EXISTING
     pre-game rolling K-rate as the "expected" baseline, not this model's
     own predicted probabilities -- using a model's own in-sample
     predictions as a feature into itself (fit on the same rows) would be
     circular/leaky; the pre-game rate is already point-in-time-safe on its
     own and carries no such risk.

Evaluated on VAL_SEASON (2024) against v6's cached fit, same protocol every
version since v7 has used -- TEST_SEASON (2025) stays untouched per
config.yaml's own comment ("held-out -- never evaluated during
development"). Two cuts: the full VAL_SEASON population (where most PAs are
early in a start and these features carry little/no information by
construction), and the subset where the pitcher has already faced >=3
batters this game -- the regime the features were actually built for.

Run from src/models/k_predictor/ with:
    python experiments/v16_in_game_context/train.py
No AWS credentials needed -- loads the already-built, already-cached
model_df_v6.parquet (733,275 PA rows, all 7 relevant seasons, v6's 42
features + gamepk/pitcher_id/play_id/is_strikeout) from
experiments/count_distribution_check/_model_cache/, per ROADMAP's own
"any future k_predictor-v6-feature-set diagnostic should load that cache
instead of rebuilding from S3" note. A first attempt at this script rebuilt
the full pipeline from raw S3 data (same shape as v6_tuned/train.py) and was
killed by the OS partway through the multi-season feature build --
apparently the same real memory-pressure issue this project has hit before
(see v6's own changelog entry: "the run itself hit real system memory
thrashing on its first attempt... needed a kill + restart"). Loading the
cache sidesteps that entirely and is also just much faster.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

from models.k_predictor.processing.features.in_game_context import (
    build_pitcher_in_game_running_stats,
    build_pitcher_in_game_hot_cold_gap,
)

STAGE = Path(__file__).parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
V6_CACHE_DIR = BASE_DIR / "backtest" / "_model_cache" / "v6"
MODEL_DF_CACHE = BASE_DIR / "experiments" / "count_distribution_check" / "_model_cache" / "model_df_v6.parquet"

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


# ── 2. Load the already-built PA-grain frame from cache + the two new
# in-game context feature calls (the only new production code this
# experiment adds). ────────────────────────────────────────────────────────────
print(f"\nLoading cached PA-grain frame from {MODEL_DF_CACHE}...")
pa_outcome = pd.read_parquet(MODEL_DF_CACHE)
print(f"Loaded {len(pa_outcome):,} rows, {pa_outcome.shape[1]} columns, seasons {sorted(pa_outcome['game_season'].unique())}")

# ── new for v16: realized in-game running state, keyed on the REALIZED
# pitcher_id (already sp-filtered upstream when this cache was built), not
# expected_pitcher_key_id -- these describe what actually happened, not a
# pre-game estimate. ────────────────────────────────────────────────────────
pa_outcome = build_pitcher_in_game_running_stats(pa_outcome)
pa_outcome = build_pitcher_in_game_hot_cold_gap(pa_outcome, pregame_rate_col="pitcher_roll_season_pa_strikeout_rate")

IN_GAME_FEATURE_COLS = [
    "pitcher_pa_faced_this_game_so_far", "pitcher_k_this_game_so_far",
    "pitcher_k_this_game_so_far_rate", "pitcher_expected_k_this_game_so_far",
    "pitcher_hot_cold_gap_this_game_so_far",
]

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
FEATURE_COLS = V6_FEATURE_COLS + IN_GAME_FEATURE_COLS
print(f"v6 feature count: {len(V6_FEATURE_COLS)}  |  + {len(IN_GAME_FEATURE_COLS)} in-game context  =  {len(FEATURE_COLS)} total")

model_df = pa_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(model_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

EARLY_STOP_SEASON = max(FIT_SEASONS)
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]
print(f"Core fit seasons: {CORE_FIT_SEASONS}  |  early-stop season: {EARLY_STOP_SEASON}  |  eval: VAL_SEASON={VAL_SEASON}")

core_train_df = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_stop_df = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()
val_df = model_df[model_df["game_season"] == VAL_SEASON].copy()


# ── 3. Fit v6's exact winning XGBoost config (max_depth=2, lr=0.03,
# early-stopped) on the EXPANDED feature set -- same hyperparameters as
# every version since v7, only the feature list changes. ──────────────────────
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

print("\nFitting XGBoost (max_depth=2, learning_rate=0.03, early-stopped)...")
v16_xgb = xgb.XGBClassifier(
    n_estimators=2000, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, verbosity=0, eval_metric="logloss", early_stopping_rounds=30,
)
v16_xgb.fit(Xcore, y_core, eval_set=[(Xearly, y_early)], verbose=False)
print(f"best_iteration={v16_xgb.best_iteration}")

CACHE_DIR = STAGE / "_model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
v16_xgb.save_model(CACHE_DIR / "xgb_model.json")
joblib.dump({
    "num_imp": num_imp, "cat_imp": cat_imp, "enc": enc,
    "feature_cols": FEATURE_COLS, "num_cols": num_cols, "cat_cols": cat_cols,
}, CACHE_DIR / "preprocessing.pkl")


# ── 4. Load v6's cached fit for a same-rows baseline comparison. ──────────────
v6_cached = joblib.load(V6_CACHE_DIR / "preprocessing.pkl")
v6_num_imp, v6_cat_imp, v6_enc = v6_cached["num_imp"], v6_cached["cat_imp"], v6_cached["enc"]
v6_feature_cols, v6_num_cols, v6_cat_cols = v6_cached["feature_cols"], v6_cached["num_cols"], v6_cached["cat_cols"]
v6_model = xgb.XGBClassifier()
v6_model.load_model(V6_CACHE_DIR / "xgb_model.json")


def v6_fit_transform(fit_df, cols):
    fit_df = fit_df[cols].copy()
    n_cols = [c for c in cols if c in v6_num_cols]
    c_cols = [c for c in cols if c in v6_cat_cols]
    if n_cols:
        fit_df[n_cols] = fit_df[n_cols].apply(pd.to_numeric, errors="coerce")
    if c_cols:
        fit_df[c_cols] = fit_df[c_cols].astype(object).fillna(np.nan)
    return fit_df


def v6_transform(df):
    df = df[v6_feature_cols].copy()
    x_num = v6_num_imp.transform(v6_fit_transform(df, v6_num_cols)) if v6_num_cols else np.empty((len(df), 0))
    if v6_cat_cols:
        x_cat = v6_enc.transform(v6_cat_imp.transform(v6_fit_transform(df, v6_cat_cols)))
    else:
        x_cat = np.empty((len(df), 0))
    return np.hstack([x_num, x_cat])


# ── 5. Evaluate PA-grain on VAL_SEASON: full population + the >=3-PA-faced
# subset the in-game features were actually built for. ────────────────────────
def pr_roc(y_true, y_prob):
    return average_precision_score(y_true, y_prob), roc_auc_score(y_true, y_prob)


val_pred_v16 = v16_xgb.predict_proba(transform(val_df))[:, 1]
val_pred_v6 = v6_model.predict_proba(v6_transform(val_df))[:, 1]
y_val = val_df[TARGET]

pr16_full, roc16_full = pr_roc(y_val, val_pred_v16)
pr6_full, roc6_full = pr_roc(y_val, val_pred_v6)

subset_mask = val_df["pitcher_pa_faced_this_game_so_far"] >= 3
n_subset = int(subset_mask.sum())
pr16_sub, roc16_sub = pr_roc(y_val[subset_mask], val_pred_v16[subset_mask])
pr6_sub, roc6_sub = pr_roc(y_val[subset_mask], val_pred_v6[subset_mask])

print(f"\n{'=' * 72}\nRESULT -- VAL_SEASON={VAL_SEASON}, {len(val_df):,} PA rows\n{'=' * 72}")
print(f"Full population:")
print(f"  v6  (pre-game only, {len(v6_feature_cols)} features):  PR-AUC={pr6_full:.4f}  ROC-AUC={roc6_full:.4f}")
print(f"  v16 (+ in-game context, {len(FEATURE_COLS)} features):  PR-AUC={pr16_full:.4f}  ROC-AUC={roc16_full:.4f}")
print(f"  delta: PR-AUC {pr16_full - pr6_full:+.4f}  ROC-AUC {roc16_full - roc6_full:+.4f}")

print(f"\n>=3 batters already faced this game ({n_subset:,} of {len(val_df):,} PA rows, "
      f"{n_subset / len(val_df):.1%} -- the regime these features were built for):")
print(f"  v6  PR-AUC={pr6_sub:.4f}  ROC-AUC={roc6_sub:.4f}")
print(f"  v16 PR-AUC={pr16_sub:.4f}  ROC-AUC={roc16_sub:.4f}")
print(f"  delta: PR-AUC {pr16_sub - pr6_sub:+.4f}  ROC-AUC {roc16_sub - roc6_sub:+.4f}")

print(f"\n{'=' * 72}\nFEATURE IMPORTANCE -- where do the 5 new in-game columns rank?\n{'=' * 72}")
importance = pd.Series(v16_xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
rank = {c: i + 1 for i, c in enumerate(importance.index)}
for c in IN_GAME_FEATURE_COLS:
    print(f"  {c:45s}  rank {rank[c]:3d} / {len(FEATURE_COLS)}   importance={importance[c]:.4f}")
print("\nTop 10 overall:")
print(importance.head(10).to_string())
