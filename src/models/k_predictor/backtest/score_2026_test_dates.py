"""
k_predictor production backtest -- score v6's tuned XGBoost on 2026,
restricted to real dates real DraftKings pitcher_strikeouts odds exist for.

ROADMAP.md's Near-term backlog (k_predictor) item 6(c): the 2025 real-odds
backtest (score_2025_test_dates.py, 7 dates / 131 matched starts) came back
statistically underpowered -- see significance_check.py. Rather than pay
The Odds API's historical endpoint's 10x credit rate for more 2025 days,
daily_odds_fetch has been collecting real 2026 player_props at the normal
rate since ~2026-05 -- 72 dates already sit in
s3://mlbdk/raw_data/odds/player_props/2026/ for free. 2026 is a cleaner
holdout than 2025, too: it isn't even in config.yaml's train_seasons list
(2025 is, though excluded from FIT_SEASONS by the val/test subtraction
below), so there's no dependency on that exclusion logic working correctly.

Copied from score_2025_test_dates.py (per this project's established
convention of copy-adapting experiment scripts rather than sharing them as
a library) with three changes:
  1. Sections 1-3 (data load, feature build, model fit) are UNCHANGED in
     substance -- FIT_SEASONS/CORE_FIT_SEASONS/EARLY_STOP_SEASON are all
     still derived from config.yaml's train_seasons, which never included
     2026 to begin with, so 2026 rows never influence fitting. The only
     change is which seasons get READ (pbp/schedule/boxscore need 2026's
     rows to build 2026 features and realized outcomes; the fit itself
     doesn't).
  2. BACKTEST_DATES is the 72 dates player_props/2026/ actually has on S3
     (as of 2026-09-01) instead of the original's 7 hand-picked 2025 dates.
     Three of these (07-13/14/15, all-star break) are near-empty odds files
     with no real games -- left in rather than hand-filtered; they just
     produce 0 matched starts downstream.
  3. Section 4 onward scores SCORE_SEASON=2026 instead of TEST_SEASON=2025.

Run from src/models/k_predictor/ with:
    python backtest/score_2026_test_dates.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
from pathlib import Path

import awswrangler as wr
import boto3
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import yaml

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import expected_role
from models.hit_predictor.processing.features import game_context

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip
from models.hit_predictor.utils.count_distribution import poisson_binomial_pmf

BASE_DIR = Path(__file__).resolve().parent.parent
WHIP_SHRINKAGE_K = 20.0
PA_SHRINKAGE_K = 5.0
MAX_SLOTS = 45
SHORT_PITCHER_WINDOW = 3
SHORT_TEAM_WINDOW = 5

# The 72 dates player_props/2026/ has on S3 as of 2026-09-01 (aws s3 ls
# s3://mlbdk/raw_data/odds/player_props/2026/) -- scoring is restricted to
# SP starts on exactly these dates, since that's the only real market data
# this backtest can compare against. Gaps (05-19..05-31, 06-17..06-30,
# 07-21..07-31, 08-16..08-26) match the pre-2026-08-27 free-tier quota
# exhaustion documented in CLAUDE.md's Known Issues, not a new bug.
BACKTEST_DATES = [
    "2026-04-30",
    "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
    "2026-05-09", "2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13",
    "2026-05-14", "2026-05-15", "2026-05-16", "2026-05-17", "2026-05-18",
    "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05",
    "2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09", "2026-06-10",
    "2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14", "2026-06-15",
    "2026-06-16",
    "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05",
    "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10",
    "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15",
    "2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19", "2026-07-20",
    "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05",
    "2026-08-06", "2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
    "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14", "2026-08-15",
    "2026-08-27", "2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31",
]
SCORE_SEASON = 2026

OUT_DIR = Path(__file__).parent


# ── 1. Config + load (fit logic unchanged from run_xgboost_uncertainty.py;
# read seasons extended with SCORE_SEASON so 2026 rows exist to score, while
# FIT_SEASONS/CORE_FIT_SEASONS/EARLY_STOP_SEASON stay derived purely from
# config.yaml's train_seasons, which never included 2026) ─────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET, REGION = cfg["bucket"], cfg["region"]
TRAIN_SEASONS, FEATURE_SEASONS = cfg["train_seasons"], cfg["feature_seasons"]
TARGET, DATE_COL = cfg["target_column"], cfg["date_column"]
TEST_SEASON, VAL_SEASON = cfg["test_season"], cfg["val_season"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)
assert SCORE_SEASON not in FIT_SEASONS, "SCORE_SEASON must never be fit on"

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))
READ_SEASONS = sorted(set(TRAIN_SEASONS) | {SCORE_SEASON})
READ_BOXSCORE_SEASONS = sorted(set(all_boxscore_seasons) | {SCORE_SEASON})


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
pbp = read_parquet_seasons("s3://{bucket}/processed_data/prepared/playbyplay/{season}/", READ_SEASONS, chunked=True)
print("\nLoading schedule...")
schedule = read_parquet_seasons("s3://{bucket}/processed_data/games/schedule/{season}/", READ_BOXSCORE_SEASONS)
print("\nLoading game info...")
game_info = read_parquet_seasons("s3://{bucket}/processed_data/games/game_info/{season}/", READ_SEASONS)
print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", READ_BOXSCORE_SEASONS)
print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", READ_BOXSCORE_SEASONS)
print("\nLoading player info...")
player_info = wr.s3.read_parquet(path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session)


# ── 2. Build PA-grain frame + v3/v6's full 42-feature set (unchanged) ─────────
print("\nBuilding PA-grain DataFrame...")
schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)
pbp = pbp.merge(schedule[["gamepk", "game_datetime"]], on="gamepk", how="left")

pa_outcome = pipeline.create_pa_outcome_strikeout(pbp, batter_boxscore, game_info, schedule)
pa_outcome["game_season"] = pa_outcome["game_date"].dt.year

pitcher_start_depth_stats = season_stats.build_pitcher_start_depth_stats(pbp)
league_avg_start_depth = season_stats.build_league_avg_start_depth(pitcher_start_depth_stats)
pa_outcome = expected_role.assign_expected_pitcher_role(pa_outcome, pitcher_start_depth_stats, league_avg_start_depth)

pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_pa_strikeout_rate"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})
pitcher_box_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_whip"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})
batter_season_stats = season_stats.build_pbp_batter_feats(pbp)[["batter_id", "game_season", "batter_last_season_pa_strikeout_rate"]]

pa_outcome = pa_outcome.merge(pitcher_role_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left")
pa_outcome = pa_outcome.merge(pitcher_box_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left")
pa_outcome = pa_outcome.merge(batter_season_stats, on=["game_season", "batter_id"], how="left")

pitcher_box_rolling = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window="season")
box_rolling_cols = [
    "pitcher_roll_season_ip", "pitcher_roll_season_whip", "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate", "pitcher_roll_season_hr_rate", "pitcher_roll_season_games_n",
]
pa_outcome = pa_outcome.merge(
    pitcher_box_rolling.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + box_rolling_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

pitcher_pbp_rolling = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window="season")
pbp_rolling_cols = [
    "pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate",
    "pitcher_roll_season_command_swinging_strike_rate", "pitcher_roll_season_pa_pitch_count_mean",
]
pa_outcome = pa_outcome.merge(
    pitcher_pbp_rolling.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + pbp_rolling_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome["pitcher_roll_season_avg_ip_per_game"] = (
    pa_outcome["pitcher_roll_season_ip"] / pa_outcome["pitcher_roll_season_games_n"].replace(0, np.nan)
)

batter_pbp_rolling = rolling_stats.build_pbp_batter_rolling_feats(pbp, window="season")
pa_outcome = pa_outcome.merge(
    batter_pbp_rolling[["batter_id", "gamepk", "batter_roll_season_pa_strikeout_rate"]],
    on=["batter_id", "gamepk"], how="left",
)

team_batter_rolling = rolling_stats.build_team_batter_strikeout_rolling_feats(pbp, batter_boxscore, window="season")
opp_team_rolling = team_batter_rolling.rename(columns={
    "team_roll_season_pa_strikeout_rate": "opp_team_roll_season_pa_strikeout_rate",
})[["batter_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")

pitching_team_rolling = team_batter_rolling.rename(columns={
    "batter_team_id": "pitcher_team_id",
    "team_roll_season_pa_strikeout_rate": "pitching_team_roll_season_pa_strikeout_rate",
})[["pitcher_team_id", "gamepk", "pitching_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")

pitcher_box_rolling3 = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window=SHORT_PITCHER_WINDOW)
box_rolling3_cols = [
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_whip",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_k_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_bb_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_hr_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n",
]
pa_outcome = pa_outcome.merge(
    pitcher_box_rolling3.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + box_rolling3_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pitcher_pbp_rolling3 = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window=SHORT_PITCHER_WINDOW)
pbp_rolling3_cols = [
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_total", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_strikeout_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_command_swinging_strike_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean",
]
pa_outcome = pa_outcome.merge(
    pitcher_pbp_rolling3.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"] + pbp_rolling3_cols
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_avg_ip_per_game"] = (
    pa_outcome[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip"]
    / pa_outcome[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n"].replace(0, np.nan)
)

opp_team_volatility_season = rolling_stats.build_team_strikeout_volatility(pbp, batter_boxscore, window="season")
pa_outcome = pa_outcome.merge(
    opp_team_volatility_season.rename(columns={
        "team_roll_season_pa_strikeout_rate_mean": "opp_team_roll_season_pa_strikeout_rate_mean",
        "team_roll_season_pa_strikeout_rate_std": "opp_team_roll_season_pa_strikeout_rate_std",
        "team_roll_season_pa_strikeout_rate_max": "opp_team_roll_season_pa_strikeout_rate_max",
    })[["batter_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate_mean",
        "opp_team_roll_season_pa_strikeout_rate_std", "opp_team_roll_season_pa_strikeout_rate_max"]],
    on=["batter_team_id", "gamepk"], how="left",
)
opp_team_volatility_short = rolling_stats.build_team_strikeout_volatility(pbp, batter_boxscore, window=SHORT_TEAM_WINDOW)
short_team_prefix = f"team_roll_last{SHORT_TEAM_WINDOW}g_pa_strikeout_rate"
pa_outcome = pa_outcome.merge(
    opp_team_volatility_short.rename(columns={
        f"{short_team_prefix}_mean": f"opp_{short_team_prefix}_mean",
        f"{short_team_prefix}_std": f"opp_{short_team_prefix}_std",
        f"{short_team_prefix}_max": f"opp_{short_team_prefix}_max",
    })[["batter_team_id", "gamepk", f"opp_{short_team_prefix}_mean",
        f"opp_{short_team_prefix}_std", f"opp_{short_team_prefix}_max"]],
    on=["batter_team_id", "gamepk"], how="left",
)

shrunk_whip = build_pitcher_shrunk_whip(
    pitcher_box_rolling, season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp), window="season", k=WHIP_SHRINKAGE_K,
)
pa_outcome = pa_outcome.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role", "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

FEATURE_COLS = [
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
FEATURE_COLS = [c for c in FEATURE_COLS if c in pa_outcome.columns]
print(f"Feature count: {len(FEATURE_COLS)} (v3/v6's set)")

model_df = pa_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(model_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

EARLY_STOP_SEASON = max(FIT_SEASONS)
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]
print(f"Core fit seasons: {CORE_FIT_SEASONS}  |  early-stop season: {EARLY_STOP_SEASON}  |  "
      f"scoring: {SCORE_SEASON} (never fit on; config's own test season {TEST_SEASON} also excluded)")

core_train_df = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_stop_df = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()


# ── 3. Fit v6's winning XGBoost config directly -- UNCHANGED, TEST_SEASON
# never appears in either core_train_df or early_stop_df above. Cached to
# disk (backtest/_model_cache/v6/) after the first fit -- CORE_FIT_SEASONS/
# EARLY_STOP_SEASON/hyperparameters/random_state are all fixed by
# config.yaml + FEATURE_COLS above, so this fit is identical regardless of
# which season gets scored afterward (2025 or 2026); no reason to pay for
# it twice. This is a reuse convenience for backtest scoring scripts only --
# NOT yet ROADMAP.md item 4's frozen production artifact, which is
# deliberately gated on the pending feature-redundancy audit (item 2). ─────
def fit_transform(fit_df, cols):
    fit_df = fit_df[cols].copy()
    n_cols = [c for c in cols if c in num_cols]
    c_cols = [c for c in cols if c in cat_cols]
    if n_cols:
        fit_df[n_cols] = fit_df[n_cols].apply(pd.to_numeric, errors="coerce")
    if c_cols:
        fit_df[c_cols] = fit_df[c_cols].astype(object).fillna(np.nan)
    return fit_df


CACHE_DIR = OUT_DIR / "_model_cache" / "v6"
CACHE_MODEL = CACHE_DIR / "xgb_model.json"
CACHE_PREPROC = CACHE_DIR / "preprocessing.pkl"

cache_hit = False
if CACHE_MODEL.exists() and CACHE_PREPROC.exists():
    cached = joblib.load(CACHE_PREPROC)
    if cached["feature_cols"] == FEATURE_COLS and cached["core_fit_seasons"] == CORE_FIT_SEASONS \
            and cached["early_stop_season"] == EARLY_STOP_SEASON:
        num_imp, cat_imp, enc = cached["num_imp"], cached["cat_imp"], cached["enc"]
        best_xgb = xgb.XGBClassifier()
        best_xgb.load_model(CACHE_MODEL)
        cache_hit = True
        print(f"\nLoaded cached v6 fit from {CACHE_DIR} (feature set + fit seasons match) -- skipping refit.")
        print(f"best_iteration={best_xgb.best_iteration}")
    else:
        print(f"\nCache at {CACHE_DIR} doesn't match current FEATURE_COLS/fit seasons -- refitting.")

if not cache_hit:
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
    best_xgb = xgb.XGBClassifier(
        n_estimators=2000, max_depth=2, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        random_state=42, verbosity=0, eval_metric="logloss", early_stopping_rounds=30,
    )
    best_xgb.fit(Xcore, y_core, eval_set=[(Xearly, y_early)], verbose=False)
    print(f"best_iteration={best_xgb.best_iteration}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    best_xgb.save_model(CACHE_MODEL)
    joblib.dump({
        "num_imp": num_imp, "cat_imp": cat_imp, "enc": enc,
        "feature_cols": FEATURE_COLS, "num_cols": num_cols, "cat_cols": cat_cols,
        "core_fit_seasons": CORE_FIT_SEASONS, "early_stop_season": EARLY_STOP_SEASON,
    }, CACHE_PREPROC)
    print(f"Cached fit to {CACHE_DIR} for reuse by future backtest scoring runs.")


def transform(df):
    df = df[FEATURE_COLS].copy()
    x_num = num_imp.transform(fit_transform(df, num_cols)) if num_cols else np.empty((len(df), 0))
    if cat_cols:
        x_cat = enc.transform(cat_imp.transform(fit_transform(df, cat_cols)))
    else:
        x_cat = np.empty((len(df), 0))
    return np.hstack([x_num, x_cat])


def score(df_feat):
    return best_xgb.predict_proba(transform(df_feat))[:, 1]


# ── 4. Expected batters faced, SCORE_SEASON 2026 SP starts, restricted to
# the 72 approved dates (the only change from run_xgboost_uncertainty.py's
# own section 4, beyond the VAL_SEASON -> SCORE_SEASON swap) ──────────────────
print("\nBuilding expected_batters_faced cascade...")
pitcher_start_pa_last_season = season_stats.build_pitcher_start_pa_stats(pbp)
league_avg_start_pa = season_stats.build_league_avg_start_pa(pitcher_start_pa_last_season)
team_avg_start_pa = season_stats.build_team_avg_start_pa(pitcher_start_pa_last_season)
pitcher_start_pa_this_season = game_context.build_pitcher_start_pa_this_season(pbp)

expected_pa = game_context.build_expected_batters_faced(
    pitcher_start_pa_last_season, pitcher_start_pa_this_season, team_avg_start_pa, league_avg_start_pa, k=PA_SHRINKAGE_K,
)
pitcher_starts_test = expected_pa[expected_pa["game_season"] == SCORE_SEASON].copy()

# expected_pa already carries game_date (from build_pitcher_start_pa_this_season's
# own groupby) -- filter on it directly rather than re-merging schedule, which
# would collide and get silently suffixed to game_date_x/game_date_y.
backtest_dates = pd.to_datetime(BACKTEST_DATES)
pitcher_starts_test = pitcher_starts_test[pitcher_starts_test["game_date"].isin(backtest_dates)].copy()
print(f"2026 SP starts with an expected_batters_faced estimate, on the {len(BACKTEST_DATES)} backtest dates: {len(pitcher_starts_test):,}")

pitcher_starts_test = pitcher_starts_test.merge(
    pitcher_role_season_stats.rename(columns={"expected_pitcher_key_id": "personId"})[
        ["personId", "game_season", "pitcher_last_season_pa_strikeout_rate"]
    ],
    on=["personId", "game_season"], how="left",
)
pitcher_starts_test = pitcher_starts_test.merge(
    pitcher_box_season_stats.rename(columns={"expected_pitcher_key_id": "personId"})[["personId", "game_season", "pitcher_last_season_whip"]],
    on=["personId", "game_season"], how="left",
)
pitcher_starts_test = pitcher_starts_test.merge(
    pitcher_box_rolling.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + box_rolling_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_test = pitcher_starts_test.merge(
    pitcher_pbp_rolling.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + pbp_rolling_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_test["pitcher_roll_season_avg_ip_per_game"] = (
    pitcher_starts_test["pitcher_roll_season_ip"] / pitcher_starts_test["pitcher_roll_season_games_n"].replace(0, np.nan)
)
pitcher_starts_test = pitcher_starts_test.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")
pitcher_starts_test = pitcher_starts_test.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId", "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]],
    on=["gamepk", "personId"], how="left",
)

schedule_teams = schedule[["gamepk", "home_id", "away_id"]].drop_duplicates("gamepk")
pitcher_starts_test = pitcher_starts_test.merge(schedule_teams, on="gamepk", how="left")
pitcher_starts_test["opp_team_id"] = np.where(
    pitcher_starts_test["pitcher_team_id"] == pitcher_starts_test["home_id"],
    pitcher_starts_test["away_id"], pitcher_starts_test["home_id"],
)

pitcher_starts_test = pitcher_starts_test.merge(
    pitcher_box_rolling3.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + box_rolling3_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_test = pitcher_starts_test.merge(
    pitcher_pbp_rolling3.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + pbp_rolling3_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_test[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_avg_ip_per_game"] = (
    pitcher_starts_test[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip"]
    / pitcher_starts_test[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n"].replace(0, np.nan)
)
pitcher_starts_test = pitcher_starts_test.merge(
    opp_team_volatility_season.rename(columns={
        "batter_team_id": "opp_team_id",
        "team_roll_season_pa_strikeout_rate_mean": "opp_team_roll_season_pa_strikeout_rate_mean",
        "team_roll_season_pa_strikeout_rate_std": "opp_team_roll_season_pa_strikeout_rate_std",
        "team_roll_season_pa_strikeout_rate_max": "opp_team_roll_season_pa_strikeout_rate_max",
    })[
        ["opp_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate_mean",
         "opp_team_roll_season_pa_strikeout_rate_std", "opp_team_roll_season_pa_strikeout_rate_max"]
    ],
    on=["opp_team_id", "gamepk"], how="left",
)
pitcher_starts_test = pitcher_starts_test.merge(
    opp_team_volatility_short.rename(columns={
        "batter_team_id": "opp_team_id",
        f"{short_team_prefix}_mean": f"opp_{short_team_prefix}_mean",
        f"{short_team_prefix}_std": f"opp_{short_team_prefix}_std",
        f"{short_team_prefix}_max": f"opp_{short_team_prefix}_max",
    })[
        ["opp_team_id", "gamepk", f"opp_{short_team_prefix}_mean",
         f"opp_{short_team_prefix}_std", f"opp_{short_team_prefix}_max"]
    ],
    on=["opp_team_id", "gamepk"], how="left",
)
throw_hand = pbp[["pitcher_id", "gamepk", "pitcher_throw_hand"]].drop_duplicates().rename(columns={"pitcher_id": "personId"})
pitcher_starts_test = pitcher_starts_test.merge(throw_hand, on=["personId", "gamepk"], how="left")
weather = game_info[["gamepk", "weather_condition", "weather_temp"]].drop_duplicates("gamepk")
pitcher_starts_test = pitcher_starts_test.merge(weather, on="gamepk", how="left")
pitcher_starts_test["expected_pitcher_role"] = "sp"


# ── 5. Expand to synthetic batter slots, attach batter + opp-team features
# (unchanged) ───────────────────────────────────────────────────────────────
print("Expanding to synthetic batter slots...")
batting_order = hp_pipeline._create_batting_order(batter_boxscore)
batter_team_lookup = batter_boxscore[["gamepk", "personId", "team_id"]].rename(
    columns={"personId": "batter_id", "team_id": "batter_team_id"}
).drop_duplicates()

slots = game_context.build_batter_slot_expansion(pitcher_starts_test, batting_order, max_slots=MAX_SLOTS)
slots = slots.merge(batter_team_lookup, on=["gamepk", "batter_id"], how="left")

bat_side = pbp[["batter_id", "gamepk", "batter_bat_side"]].drop_duplicates()
slots = slots.merge(bat_side, on=["batter_id", "gamepk"], how="left")
slots = slots.merge(batter_season_stats, on=["batter_id", "game_season"], how="left")
slots = slots.merge(
    batter_pbp_rolling[["batter_id", "gamepk", "batter_roll_season_pa_strikeout_rate"]],
    on=["batter_id", "gamepk"], how="left",
)
slots = slots.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")

print(f"{len(pitcher_starts_test):,} starts -> {len(slots):,} synthetic batter-slots "
      f"(mean {len(slots) / max(len(pitcher_starts_test), 1):.1f} slots/start)")


# ── 6. Score each slot, combine into a total-K distribution per start
# (unchanged) ───────────────────────────────────────────────────────────────
print("Scoring synthetic slots with XGBoost...")
slots["k_prob"] = score(slots)

print("Combining via exact Poisson-binomial...")
results = []
for (gamepk, person_id), grp in slots.groupby(["gamepk", "personId"]):
    probs = grp["k_prob"].to_numpy()
    pmf = poisson_binomial_pmf(list(probs))
    results.append({
        "gamepk": gamepk, "personId": person_id, "n_slots": len(probs),
        "predicted_mean_k": probs.sum(), "pmf": list(pmf),
    })
pred_df = pd.DataFrame(results)


# ── 7. Attach realized outcomes + pitcher name (for Epic 4's join to odds
# and its "did the over/under actually hit" sanity check) ─────────────────────
print("Attaching realized outcomes + pitcher names...")
role_lookup = season_stats._pitcher_role_lookup(pbp)[["gamepk", "pitcher_id", "pitcher_role"]].rename(columns={"pitcher_id": "personId"})
pitcher_box_tagged = pitcher_boxscore.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
).merge(
    role_lookup.assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str)),
    on=["gamepk", "personId"], how="left",
)
realized_k = pitcher_box_tagged[
    (pitcher_box_tagged["pitcher_role"] == "sp") & (pitcher_box_tagged["game_season"] == SCORE_SEASON)
][["personId", "gamepk", "k", "player_name"]].rename(columns={"k": "realized_k"})

pred_df["personId"] = pred_df["personId"].astype(str)
pred_df["gamepk"] = pred_df["gamepk"].astype(str)
pred_df = pred_df.merge(realized_k, on=["personId", "gamepk"], how="inner")
pred_df = pred_df.merge(
    pitcher_starts_test.assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str))[
        ["personId", "gamepk", "game_date", "expected_batters_faced", "expected_batters_faced_weight"]
    ],
    on=["personId", "gamepk"], how="left",
)
pred_df["residual"] = pred_df["predicted_mean_k"] - pred_df["realized_k"]

print(f"\n{'=' * 72}\nSCORED: SCORE_SEASON {SCORE_SEASON}, {len(BACKTEST_DATES)} backtest dates, {len(pred_df):,} SP starts\n{'=' * 72}")
print(f"Predicted mean K:  {pred_df['predicted_mean_k'].mean():.3f}")
print(f"Realized mean K:   {pred_df['realized_k'].mean():.3f}")
print(f"MAE (predicted mean vs. realized): {pred_df['residual'].abs().mean():.3f}")

out_path = OUT_DIR / "pred_df_test2026.parquet"
pred_df.to_parquet(out_path, index=False)
print(f"\nWrote {len(pred_df):,} rows to {out_path}")
