"""
Builds a PA-grain scored dataset for k_predictor v6 (the tuned XGBoost
cached at src/models/k_predictor/backtest/_model_cache/v6/) on the held-out
TEST_SEASON (config.yaml's test_season, 2025) for ad hoc uncertainty / error
analysis via the shared model_dashboard package (see CLAUDE.md's "Model
Layer" -- dashboard_spec.md).

Not part of the Lambda pipeline or TDD scope -- a one-off data-generation
script, same convention as backtest/score_2026_test_dates.py (copy-adapted,
not shared as a library). Sections 1-3 below are copied near-verbatim from
that script (whose own header note says "Sections 1-3 unchanged in
substance" vs. score_2025_test_dates.py) -- this script stops after scoring
PA-grain rows rather than continuing to the pitcher-game aggregation
cascade (poisson_binomial_pmf etc.), and reads only TRAIN_SEASONS (2025 is
already in there) since there's no need to score 2026 here.

Run from repo root with:
    PYTHONPATH=src python src/models/k_predictor/diagnostics/build_data.py
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

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip

K_PREDICTOR_DIR = Path(__file__).resolve().parent.parent
WHIP_SHRINKAGE_K = 20.0
SHORT_PITCHER_WINDOW = 3
SHORT_TEAM_WINDOW = 5

OUT_DIR = Path(__file__).parent


# ── 1. Config + load (copied from backtest/score_2026_test_dates.py section
# 1, minus the SCORE_SEASON extension -- TEST_SEASON is already in
# TRAIN_SEASONS, so no extra seasons need reading) ────────────────────────────
with open(K_PREDICTOR_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET, REGION = cfg["bucket"], cfg["region"]
TRAIN_SEASONS, FEATURE_SEASONS = cfg["train_seasons"], cfg["feature_seasons"]
TARGET, DATE_COL = cfg["target_column"], cfg["date_column"]
TEST_SEASON, VAL_SEASON = cfg["test_season"], cfg["val_season"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))
READ_SEASONS = sorted(set(TRAIN_SEASONS))
READ_BOXSCORE_SEASONS = sorted(set(all_boxscore_seasons))


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


# ── 2. Build PA-grain frame + v3/v6's full 42-feature set (copied from
# backtest/score_2026_test_dates.py section 2, unchanged) ─────────────────────
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

# Identity/context columns kept alongside FEATURE_COLS purely for dashboard
# display (dropdowns, top-loss tables) -- never fed to the model.
IDENTITY_COLS = [
    "gamepk", "batter_id", "batter_name", "batter_team_name",
    "pitcher_id", "pitcher_name", "platoon_matchup",
    "batting_order", "estimated_team_pa_position", "venue_id",
]
IDENTITY_COLS = [c for c in IDENTITY_COLS if c in pa_outcome.columns]

model_df = pa_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season"] + IDENTITY_COLS].copy()
model_df["game_season"] = model_df["game_season"].astype(int)

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(model_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

EARLY_STOP_SEASON = max(FIT_SEASONS)
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]
print(f"Core fit seasons: {CORE_FIT_SEASONS}  |  early-stop season: {EARLY_STOP_SEASON}  |  "
      f"scoring: TEST_SEASON={TEST_SEASON} (never fit on)")

core_train_df = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_stop_df = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()


# ── 3. Load v6's cached fit (copied from backtest/score_2026_test_dates.py
# section 3) -- falls back to refitting only if the cache doesn't match
# this script's FEATURE_COLS/fit seasons. ──────────────────────────────────────
def fit_transform(fit_df, cols):
    fit_df = fit_df[cols].copy()
    n_cols = [c for c in cols if c in num_cols]
    c_cols = [c for c in cols if c in cat_cols]
    if n_cols:
        fit_df[n_cols] = fit_df[n_cols].apply(pd.to_numeric, errors="coerce")
    if c_cols:
        fit_df[c_cols] = fit_df[c_cols].astype(object).fillna(np.nan)
    return fit_df


CACHE_DIR = K_PREDICTOR_DIR / "backtest" / "_model_cache" / "v6"
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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    best_xgb.save_model(CACHE_MODEL)
    joblib.dump({
        "num_imp": num_imp, "cat_imp": cat_imp, "enc": enc,
        "feature_cols": FEATURE_COLS, "num_cols": num_cols, "cat_cols": cat_cols,
        "core_fit_seasons": CORE_FIT_SEASONS, "early_stop_season": EARLY_STOP_SEASON,
    }, CACHE_PREPROC)


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


# ── 4. Score every TEST_SEASON PA row (no odds-date restriction -- this is
# for error/uncertainty analysis, not an odds-matched backtest) and write
# the joined CSV the shared dashboard reads. ───────────────────────────────────
test_df = model_df[model_df["game_season"] == TEST_SEASON].copy()
print(f"\nScoring {len(test_df):,} PA rows from TEST_SEASON={TEST_SEASON}...")
test_df["pred_prob"] = score(test_df)

out_cols = IDENTITY_COLS + [DATE_COL, "game_season"] + FEATURE_COLS + [TARGET, "pred_prob"]
test_df = test_df[out_cols]

out_path = OUT_DIR / "data.csv"
test_df.to_csv(out_path, index=False)
print(f"\nWrote {len(test_df):,} rows x {len(out_cols)} cols to {out_path}")
