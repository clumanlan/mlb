"""
v15 slice verification: do estimated_team_pa_position and
pitcher_projected_pitches_before_pa actually help, or was v15's flat
aggregate PR-AUC/game-grain result (see run_output_fixed.log) masking a real
localized effect -- same question asked of platoon_matchup in v14's own
slice_verification.py, and the answer there was "yes, real, just invisible
in aggregate."

The sharpest slice here isn't platoon_matchup, it's expected_times_through_order
(TTO) itself: TTO is capped at 3, and goes to NaN entirely once a PA is
estimated to be past the starter's own average depth -- at that point the OLD
feature set has ZERO information about how deep into the game this PA
actually is (a NaN gets silently imputed to the training median). That's
exactly the population the new uncapped PA-position and projected-pitch-count
features were built to describe. This script fits the v14 feature set (43
features, includes platoon_matchup) and the v15 feature set (v14 + the 2
workload features, with the corrected starter-scoped pace join) side by side
on the IDENTICAL train/val split and XGBoost hyperparameters, then compares
calibration/discrimination by TTO bucket, with a paired bootstrap on the
TTO=NaN slice specifically.

Run from src/models/k_predictor/ with: python experiments/v15_pa_workload/slice_verification.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import expected_role
from models.hit_predictor.utils.eval import get_calibration_df, murphy_decomposition

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import (
    build_pitcher_shrunk_whip, build_pitcher_projected_workload,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

WHIP_SHRINKAGE_K = 20.0
SHORT_PITCHER_WINDOW = 3
SHORT_TEAM_WINDOW = 5

# The config v6_tuned/v14/v15's own grid searches all independently picked
# as best (or within noise of it) -- fixed here so the ONLY difference
# between the two models fit below is the two new features, not re-tuning.
XGB_BEST_CONFIG = {"max_depth": 2, "learning_rate": 0.03}

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


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

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))


# ── 2. Load data from S3 (identical to v15/train.py) ─────────────────────────
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
pbp = read_parquet_seasons("s3://{bucket}/processed_data/prepared/playbyplay/{season}/", TRAIN_SEASONS, chunked=True)
print("\nLoading schedule...")
schedule = read_parquet_seasons("s3://{bucket}/processed_data/games/schedule/{season}/", all_boxscore_seasons)
print("\nLoading game info...")
game_info = read_parquet_seasons("s3://{bucket}/processed_data/games/game_info/{season}/", TRAIN_SEASONS)
print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/", all_boxscore_seasons)
print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons("s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/", all_boxscore_seasons)
print("\nLoading player info...")
player_info = wr.s3.read_parquet(path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session)


# ── 3. Build PA-grain DataFrame (identical to v15/train.py, incl. pace fix) ──
print("\nBuilding PA-grain DataFrame...")

schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome_strikeout(pbp, batter_boxscore, game_info, schedule)
pa_outcome["game_season"] = pa_outcome["game_date"].dt.year

pitcher_start_depth_stats = season_stats.build_pitcher_start_depth_stats(pbp)
league_avg_start_depth = season_stats.build_league_avg_start_depth(pitcher_start_depth_stats)
pa_outcome = expected_role.assign_expected_pitcher_role(
    pa_outcome, pitcher_start_depth_stats, league_avg_start_depth
)

pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_pa_strikeout_rate"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})
pitcher_box_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_whip"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})
batter_season_stats = season_stats.build_pbp_batter_feats(pbp)[
    ["batter_id", "game_season", "batter_last_season_pa_strikeout_rate"]
]
pa_outcome = pa_outcome.merge(
    pitcher_role_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
pa_outcome = pa_outcome.merge(
    pitcher_box_season_stats, on=["game_season", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)
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

# Corrected pace join (the bug fix): starter's OWN trailing-3g pace, keyed
# on starting_pitcher_id, not the expected-role-gated merge above.
starter_pace_col = f"starting_pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean"
sp_only_pace = pitcher_pbp_rolling3[pitcher_pbp_rolling3["pitcher_role"] == "sp"][
    ["pitcher_key_id", "gamepk", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean"]
].rename(columns={
    "pitcher_key_id": "starting_pitcher_id",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean": starter_pace_col,
})
pa_outcome["starting_pitcher_id"] = pa_outcome["starting_pitcher_id"].astype(str)
pa_outcome["gamepk"] = pa_outcome["gamepk"].astype(str)
sp_only_pace["starting_pitcher_id"] = sp_only_pace["starting_pitcher_id"].astype(str)
sp_only_pace["gamepk"] = sp_only_pace["gamepk"].astype(str)
pa_outcome = pa_outcome.merge(sp_only_pace, on=["gamepk", "starting_pitcher_id"], how="left")
pa_outcome = build_pitcher_projected_workload(pa_outcome, pace_col=starter_pace_col)

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
    pitcher_box_rolling, season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp),
    window="season", k=WHIP_SHRINKAGE_K,
)
pa_outcome = pa_outcome.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role", "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)

is_switch = (pa_outcome['batter_bat_side'] == 'S').fillna(False)
is_same_hand = (pa_outcome['batter_bat_side'] == pa_outcome['pitcher_throw_hand']).fillna(False)
pa_outcome['platoon_matchup'] = 'opposite_hand'
pa_outcome.loc[is_same_hand, 'platoon_matchup'] = 'same_hand'
pa_outcome.loc[is_switch, 'platoon_matchup'] = 'switch_hitter'
pa_outcome.loc[pa_outcome['batter_bat_side'].isna() | pa_outcome['pitcher_throw_hand'].isna(), 'platoon_matchup'] = np.nan


# ── 4. Feature sets: v14 (43) vs v15 (v14 + 2 workload features) ─────────────
FEATURE_COLS_V14 = [
    "expected_pitcher_role", "pitcher_throw_hand", "batter_bat_side", "platoon_matchup",
    "pitcher_last_season_pa_strikeout_rate", "pitcher_last_season_whip", "batter_last_season_pa_strikeout_rate",
    "pitcher_roll_season_ip", "pitcher_roll_season_whip", "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate", "pitcher_roll_season_hr_rate", "pitcher_roll_season_games_n",
    "pitcher_roll_season_avg_ip_per_game", "pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate",
    "batter_roll_season_pa_strikeout_rate", "opp_team_roll_season_pa_strikeout_rate",
    "pitching_team_roll_season_pa_strikeout_rate",
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
FEATURE_COLS_V14 = [c for c in FEATURE_COLS_V14 if c in pa_outcome.columns]
FEATURE_COLS_V15 = FEATURE_COLS_V14 + ["estimated_team_pa_position", "pitcher_projected_pitches_before_pa"]

EXTRA_DISPLAY_COLS = ["pitcher_name", "batter_name"]
GAME_GRAIN_KEY_COLS = ["gamepk", "batter_id"]

display_df = pa_outcome[
    FEATURE_COLS_V15 + [TARGET, DATE_COL, "game_season"] + GAME_GRAIN_KEY_COLS + EXTRA_DISPLAY_COLS
].copy()
display_df["game_season"] = display_df["game_season"].astype(int)

train_df = display_df[display_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = display_df[display_df["game_season"] == VAL_SEASON].copy()

print(f"\nFit seasons: {FIT_SEASONS}  |  Val season: {VAL_SEASON}  ({len(val_df):,} PAs)")


# ── 5. Fit v14-feature and v15-feature XGBoost, same fixed config ────────────
def fit_and_predict(feature_cols, train_df, val_df):
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    X_tr = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    if num_cols:
        X_tr[num_cols] = X_tr[num_cols].apply(pd.to_numeric, errors="coerce")
        X_val[num_cols] = X_val[num_cols].apply(pd.to_numeric, errors="coerce")
    if cat_cols:
        X_tr[cat_cols] = X_tr[cat_cols].astype(object).fillna(np.nan)
        X_val[cat_cols] = X_val[cat_cols].astype(object).fillna(np.nan)

    num_imp = SimpleImputer(strategy="median")
    Xtr_num = num_imp.fit_transform(X_tr[num_cols]) if num_cols else np.empty((len(X_tr), 0))
    Xval_num = num_imp.transform(X_val[num_cols]) if num_cols else np.empty((len(X_val), 0))

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        Xtr_cat_imp = cat_imp.fit_transform(X_tr[cat_cols])
        Xval_cat_imp = cat_imp.transform(X_val[cat_cols])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr_cat = enc.fit_transform(Xtr_cat_imp)
        Xval_cat = enc.transform(Xval_cat_imp)
    else:
        Xtr_cat = np.empty((len(X_tr), 0))
        Xval_cat = np.empty((len(X_val), 0))

    Xtr = np.hstack([Xtr_num, Xtr_cat])
    Xval = np.hstack([Xval_num, Xval_cat])

    model = xgb.XGBClassifier(
        n_estimators=2000, max_depth=XGB_BEST_CONFIG["max_depth"], learning_rate=XGB_BEST_CONFIG["learning_rate"],
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        random_state=42, verbosity=0, eval_metric="logloss",
    )
    model.fit(Xtr, train_df[TARGET])
    return model.predict_proba(Xval)[:, 1]


print("\nFitting v14 feature set (43 features)...")
val_df["pred_v14"] = fit_and_predict(FEATURE_COLS_V14, train_df, val_df)

print("Fitting v15 feature set (v14's 43 + 2 workload features)...")
val_df["pred_v15"] = fit_and_predict(FEATURE_COLS_V15, train_df, val_df)


# ── 6. Calibration by TTO bucket (the sharpest slice for this hypothesis) ────
val_df["tto_bucket"] = val_df["expected_times_through_order"].apply(
    lambda x: "TTO=NaN (expected bullpen)" if pd.isna(x) else f"TTO={int(x)}"
)

print("\n" + "#" * 78)
print("# CALIBRATION BY TIMES-THROUGH-ORDER BUCKET — v14 (no workload feats) vs v15 (+ workload feats)")
print("#" * 78)

SLICE_MIN_N = 200
slice_summary = []
for slice_name in ["ALL PAs", "TTO=1", "TTO=2", "TTO=3", "TTO=NaN (expected bullpen)"]:
    subset = val_df if slice_name == "ALL PAs" else val_df[val_df["tto_bucket"] == slice_name]
    if len(subset) == 0:
        continue
    print(f"\n{'=' * 78}\nSLICE: {slice_name}  (n={len(subset):,})\n{'=' * 78}")
    for model_name, pred_col in [("v14 (no workload feats)", "pred_v14"), ("v15 (+ workload feats)", "pred_v15")]:
        cal_df = get_calibration_df(subset[TARGET], subset[pred_col], n_bins=5, min_n=SLICE_MIN_N)
        decomp = murphy_decomposition(subset[TARGET], cal_df)
        print(f"\n  -- {model_name} --")
        print(cal_df.to_string(index=False).replace("\n", "\n  "))
        print(f"  reliability={decomp['reliability']:.5f}  resolution={decomp['resolution']:.5f}  "
              f"brier_reconstructed={decomp['brier_reconstructed']:.5f}")
        slice_summary.append({
            "slice": slice_name, "model": model_name, "n": len(subset),
            "reliability": decomp["reliability"], "resolution": decomp["resolution"],
        })

summary_df = pd.DataFrame(slice_summary)
print("\n" + "#" * 78)
print("# SUMMARY — reliability/resolution delta (v15 minus v14; reliability: lower=better, resolution: higher=better)")
print("#" * 78)
pivot = summary_df.pivot(index="slice", columns="model", values=["reliability", "resolution"])
for slice_name in ["ALL PAs", "TTO=1", "TTO=2", "TTO=3", "TTO=NaN (expected bullpen)"]:
    if slice_name not in pivot.index:
        continue
    rel_v14 = pivot.loc[slice_name, ("reliability", "v14 (no workload feats)")]
    rel_v15 = pivot.loc[slice_name, ("reliability", "v15 (+ workload feats)")]
    res_v14 = pivot.loc[slice_name, ("resolution", "v14 (no workload feats)")]
    res_v15 = pivot.loc[slice_name, ("resolution", "v15 (+ workload feats)")]
    print(f"  {slice_name:<28} reliability {rel_v14:.5f} -> {rel_v15:.5f} ({rel_v15 - rel_v14:+.5f})   "
          f"resolution {res_v14:.5f} -> {res_v15:.5f} ({res_v15 - res_v14:+.5f})")


# ── 7. Paired bootstrap — TTO=NaN slice (zero info under old features) ──────
print("\n" + "#" * 78)
print("# PAIRED BOOTSTRAP — TTO=NaN slice, Brier score (v14 - v15; positive = v15 better)")
print("#" * 78)
tto_nan = val_df[val_df["tto_bucket"] == "TTO=NaN (expected bullpen)"]
if len(tto_nan) >= SLICE_MIN_N:
    y = tto_nan[TARGET].to_numpy(dtype=np.float64)
    p14 = tto_nan["pred_v14"].to_numpy(dtype=np.float64)
    p15 = tto_nan["pred_v15"].to_numpy(dtype=np.float64)
    brier14_row = (p14 - y) ** 2
    brier15_row = (p15 - y) ** 2
    point_delta = float(brier14_row.mean() - brier15_row.mean())

    rng = np.random.default_rng(42)
    n = len(y)
    idx = rng.integers(0, n, size=(1000, n))
    boot_delta = brier14_row[idx].mean(axis=1) - brier15_row[idx].mean(axis=1)
    lo, hi = np.percentile(boot_delta, [2.5, 97.5])
    print(f"  n={n:,}  Brier(v14)={brier14_row.mean():.5f}  Brier(v15)={brier15_row.mean():.5f}")
    print(f"  delta={point_delta:+.5f}  95% CI=[{lo:+.5f}, {hi:+.5f}]  "
          f"{'REAL improvement' if lo > 0 else ('REAL regression' if hi < 0 else 'CI includes zero — not distinguishable from noise')}")
else:
    print(f"  TTO=NaN slice too small (n={len(tto_nan)} < {SLICE_MIN_N}) for a bootstrap check")

# ── 8. Paired bootstrap — TTO=3 slice (also compressed vs. old features) ────
print("\n" + "#" * 78)
print("# PAIRED BOOTSTRAP — TTO=3 slice, Brier score (v14 - v15; positive = v15 better)")
print("#" * 78)
tto_3 = val_df[val_df["tto_bucket"] == "TTO=3"]
if len(tto_3) >= SLICE_MIN_N:
    y = tto_3[TARGET].to_numpy(dtype=np.float64)
    p14 = tto_3["pred_v14"].to_numpy(dtype=np.float64)
    p15 = tto_3["pred_v15"].to_numpy(dtype=np.float64)
    brier14_row = (p14 - y) ** 2
    brier15_row = (p15 - y) ** 2
    point_delta = float(brier14_row.mean() - brier15_row.mean())

    rng = np.random.default_rng(42)
    n = len(y)
    idx = rng.integers(0, n, size=(1000, n))
    boot_delta = brier14_row[idx].mean(axis=1) - brier15_row[idx].mean(axis=1)
    lo, hi = np.percentile(boot_delta, [2.5, 97.5])
    print(f"  n={n:,}  Brier(v14)={brier14_row.mean():.5f}  Brier(v15)={brier15_row.mean():.5f}")
    print(f"  delta={point_delta:+.5f}  95% CI=[{lo:+.5f}, {hi:+.5f}]  "
          f"{'REAL improvement' if lo > 0 else ('REAL regression' if hi < 0 else 'CI includes zero — not distinguishable from noise')}")
else:
    print(f"  TTO=3 slice too small (n={len(tto_3)} < {SLICE_MIN_N}) for a bootstrap check")

print("\nDone.")
