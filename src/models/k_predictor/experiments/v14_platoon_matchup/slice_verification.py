"""
v14 slice verification: does platoon_matchup actually close the same-hand
calibration gap the slice diagnostic found, or was v14's flat aggregate
PR-AUC/game-grain result (see v14_results.md) masking a real localized fix?

Aggregate metrics can't answer this: same-hand PAs are roughly half the
population, and the reported gap was a few tenths of a percentage point --
easily diluted to invisible in a whole-population PR-AUC or reliability
number. This script fits the v6 feature set (42 features) and the v14
feature set (v6's 42 + platoon_matchup) side by side on the IDENTICAL
train/val split and IDENTICAL XGBoost hyperparameters (max_depth=2,
learning_rate=0.03 -- the winning config both v6_tuned/train.py and
v14_platoon_matchup/train.py's own grid searches picked independently), so
any difference in the two models' predictions is attributable to the one
feature, not a confound from re-tuning.

Also produces the case-study leaderboards requested alongside this check:
highest-confidence predicted strikeouts, and the biggest misses in each
direction, with the pre-game-knowable stats that fed each prediction.

Run from src/models/k_predictor/ with: python experiments/v14_platoon_matchup/slice_verification.py
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
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip

BASE_DIR = Path(__file__).resolve().parent.parent.parent

WHIP_SHRINKAGE_K = 20.0
SHORT_PITCHER_WINDOW = 3
SHORT_TEAM_WINDOW = 5

# The config both v6_tuned and v14_platoon_matchup's own grid searches
# independently picked as best -- fixed here so the ONLY difference between
# the two models fit below is the platoon_matchup feature, not re-tuning.
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


# ── 2. Load data from S3 (identical to v14/train.py) ─────────────────────────
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
print("\nLoading game info...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/", TRAIN_SEASONS,
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


# ── 3. Build PA-grain DataFrame (identical to v14/train.py) ──────────────────
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

team_batter_rolling = rolling_stats.build_team_batter_strikeout_rolling_feats(
    pbp, batter_boxscore, window="season"
)
opp_team_rolling = team_batter_rolling.rename(columns={
    "team_roll_season_pa_strikeout_rate": "opp_team_roll_season_pa_strikeout_rate",
})[["batter_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")

pitching_team_rolling = team_batter_rolling.rename(columns={
    "batter_team_id": "pitcher_team_id",
    "team_roll_season_pa_strikeout_rate": "pitching_team_roll_season_pa_strikeout_rate",
})[["pitcher_team_id", "gamepk", "pitching_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")

pitcher_box_rolling3 = rolling_stats.build_pitcher_rolling_stats_all_roles(
    pitcher_boxscore, pbp, window=SHORT_PITCHER_WINDOW,
)
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
    pitcher_box_rolling, season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp),
    window="season", k=WHIP_SHRINKAGE_K,
)
pa_outcome = pa_outcome.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})[
        ["gamepk", "expected_pitcher_key_id", "expected_pitcher_role", "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]
    ],
    on=["gamepk", "expected_pitcher_key_id", "expected_pitcher_role"], how="left",
)


# ── 4. Feature sets: v6 (42) vs v14 (v6 + platoon_matchup) ────────────────────
FEATURE_COLS_V6 = [
    "expected_pitcher_role", "pitcher_throw_hand", "batter_bat_side",
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
FEATURE_COLS_V6 = [c for c in FEATURE_COLS_V6 if c in pa_outcome.columns]
FEATURE_COLS_V14 = FEATURE_COLS_V6 + ["platoon_matchup"]

EXTRA_DISPLAY_COLS = ["pitcher_name", "batter_name"]
GAME_GRAIN_KEY_COLS = ["gamepk", "batter_id"]

display_df = pa_outcome[
    FEATURE_COLS_V14 + [TARGET, DATE_COL, "game_season"] + GAME_GRAIN_KEY_COLS + EXTRA_DISPLAY_COLS
].copy()
display_df["game_season"] = display_df["game_season"].astype(int)

train_df = display_df[display_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = display_df[display_df["game_season"] == VAL_SEASON].copy()

print(f"\nFit seasons: {FIT_SEASONS}  |  Val season: {VAL_SEASON}  ({len(val_df):,} PAs)")


# ── 5. Fit v6-feature and v14-feature XGBoost, same fixed config ─────────────
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


print("\nFitting v6 feature set (42 features, no platoon_matchup)...")
val_df["pred_v6"] = fit_and_predict(FEATURE_COLS_V6, train_df, val_df)

print("Fitting v14 feature set (v6's 42 + platoon_matchup)...")
val_df["pred_v14"] = fit_and_predict(FEATURE_COLS_V14, train_df, val_df)


# ── 6. Calibration by platoon_matchup slice ───────────────────────────────────
print("\n" + "#" * 78)
print("# CALIBRATION BY PLATOON_MATCHUP SLICE — v6 (no feature) vs v14 (+ feature)")
print("#" * 78)

SLICE_MIN_N = 200
slice_summary = []
for slice_name in ["ALL PAs", "same_hand", "opposite_hand", "switch_hitter"]:
    subset = val_df if slice_name == "ALL PAs" else val_df[val_df["platoon_matchup"] == slice_name]
    if len(subset) == 0:
        continue
    print(f"\n{'=' * 78}\nSLICE: {slice_name}  (n={len(subset):,})\n{'=' * 78}")
    for model_name, pred_col in [("v6 (no platoon_matchup)", "pred_v6"), ("v14 (+ platoon_matchup)", "pred_v14")]:
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
print("# SUMMARY — reliability/resolution delta (v14 minus v6; reliability: lower=better, resolution: higher=better)")
print("#" * 78)
pivot = summary_df.pivot(index="slice", columns="model", values=["reliability", "resolution"])
for slice_name in ["ALL PAs", "same_hand", "opposite_hand", "switch_hitter"]:
    if slice_name not in pivot.index:
        continue
    rel_v6 = pivot.loc[slice_name, ("reliability", "v6 (no platoon_matchup)")]
    rel_v14 = pivot.loc[slice_name, ("reliability", "v14 (+ platoon_matchup)")]
    res_v6 = pivot.loc[slice_name, ("resolution", "v6 (no platoon_matchup)")]
    res_v14 = pivot.loc[slice_name, ("resolution", "v14 (+ platoon_matchup)")]
    print(f"  {slice_name:<16} reliability {rel_v6:.5f} -> {rel_v14:.5f} ({rel_v14 - rel_v6:+.5f})   "
          f"resolution {res_v6:.5f} -> {res_v14:.5f} ({res_v14 - res_v6:+.5f})")


# ── 7. Paired bootstrap on the same_hand slice — is the Brier delta real? ────
print("\n" + "#" * 78)
print("# PAIRED BOOTSTRAP — same_hand slice, Brier score (v6 - v14; positive = v14 better)")
print("#" * 78)
same_hand = val_df[val_df["platoon_matchup"] == "same_hand"]
if len(same_hand) >= SLICE_MIN_N:
    y = same_hand[TARGET].to_numpy(dtype=np.float64)
    p6 = same_hand["pred_v6"].to_numpy(dtype=np.float64)
    p14 = same_hand["pred_v14"].to_numpy(dtype=np.float64)
    brier6_row = (p6 - y) ** 2
    brier14_row = (p14 - y) ** 2
    point_delta = float(brier6_row.mean() - brier14_row.mean())

    rng = np.random.default_rng(42)
    n = len(y)
    idx = rng.integers(0, n, size=(1000, n))
    boot_delta = brier6_row[idx].mean(axis=1) - brier14_row[idx].mean(axis=1)
    lo, hi = np.percentile(boot_delta, [2.5, 97.5])
    print(f"  n={n:,}  Brier(v6)={brier6_row.mean():.5f}  Brier(v14)={brier14_row.mean():.5f}")
    print(f"  delta={point_delta:+.5f}  95% CI=[{lo:+.5f}, {hi:+.5f}]  "
          f"{'REAL improvement' if lo > 0 else ('REAL regression' if hi < 0 else 'CI includes zero — not distinguishable from noise')}")
else:
    print(f"  same_hand slice too small (n={len(same_hand)} < {SLICE_MIN_N}) for a bootstrap check")


# ── 8. Case studies — v14's most confident predictions and biggest misses ───
LEADING_STATS = [
    "platoon_matchup", "expected_times_through_order",
    "pitcher_roll_last3g_k_rate", "pitcher_roll_last3g_whip", "pitcher_roll_last3g_command_swinging_strike_rate",
    "pitcher_roll_season_k_rate", "pitcher_last_season_pa_strikeout_rate",
    "batter_last_season_pa_strikeout_rate", "batter_roll_season_pa_strikeout_rate",
]
LEADING_STATS = [c for c in LEADING_STATS if c in val_df.columns]
DISPLAY_COLS = ["game_date", "pitcher_name", "batter_name", TARGET, "pred_v14"] + LEADING_STATS

print("\n" + "#" * 78)
print("# MOST CONFIDENT PREDICTED STRIKEOUTS (top 15 by pred_v14) — v14 model, val season "
      f"{VAL_SEASON}")
print("#" * 78)
top_confident = val_df.sort_values("pred_v14", ascending=False).head(15)
print(top_confident[DISPLAY_COLS].to_string(index=False))
hit_rate_in_top15 = top_confident[TARGET].mean()
print(f"\n  Actual strikeout rate among these 15: {hit_rate_in_top15:.2f}")

print("\n" + "#" * 78)
print("# BIGGEST MISSES — predicted HIGH, no strikeout happened (top 10 false positives)")
print("#" * 78)
false_pos = val_df[val_df[TARGET] == 0].sort_values("pred_v14", ascending=False).head(10)
print(false_pos[DISPLAY_COLS].to_string(index=False))

print("\n" + "#" * 78)
print("# BIGGEST MISSES — predicted LOW, strikeout happened anyway (top 10 false negatives)")
print("#" * 78)
false_neg = val_df[val_df[TARGET] == 1].sort_values("pred_v14", ascending=True).head(10)
print(false_neg[DISPLAY_COLS].to_string(index=False))

print("\nDone.")
