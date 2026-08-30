"""
k_predictor: within-game residual correlation check -- does one plate
appearance's outcome (relative to the model's own predicted probability)
predict a LATER plate appearance's outcome in the SAME start?

Run from src/models/k_predictor/ with:
    python experiments/count_distribution_check/within_game_correlation_check.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2), UNLESS a
cached model_df from a prior run already exists under _model_cache/ (see
Section 2) -- in that case this skips S3 entirely and goes straight to the fit.
"""
# ---------------------------------------------------------------------------- #
#                                    SUMMARY                                   #
# ---------------------------------------------------------------------------- #
# run_xgboost_uncertainty.py found the predicted total-K distribution has thin
# tails -- it can't foresee an unusually dominant start, and run_naive_batter_
# uncertainty.py showed this isn't fixable by better per-slot features, because
# it's structural: the Poisson-binomial combinator treats every plate
# appearance in a start as INDEPENDENT of every other one in the same start.
# That's a real, checkable assumption. If a pitcher's actual "stuff" varies
# start-to-start (a shared, persistent cause behind every PA in that game --
# not a batter-to-batter psychological effect, a pitcher-level one), residuals
# WITHIN a start should be positively correlated: over-performing the model's
# per-slot prediction early in a start should predict over-performing later in
# that SAME start, beyond what the pre-game features already captured.
#
# This is a pure diagnostic on real 2024 PA-grain rows, reusing v6's exact
# fitted model (same feature build + single best XGBoost config as
# run_xgboost_uncertainty.py) -- no new features, no architecture change. For
# each 2024 SP start with >=10 real plate appearances against him, split those
# PAs by ODD/EVEN position in real chronological order (play_id) rather than
# first-half/second-half, since an interleaved split cancels out smooth
# within-game trends (fatigue, times-through-order creep) that would otherwise
# create a spurious correlation unrelated to a persistent "quality today"
# effect. For each start, sum (actual - predicted) over the odd PAs and over
# the even PAs, then correlate those two sums across all starts.
#
# CACHING: the S3 load + 42-feature build is the expensive, deterministic part
# of every k_predictor diagnostic script built this session (~20-30 min each
# time, paid three separate times before this) -- the actual model fit itself
# takes seconds. This script caches the fully-built model_df + FEATURE_COLS to
# _model_cache/ after building them once, and loads that cache on any later
# run instead of rebuilding from S3. Any future k_predictor v6-feature-set
# diagnostic can reuse this same cache directly.
# ---------------------------------------------------------------------------- #
import json
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import expected_role
from models.hit_predictor.processing.features import game_context

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WHIP_SHRINKAGE_K = 20.0
SHORT_PITCHER_WINDOW = 3
SHORT_TEAM_WINDOW = 5
MIN_PA_PER_START = 10

OUT_DIR = Path(__file__).parent / "within_game_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(__file__).parent / "_model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DF_CACHE = CACHE_DIR / "model_df_v6.parquet"
FEATURE_COLS_CACHE = CACHE_DIR / "feature_cols_v6.json"


# ── 1. Config ───────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET, REGION = cfg["bucket"], cfg["region"]
TRAIN_SEASONS, FEATURE_SEASONS = cfg["train_seasons"], cfg["feature_seasons"]
TARGET, DATE_COL = cfg["target_column"], cfg["date_column"]
TEST_SEASON, VAL_SEASON = cfg["test_season"], cfg["val_season"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))


# ── 2. Build PA-grain frame + v3/v6's full 42-feature set -- OR load the ──────
# cached one from a prior run of this or a sibling script. ────────────────────
if MODEL_DF_CACHE.exists() and FEATURE_COLS_CACHE.exists():
    print(f"\nFound cached model_df at {MODEL_DF_CACHE} -- loading instead of hitting S3.")
    model_df = pd.read_parquet(MODEL_DF_CACHE)
    with open(FEATURE_COLS_CACHE) as f:
        FEATURE_COLS = json.load(f)
    print(f"Feature count: {len(FEATURE_COLS)} (v3/v6's set, from cache)")
else:
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

    # Extra columns vs. run_xgboost_uncertainty.py: pitcher_id + play_id, needed
    # here (and cached for reuse) to group PAs by real start and order them
    # chronologically -- not needed there since that script never scores real
    # PA rows directly.
    model_df = pa_outcome[FEATURE_COLS + [TARGET, DATE_COL, "game_season", "gamepk", "pitcher_id", "play_id"]].copy()
    model_df["game_season"] = model_df["game_season"].astype(int)

    model_df.to_parquet(MODEL_DF_CACHE, index=False)
    with open(FEATURE_COLS_CACHE, "w") as f:
        json.dump(FEATURE_COLS, f)
    print(f"Cached model_df ({len(model_df):,} rows) + FEATURE_COLS to {CACHE_DIR} for future reuse")


num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(model_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

EARLY_STOP_SEASON = max(FIT_SEASONS)
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]
print(f"Core fit seasons: {CORE_FIT_SEASONS}  |  early-stop season: {EARLY_STOP_SEASON}  |  val: {VAL_SEASON}")

core_train_df = model_df[model_df["game_season"].isin(CORE_FIT_SEASONS)].copy()
early_stop_df = model_df[model_df["game_season"] == EARLY_STOP_SEASON].copy()
val_df = model_df[model_df["game_season"] == VAL_SEASON].copy()


# ── 3. Fit v6's winning XGBoost config directly (identical to ─────────────────
# run_xgboost_uncertainty.py) ───────────────────────────────────────────────────
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
best_xgb = xgb.XGBClassifier(
    n_estimators=2000, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, verbosity=0, eval_metric="logloss", early_stopping_rounds=30,
)
best_xgb.fit(Xcore, y_core, eval_set=[(Xearly, y_early)], verbose=False)
print(f"best_iteration={best_xgb.best_iteration}")


# ── 4. Score real 2024 PA rows, split each start odd/even by real order ───────
print("\nScoring real 2024 plate appearances...")
val_df["pred_prob"] = best_xgb.predict_proba(transform(val_df))[:, 1]
val_df["residual"] = val_df[TARGET].astype(float) - val_df["pred_prob"]

val_df = val_df.sort_values(["gamepk", "pitcher_id", "play_id"])
val_df["pa_rank"] = val_df.groupby(["gamepk", "pitcher_id"]).cumcount()  # 0-indexed, real chronological order
val_df["half"] = np.where(val_df["pa_rank"] % 2 == 0, "even", "odd")

start_counts = val_df.groupby(["gamepk", "pitcher_id"]).size()
eligible_starts = start_counts[start_counts >= MIN_PA_PER_START].index
val_df_idx = val_df.set_index(["gamepk", "pitcher_id"])
val_df = val_df_idx[val_df_idx.index.isin(eligible_starts)].reset_index()
print(f"{len(eligible_starts):,} 2024 starts with >= {MIN_PA_PER_START} real PAs against the starter")

split_sums = (
    val_df.groupby(["gamepk", "pitcher_id", "half"])["residual"]
    .sum()
    .unstack("half")
    .dropna()
)

r, p_value = stats.pearsonr(split_sums["odd"], split_sums["even"])
print(f"\n{'=' * 72}\nWITHIN-GAME RESIDUAL CORRELATION (odd PAs vs. even PAs, same start)\n{'=' * 72}")
print(f"n starts: {len(split_sums):,}")
print(f"Pearson r: {r:.4f}   p-value: {p_value:.4g}")
print(f"Interpretation: r > 0 and significant => real within-game 'quality today' signal")
print(f"                the current independent-per-slot pmf assumption is missing.")
print(f"                r ~ 0 => residuals really do behave independently within a start,")
print(f"                        and the thin-tail problem needs a different explanation.")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(split_sums["odd"], split_sums["even"], s=14, alpha=0.4, color="#1f77b4")
lims = [min(split_sums.min()) - 0.5, max(split_sums.max()) + 0.5]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Sum(actual - predicted) over odd-numbered PAs this start")
ax.set_ylabel("Sum(actual - predicted) over even-numbered PAs this start")
ax.set_title(f"Within-game residual correlation, r={r:.3f} (p={p_value:.3g}), n={len(split_sums):,}")
plt.tight_layout()
plt.savefig(OUT_DIR / "within_game_correlation.png", dpi=130)
plt.close()

split_sums.to_parquet(OUT_DIR / "split_sums.parquet")
print(f"\nSaved {OUT_DIR / 'within_game_correlation.png'}")
