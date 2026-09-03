"""
k_predictor v12 -- start-grain Negative Binomial (NB2) total-K model.

See src/models/implementation_plan.md for the full plan this implements.

SUMMARY: v6's Poisson-binomial combination models total strikeouts per SP
start as the sum of ~22 independent per-slot Bernoulli trials -- a real
1,160-start 2026 DK-odds backtest (ROADMAP.md item 6(c)) found this
aggregation significantly overconfident (reliability 0.0145 vs. market's
0.0011, 13x, bootstrap CI excludes zero), consistent with under-modeling the
real within-game correlation across slots (same game/park/weather/bullpen
trajectory). v12 tests a different lever: model total K per start directly
as its own count-distributed target (Negative Binomial, NB2 parameterization,
`statsmodels.discrete.discrete_model.NegativeBinomial`), with no per-slot
independence assumption anywhere in the chain. No PA-grain slot expansion,
no Poisson-binomial combine -- an MLE fit yields a per-start predicted mean
and a single fitted dispersion alpha directly.

Run from src/models/k_predictor/ with:
    PYTHONPATH=../../../src python experiments/v12_start_grain_negbin/train.py
(or, equivalently, from repo root: PYTHONPATH=src python
src/models/k_predictor/experiments/v12_start_grain_negbin/train.py)
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
import json
import unicodedata
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.utils.eval import evaluate_hit_predictor, get_calibration_df, murphy_decomposition
from models.hit_predictor.utils.count_distribution import negative_binomial_pmf, prob_exceeds_line
from models.hit_predictor.utils.odds_math import devig_two_way

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backtest"))
from fetch_2025_odds import flatten_props

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).parent
WHIP_SHRINKAGE_K = 20.0
PA_SHRINKAGE_K = 5.0
SHORT_PITCHER_WINDOW = 3
SHORT_TEAM_WINDOW = 5
MAX_K = 30
N_BOOT = 10_000
RNG = np.random.default_rng(42)

# Same 72 dates score_2026_test_dates.py / edge_report_2026.py use -- the
# only dates real DraftKings pitcher_strikeouts odds exist for.
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


# ── 1. Config + load ───────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET, REGION = cfg["bucket"], cfg["region"]
TRAIN_SEASONS, FEATURE_SEASONS = cfg["train_seasons"], cfg["feature_seasons"]
TARGET, DATE_COL = cfg["target_column"], cfg["date_column"]
TEST_SEASON, VAL_SEASON = cfg["test_season"], cfg["val_season"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
FIT_SEASONS.remove(2017)
EARLY_STOP_SEASON = max(FIT_SEASONS)
CORE_FIT_SEASONS = [s for s in FIT_SEASONS if s != EARLY_STOP_SEASON]
# v12 has no early-stopping mechanism (MLE fit, not boosting) -- fits on the
# combined footprint instead of holding EARLY_STOP_SEASON out. See plan's
# Scope Decisions for the full rationale.
NB_FIT_SEASONS = CORE_FIT_SEASONS + [EARLY_STOP_SEASON]
assert SCORE_SEASON not in NB_FIT_SEASONS, "SCORE_SEASON must never be fit on"
print(f"NB fit seasons: {NB_FIT_SEASONS}  |  val (coverage check): {VAL_SEASON}  |  "
      f"score (real-odds backtest): {SCORE_SEASON}")

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


# ── 2. Build the shared feature-family tables v6's own scripts build --
# no PA-outcome frame, no XGBoost fit: v12 is start-grain only, and none of
# these builder calls actually depend on the PA-outcome frame (they read
# pbp/pitcher_boxscore/batter_boxscore directly) -- reused byte-for-byte
# from score_2026_test_dates.py's Section 2 minus the PA-grain-only parts. ──
print("\nBuilding shared feature tables...")
schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)
pbp = pbp.merge(schedule[["gamepk", "game_datetime"]], on="gamepk", how="left")

pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_pa_strikeout_rate"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})
pitcher_box_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)[
    ["pitcher_key_id", "pitcher_role", "game_season", "pitcher_last_season_whip"]
].rename(columns={"pitcher_key_id": "expected_pitcher_key_id", "pitcher_role": "expected_pitcher_role"})

pitcher_box_rolling = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window="season")
box_rolling_cols = [
    "pitcher_roll_season_ip", "pitcher_roll_season_whip", "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate", "pitcher_roll_season_hr_rate", "pitcher_roll_season_games_n",
]

pitcher_pbp_rolling = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window="season")
pbp_rolling_cols = [
    "pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate",
    "pitcher_roll_season_command_swinging_strike_rate", "pitcher_roll_season_pa_pitch_count_mean",
]

team_batter_rolling = rolling_stats.build_team_batter_strikeout_rolling_feats(pbp, batter_boxscore, window="season")
pitching_team_rolling = team_batter_rolling.rename(columns={
    "batter_team_id": "pitcher_team_id",
    "team_roll_season_pa_strikeout_rate": "pitching_team_roll_season_pa_strikeout_rate",
})[["pitcher_team_id", "gamepk", "pitching_team_roll_season_pa_strikeout_rate"]]

# EPIC 2 -- the actual opposing (batting) side's K rate, not the pitcher's
# own team's. Same already-tested team_batter_rolling table, just merged
# onto opp_team_id/gamepk instead of pitcher_team_id/gamepk.
opp_team_rolling = team_batter_rolling.rename(columns={
    "batter_team_id": "opp_team_id",
    "team_roll_season_pa_strikeout_rate": "opp_team_roll_season_pa_strikeout_rate",
})[["opp_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate"]]

pitcher_box_rolling3 = rolling_stats.build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window=SHORT_PITCHER_WINDOW)
box_rolling3_cols = [
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_whip",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_k_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_bb_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_hr_rate", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n",
]
pitcher_pbp_rolling3 = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window=SHORT_PITCHER_WINDOW)
pbp_rolling3_cols = [
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_total", f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_strikeout_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_command_swinging_strike_rate",
    f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_pa_pitch_count_mean",
]

opp_team_volatility_season = rolling_stats.build_team_strikeout_volatility(pbp, batter_boxscore, window="season")
opp_team_volatility_short = rolling_stats.build_team_strikeout_volatility(pbp, batter_boxscore, window=SHORT_TEAM_WINDOW)
short_team_prefix = f"team_roll_last{SHORT_TEAM_WINDOW}g_pa_strikeout_rate"

shrunk_whip = build_pitcher_shrunk_whip(
    pitcher_box_rolling, season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp), window="season", k=WHIP_SHRINKAGE_K,
)

schedule_teams = schedule[["gamepk", "home_id", "away_id"]].drop_duplicates("gamepk")
throw_hand = pbp[["pitcher_id", "gamepk", "pitcher_throw_hand"]].drop_duplicates().rename(columns={"pitcher_id": "personId"})
weather = game_info[["gamepk", "weather_condition", "weather_temp"]].drop_duplicates("gamepk")


# ── 3. Start-grain frame assembly (EPIC 2 + 3, STORY 3.1) -- one row per
# pitcher-start, built once across all seasons this experiment needs
# (NB_FIT_SEASONS + VAL_SEASON + SCORE_SEASON all come from the same
# expected_pa cascade), then sliced per use. Reused verbatim from
# score_2026_test_dates.py's Section 4, generalized into a function so it
# isn't duplicated 3x for fit/val/score. ────────────────────────────────────
print("\nBuilding expected_batters_faced cascade + start-grain frame...")
pitcher_start_pa_last_season = season_stats.build_pitcher_start_pa_stats(pbp)
league_avg_start_pa = season_stats.build_league_avg_start_pa(pitcher_start_pa_last_season)
team_avg_start_pa = season_stats.build_team_avg_start_pa(pitcher_start_pa_last_season)
pitcher_start_pa_this_season = game_context.build_pitcher_start_pa_this_season(pbp)

expected_pa = game_context.build_expected_batters_faced(
    pitcher_start_pa_last_season, pitcher_start_pa_this_season, team_avg_start_pa, league_avg_start_pa, k=PA_SHRINKAGE_K,
)

starts = expected_pa.copy()
starts = starts.merge(
    pitcher_role_season_stats.rename(columns={"expected_pitcher_key_id": "personId"})[
        ["personId", "game_season", "pitcher_last_season_pa_strikeout_rate"]
    ],
    on=["personId", "game_season"], how="left",
)
starts = starts.merge(
    pitcher_box_season_stats.rename(columns={"expected_pitcher_key_id": "personId"})[["personId", "game_season", "pitcher_last_season_whip"]],
    on=["personId", "game_season"], how="left",
)
starts = starts.merge(
    pitcher_box_rolling.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + box_rolling_cols],
    on=["gamepk", "personId"], how="left",
)
starts = starts.merge(
    pitcher_pbp_rolling.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + pbp_rolling_cols],
    on=["gamepk", "personId"], how="left",
)
starts["pitcher_roll_season_avg_ip_per_game"] = (
    starts["pitcher_roll_season_ip"] / starts["pitcher_roll_season_games_n"].replace(0, np.nan)
)
starts = starts.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")
starts = starts.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId", "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]],
    on=["gamepk", "personId"], how="left",
)

starts = starts.merge(schedule_teams, on="gamepk", how="left")
starts["opp_team_id"] = np.where(
    starts["pitcher_team_id"] == starts["home_id"], starts["away_id"], starts["home_id"],
)

# EPIC 2's new merge -- the opponent's own batting K rate.
starts = starts.merge(opp_team_rolling, on=["opp_team_id", "gamepk"], how="left")

starts = starts.merge(
    pitcher_box_rolling3.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + box_rolling3_cols],
    on=["gamepk", "personId"], how="left",
)
starts = starts.merge(
    pitcher_pbp_rolling3.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + pbp_rolling3_cols],
    on=["gamepk", "personId"], how="left",
)
starts[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_avg_ip_per_game"] = (
    starts[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_ip"]
    / starts[f"pitcher_roll_last{SHORT_PITCHER_WINDOW}g_games_n"].replace(0, np.nan)
)
starts = starts.merge(
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
starts = starts.merge(
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
starts = starts.merge(throw_hand, on=["personId", "gamepk"], how="left")
starts = starts.merge(weather, on="gamepk", how="left")

# Sanity check on EPIC 2's new column (STORY 2.1's acceptance criterion --
# non-null rate matching every other rolling-season feature in this
# pipeline, same convention v10/v11's results docs use).
nn_rate = starts["opp_team_roll_season_pa_strikeout_rate"].notna().mean()
print(f"\nopp_team_roll_season_pa_strikeout_rate: non-null={nn_rate:.1%}, "
      f"mean={starts['opp_team_roll_season_pa_strikeout_rate'].mean():.4f}")

NUMERIC_FEATURE_COLS = [
    "pitcher_last_season_pa_strikeout_rate", "pitcher_last_season_whip",
    "pitcher_roll_season_ip", "pitcher_roll_season_whip", "pitcher_roll_season_k_rate",
    "pitcher_roll_season_bb_rate", "pitcher_roll_season_hr_rate", "pitcher_roll_season_games_n",
    "pitcher_roll_season_avg_ip_per_game", "pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate",
    "pitching_team_roll_season_pa_strikeout_rate", "opp_team_roll_season_pa_strikeout_rate",
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
    "weather_temp", "expected_batters_faced", "expected_batters_faced_weight",
]
CATEGORICAL_FEATURE_COLS = ["pitcher_throw_hand", "weather_condition"]
NUMERIC_FEATURE_COLS = [c for c in NUMERIC_FEATURE_COLS if c in starts.columns]
CATEGORICAL_FEATURE_COLS = [c for c in CATEGORICAL_FEATURE_COLS if c in starts.columns]
print(f"Feature count: {len(NUMERIC_FEATURE_COLS)} numeric + {len(CATEGORICAL_FEATURE_COLS)} categorical")

# realized total K -- reuse the SAME pitcher_role=='sp' filter every prior
# scoring script uses.
role_lookup = season_stats._pitcher_role_lookup(pbp)[["gamepk", "pitcher_id", "pitcher_role"]].rename(columns={"pitcher_id": "personId"})
pitcher_box_tagged = pitcher_boxscore.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
).merge(
    role_lookup.assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str)),
    on=["gamepk", "personId"], how="left",
)
realized_k = pitcher_box_tagged[pitcher_box_tagged["pitcher_role"] == "sp"][
    ["personId", "gamepk", "game_season", "k", "player_name"]
].rename(columns={"k": "realized_k"})

starts["personId"] = starts["personId"].astype(str)
starts["gamepk"] = starts["gamepk"].astype(str)
starts = starts.merge(realized_k, on=["personId", "gamepk", "game_season"], how="inner")
print(f"Total SP starts with a realized K count, across all read seasons: {len(starts):,}")


# ── 4. Fit the NB2 GLM (EPIC 3, STORY 3.2) ─────────────────────────────────
fit_df = starts[starts["game_season"].isin(NB_FIT_SEASONS)].copy()
print(f"\nFitting NegativeBinomial GLM on {len(fit_df):,} starts ({NB_FIT_SEASONS})...")

num_imp = SimpleImputer(strategy="median")
num_imp.fit(fit_df[NUMERIC_FEATURE_COLS])


def build_design_matrix(df, cat_dummy_cols=None):
    """Median-impute numeric features (fit on NB_FIT_SEASONS only), one-hot
    encode categoricals (appropriate for a linear-in-log-mean GLM, unlike
    tree-model ordinal encoding), add a constant. cat_dummy_cols, when
    passed, pins the fit's own dummy columns so eval/score frames get
    exactly the same design matrix shape even if a category is missing or
    novel in that slice."""
    x_num = pd.DataFrame(num_imp.transform(df[NUMERIC_FEATURE_COLS]), columns=NUMERIC_FEATURE_COLS, index=df.index)
    # drop_first=True -- dummy_na=True alone means every row's dummies for a
    # given categorical sum to exactly 1, which is exactly collinear with
    # the intercept and leaves the Hessian singular (this is what produced
    # every coefficient's std err coming back NaN on the first fit attempt).
    x_cat = pd.get_dummies(df[CATEGORICAL_FEATURE_COLS].astype(object), dummy_na=True, drop_first=True) if CATEGORICAL_FEATURE_COLS else pd.DataFrame(index=df.index)
    if cat_dummy_cols is not None:
        x_cat = x_cat.reindex(columns=cat_dummy_cols, fill_value=False)
    x = pd.concat([x_num, x_cat.astype(float)], axis=1)
    return sm.add_constant(x, has_constant="add"), list(x_cat.columns)


X_fit, cat_dummy_cols = build_design_matrix(fit_df)
y_fit = fit_df["realized_k"].astype(float)

nb_model = sm.NegativeBinomial(y_fit, X_fit)
nb_result = nb_model.fit(disp=False, maxiter=200)
converged = bool(nb_result.mle_retvals.get("converged", False))
fitted_alpha = float(nb_result.params["alpha"])
print(f"Converged: {converged}  |  fitted alpha: {fitted_alpha:.4f}")
print(nb_result.summary())


def predict_mean(df):
    x, _ = build_design_matrix(df, cat_dummy_cols=cat_dummy_cols)
    x = x.reindex(columns=X_fit.columns, fill_value=0.0)
    return nb_result.predict(x)


# ── 5. Per-start pmf + 2024 val-season coverage/threshold check (EPIC 3,
# STORY 3.3, parts 1-2) -- same coverage-check + threshold-check machinery
# run_xgboost_uncertainty.py already runs on v6, applied to v12's pmfs. ────
print(f"\n{'=' * 72}\n2024 VAL-SEASON COVERAGE CHECK\n{'=' * 72}")
val_df = starts[starts["game_season"] == VAL_SEASON].copy()
val_df["predicted_mean_k"] = predict_mean(val_df)
val_df["pmf"] = val_df["predicted_mean_k"].apply(lambda mu: negative_binomial_pmf(mu, fitted_alpha, MAX_K))
val_df["residual"] = val_df["predicted_mean_k"] - val_df["realized_k"]
val_df["abs_error"] = val_df["residual"].abs()
print(f"n starts: {len(val_df):,}")
print(f"Predicted mean K: {val_df['predicted_mean_k'].mean():.3f}  |  Realized mean K: {val_df['realized_k'].mean():.3f}  |  MAE: {val_df['abs_error'].mean():.3f}")

LINE = round(val_df["realized_k"].median()) + 0.5
val_df["p_over_line"] = val_df["pmf"].apply(lambda pmf: prob_exceeds_line(pmf, LINE))
val_df["realized_over_line"] = (val_df["realized_k"] > LINE).astype(int)

print(f"\nTHRESHOLD CHECK -- P(total K > {LINE}) vs. realized")
threshold_metrics = evaluate_hit_predictor(
    y_true=val_df["realized_over_line"], y_prob=val_df["p_over_line"],
    n_bins=8, min_n=30, base_rate=val_df["realized_over_line"].mean(),
)


def interval_bounds(pmf, level):
    cdf = np.cumsum(pmf)
    alpha = (1 - level) / 2
    lower = int(np.searchsorted(cdf, alpha, side="left"))
    upper = int(np.searchsorted(cdf, 1 - alpha, side="left"))
    return lower, upper


LEVELS = [0.50, 0.80, 0.95]
print(f"\nCOVERAGE CHECK")
print(f"{'level (nominal)':<18} {'n':>6} {'empirical coverage':>20} {'gap (empirical - nominal)':>28}")
coverage_by_level = {}
for level in LEVELS:
    bounds = val_df["pmf"].apply(lambda pmf: interval_bounds(pmf, level))
    lower = bounds.apply(lambda t: t[0])
    upper = bounds.apply(lambda t: t[1])
    covered = (val_df["realized_k"] >= lower) & (val_df["realized_k"] <= upper)
    emp = covered.mean()
    coverage_by_level[str(int(level * 100))] = round(float(emp), 4)
    print(f"{level:<18.0%} {len(val_df):>6} {emp:>20.1%} {emp - level:>+28.1%}")


# ── 6. 2026 real-odds backtest (EPIC 3, STORY 3.3, part 3) -- SAME dates,
# SAME devig/edge logic as edge_report_2026.py, pointed at v12's pmfs, then
# significance_check_2026.py's reliability/resolution/disagreement-win-rate
# checks on v12's results. ──────────────────────────────────────────────────
print(f"\n{'=' * 72}\n2026 REAL-ODDS BACKTEST\n{'=' * 72}")
score_df = starts[starts["game_season"] == SCORE_SEASON].copy()
score_df = score_df[score_df["game_date"].isin(pd.to_datetime(BACKTEST_DATES))].copy()
score_df["predicted_mean_k"] = predict_mean(score_df)
score_df["pmf"] = score_df["predicted_mean_k"].apply(lambda mu: negative_binomial_pmf(mu, fitted_alpha, MAX_K))
print(f"2026 SP starts on the {len(BACKTEST_DATES)} backtest dates: {len(score_df):,}")

pred_df_2026 = score_df[["personId", "gamepk", "game_date", "player_name", "realized_k", "pmf", "predicted_mean_k"]].copy()
pred_df_2026["gamepk"] = pred_df_2026["gamepk"].astype(str)


def normalize_name(name):
    if pd.isna(name):
        return None
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").strip()


def load_odds_2026():
    frames = []
    for date in BACKTEST_DATES:
        path = f"s3://{BUCKET}/raw_data/odds/player_props/2026/{date}.parquet"
        try:
            raw = wr.s3.read_parquet(path=path, boto3_session=boto_session)
        except Exception as e:
            print(f"  skipping {date}: {e}")
            continue
        if raw.empty:
            continue
        game_props = raw.to_dict("records")
        df = flatten_props(date, game_props)
        frames.append(df)
    odds = pd.concat(frames, ignore_index=True)
    odds["date"] = pd.to_datetime(odds["date"])
    odds["name_norm"] = odds["pitcher_name"].apply(normalize_name)
    over = odds[odds["side"] == "Over"][["date", "event_id", "pitcher_name", "name_norm", "line", "price"]].rename(columns={"price": "over_price"})
    under = odds[odds["side"] == "Under"][["date", "event_id", "name_norm", "line", "price"]].rename(columns={"price": "under_price"})
    return over.merge(under, on=["date", "event_id", "name_norm", "line"], how="inner")


pred_df_2026["name_norm"] = pred_df_2026["player_name"].apply(normalize_name)
pred_df_2026["date"] = pd.to_datetime(pred_df_2026["game_date"])

print("Loading real DK pitcher_strikeouts odds from S3...")
odds = load_odds_2026()
matched = pred_df_2026.merge(odds, on=["date", "name_norm"], how="inner")
print(f"Matched {len(matched)} pitcher-starts to real DK lines.")

matched["market_p_over"], matched["market_p_under"] = zip(*matched.apply(
    lambda r: devig_two_way(r["over_price"], r["under_price"]), axis=1
))
matched["model_p_over"] = matched.apply(lambda r: prob_exceeds_line(r["pmf"], r["line"]), axis=1)
matched["edge"] = matched["model_p_over"] - matched["market_p_over"]
matched["realized_over"] = (matched["realized_k"] > matched["line"]).astype(int)
matched["model_favored_over"] = matched["model_p_over"] > 0.5
matched["market_favored_over"] = matched["market_p_over"] > 0.5
matched["disagree_direction"] = matched["model_favored_over"] != matched["market_favored_over"]

print(f"Mean edge (model - devigged market): {matched['edge'].mean():+.4f}")

disagreement = matched[matched["disagree_direction"]]
n_win, n_total, win_rate = 0, 0, float("nan")
if len(disagreement):
    follow_correct = (disagreement["model_favored_over"] == disagreement["realized_over"].astype(bool))
    n_win, n_total = int(follow_correct.sum()), len(disagreement)
    win_rate = n_win / n_total
    print(f"Disagreement win rate: {n_win}/{n_total} = {win_rate:.1%}")
    for label, p0 in [("coin flip", 0.50), ("~-110 break-even", 0.524)]:
        r = binomtest(n_win, n_total, p=p0, alternative="two-sided")
        ci = r.proportion_ci(confidence_level=0.95, method="wilson")
        sig = "SIGNIFICANT" if not (ci.low <= p0 <= ci.high) else "not significant"
        print(f"  vs. {label} ({p0:.1%}): p={r.pvalue:.4f}  95% CI=[{ci.low:.1%}, {ci.high:.1%}]  -> {sig}")


def resolution(y_true, y_prob, n_bins=8):
    cal_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=0)
    return murphy_decomposition(y_true, cal_df)["resolution"]


def reliability(y_true, y_prob, n_bins=8):
    cal_df = get_calibration_df(y_true, y_prob, n_bins=n_bins, min_n=0)
    return murphy_decomposition(y_true, cal_df)["reliability"]


y = matched["realized_over"].to_numpy()
mp = matched["model_p_over"].to_numpy()
kp = matched["market_p_over"].to_numpy()
m_rel, k_rel = reliability(y, mp), reliability(y, kp)
m_res, k_res = resolution(y, mp), resolution(y, kp)
print(f"\nReliability (lower=better): v12 model={m_rel:.5f}  market={k_rel:.5f}  gap={m_rel - k_rel:+.5f}")
print(f"Resolution (higher=better): v12 model={m_res:.5f}  market={k_res:.5f}  gap(market-model)={k_res - m_res:+.5f}")


def bootstrap_gap(y, a, b, stat_fn, n_boot=N_BOOT):
    n = len(y)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            gaps[i] = np.nan
            continue
        gaps[i] = stat_fn(yb, a[idx]) - stat_fn(yb, b[idx])
    valid = gaps[~np.isnan(gaps)]
    return np.percentile(valid, 2.5), np.percentile(valid, 97.5)


rel_lo, rel_hi = bootstrap_gap(y, mp, kp, reliability)
rel_sig = "SIGNIFICANT -- model IS more miscalibrated than the market" if rel_lo > 0 or rel_hi < 0 else "not significant"
print(f"  95% bootstrap CI on reliability gap=[{rel_lo:+.5f}, {rel_hi:+.5f}]  -> {rel_sig}")
res_lo, res_hi = bootstrap_gap(y, kp, mp, resolution)
res_sig = "SIGNIFICANT" if res_lo > 0 or res_hi < 0 else "not significant"
print(f"  95% bootstrap CI on resolution gap=[{res_lo:+.5f}, {res_hi:+.5f}]  -> {res_sig}")


# ── 7. Persist everything the results doc needs ────────────────────────────
summary = {
    "converged": converged,
    "fitted_alpha": round(fitted_alpha, 4),
    "nb_fit_seasons": NB_FIT_SEASONS,
    "nb_fit_n_starts": int(len(fit_df)),
    "opp_team_roll_season_pa_strikeout_rate_nonnull_rate": round(float(nn_rate), 4),
    "val_season": VAL_SEASON,
    "val_n_starts": int(len(val_df)),
    "val_predicted_mean_k": round(float(val_df["predicted_mean_k"].mean()), 3),
    "val_realized_mean_k": round(float(val_df["realized_k"].mean()), 3),
    "val_mae": round(float(val_df["abs_error"].mean()), 3),
    "coverage_by_level": coverage_by_level,
    "threshold_check": {
        "line": LINE,
        "reliability": round(float(threshold_metrics["reliability"]), 4),
        "resolution": round(float(threshold_metrics["resolution"]), 4),
        "roc_auc": round(float(threshold_metrics["roc_auc"]), 4),
        "pr_auc": round(float(threshold_metrics["pr_auc"]), 4),
    },
    "backtest_2026": {
        "n_matched": int(len(matched)),
        "mean_edge": round(float(matched["edge"].mean()), 4),
        "n_disagreement": n_total,
        "disagreement_win_rate": round(win_rate, 4) if n_total else None,
        "reliability": round(float(m_rel), 5),
        "market_reliability": round(float(k_rel), 5),
        "resolution": round(float(m_res), 5),
        "market_resolution": round(float(k_res), 5),
        "reliability_gap_ci": [round(float(rel_lo), 5), round(float(rel_hi), 5)],
        "resolution_gap_ci": [round(float(res_lo), 5), round(float(res_hi), 5)],
    },
}
with open(OUT_DIR / "v12_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
val_df.drop(columns=["pmf"]).to_parquet(OUT_DIR / "pred_df_val2024.parquet", index=False)
matched.drop(columns=["pmf"]).to_parquet(OUT_DIR / "pred_df_test2026.parquet", index=False)
print(f"\nSummary written to {OUT_DIR / 'v12_summary.json'}")
