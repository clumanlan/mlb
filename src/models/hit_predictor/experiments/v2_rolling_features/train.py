
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from models.hit_predictor.utils.eval import evaluate_hit_predictor, plot_calibration_curve
from models.hit_predictor.utils.model_prep import (
    add_missing_indicators,
    get_columns_with_nulls,
    impute_and_encode,
    plot_feature_importance,
)

import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
from models.hit_predictor.utils.mlflow_logging import log_evaluation_to_mlflow, get_git_sha

import models.hit_predictor.processing.pipeline as pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats


STAGE = Path(__file__).parent

# Short rolling window size, in games. Change this one constant to try a different
# recent-form window — it flows through every build_*_rolling_stats(window=...) call below.
SHORT_WINDOW_GAMES = 10

pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── 1. Config ────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET           = cfg["bucket"]
REGION           = cfg["region"]
TRAIN_SEASONS    = cfg["train_seasons"]
FEATURE_SEASONS  = cfg["feature_seasons"]
TARGET           = cfg["target_column"]
DATE_COL         = cfg["date_column"]
TEST_SEASON      = cfg["test_season"]
VAL_SEASON       = cfg["val_season"]
MODEL_NAME       = cfg["model_name"]


# Seasons used for model fitting (everything that isn't val or test)
FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]

# have to drop 2017 bc of lagged stats - can fill in later
FIT_SEASONS.remove(2017)

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))

# ---------------------------------------------------------------------------- #
#                                 READ IN DATA                                 #
# ---------------------------------------------------------------------------- #
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
    "s3://{bucket}/processed_data/prepared/playbyplay/{season}/",
    TRAIN_SEASONS,
    chunked=True,
)

print("\nLoading schedule...")
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/",
    TRAIN_SEASONS,
)

print("\nLoading game info...")
game_info = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/game_info/{season}/",
    TRAIN_SEASONS,
)

print("\nLoading batter boxscore...")
batter_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/batter_boxscore/{season}/",
    all_boxscore_seasons,
)

print("\nLoading pitcher boxscore...")
pitcher_boxscore = read_parquet_seasons(
    "s3://{bucket}/processed_data/prepared/pitcher_boxscore/{season}/",
    all_boxscore_seasons,
)

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/",
    boto3_session=boto_session,
)


game_info = pipeline.process_game_info(game_info)
pitcher_boxscore = pipeline.process_pitcher_boxscore(pitcher_boxscore)
schedule = pipeline.process_schedule(schedule)

pbp = pipeline.build_pbp_features(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

# The starting pitcher is known pre-game, but which specific reliever will face a given
# batter is not — pitcher_key_id resolves to the individual pitcher for sp-role PAs and
# to that pitcher's TEAM for bullpen-role PAs, so bullpen features can be pooled at team
# grain (matching src/features/transforms/bullpen_pitcher_base.py's existing precedent)
# instead of relying on an individual reliever's identity that won't be known at serving
# time. pandas .where() keeps the value where the condition is True, replaces it where
# False — i.e. "sp -> pitcher_id, else -> pitcher_team_id".
pa_outcome['pitcher_key_id'] = pa_outcome['pitcher_id'].where(
    pa_outcome['pitcher_role'] == 'sp', pa_outcome['pitcher_team_id']
)

batter_season_stats = season_stats.build_batter_stats(batter_boxscore).rename(columns={'personId': 'batter_id'})
# Role-tagged AND role-pooled: a swingman (starts some games, relieves others in the same
# season) gets separate 'sp'/'bullpen' rows rather than one blended aggregate, and the
# bullpen half is pooled by team (not kept per individual pitcher_id) since a specific
# reliever's identity isn't knowable pre-game. Both output a pitcher_key_id column so they
# join onto pa_outcome's own pitcher_key_id uniformly regardless of role.
pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp)
pitcher_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp)

# rolling season (expanding, resets each season) + rolling short (trailing
# SHORT_WINDOW_GAMES, carries across season boundaries) — same stat categories as
# season_stats above, updated game-by-game instead of once a year. personId is
# renamed to the entity col up front so these merge cleanly on ['gamepk', 'batter_id'
# / 'pitcher_id'] below, matching how sp_season_stats already merges on pitcher_id.
batter_rolling_season_stats = (
    rolling_stats.build_batter_rolling_stats(batter_boxscore, window='season')
    .rename(columns={'personId': 'batter_id'})
)
batter_rolling_short_stats = (
    rolling_stats.build_batter_rolling_stats(batter_boxscore, window=SHORT_WINDOW_GAMES)
    .rename(columns={'personId': 'batter_id'})
)

# Role-tagged AND role-pooled, same rationale as pitcher_season_stats above.
pitcher_rolling_season_stats = rolling_stats.build_pitcher_rolling_stats_all_roles(
    pitcher_boxscore, pbp, window='season'
)
pitcher_rolling_short_stats = rolling_stats.build_pitcher_rolling_stats_all_roles(
    pitcher_boxscore, pbp, window=SHORT_WINDOW_GAMES
)

# Same role-tagged, role-pooled pattern as pitcher_role_season_stats above, rolling instead of season.
pitcher_role_rolling_season_stats = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window='season')
pitcher_role_rolling_short_stats = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(pbp, window=SHORT_WINDOW_GAMES)


# ---------------------------------------------------------------------------- #
#                              ASSEMBLE MODEL FRAME                            #
# ---------------------------------------------------------------------------- #
# season_stats keyed by (game_season, entity_id) — one row per player-season, same
# number for every game that season. rolling_stats keyed by (gamepk, entity_id) —
# one row per player-game, updates every game. game_date/game_season are dropped
# from the rolling frames before merging since pa_outcome already carries those from
# game_info/schedule; keeping them would collide as *_x/*_y on every merge below.
model_df = pa_outcome.merge(
    batter_season_stats, on=['game_season', 'batter_id'], how='left'
)
# pitcher_key_id + pitcher_role join as the key: pa_outcome carries each PA's own
# game-context role and resolved key (sp -> pitcher_id, bullpen -> pitcher_team_id), so
# this picks up exactly the sp/team-pooled-bullpen row matching that role.
model_df = model_df.merge(
    pitcher_season_stats, on=['game_season', 'pitcher_key_id', 'pitcher_role'], how='left'
)
model_df = model_df.merge(
    pitcher_role_season_stats, on=['game_season', 'pitcher_key_id', 'pitcher_role'], how='left'
)

for rolling_df in (batter_rolling_season_stats, batter_rolling_short_stats):
    rolling_df = rolling_df.drop(columns=['game_date', 'game_season'])
    model_df = model_df.merge(rolling_df, on=['gamepk', 'batter_id'], how='left')

# Role-tagged (and for bullpen, role-pooled) rolling tables need pitcher_key_id +
# pitcher_role in the merge key, so they can't share the batter loop above.
for rolling_df in (
    pitcher_rolling_season_stats, pitcher_rolling_short_stats,
    pitcher_role_rolling_season_stats, pitcher_role_rolling_short_stats,
):
    rolling_df = rolling_df.drop(columns=['game_date', 'game_season'])
    model_df = model_df.merge(rolling_df, on=['gamepk', 'pitcher_key_id', 'pitcher_role'], how='left')

# A specific reliever's throwing hand isn't knowable pre-game either (same reasoning as
# the stat pooling above) — null it for bullpen rows so the existing missing-indicator +
# "missing"-sentinel-category machinery treats it as genuinely unknown rather than
# training on the actual (post-hoc-only-known) hand of whoever happened to relieve.
model_df.loc[model_df['pitcher_role'] == 'bullpen', 'pitcher_throw_hand'] = None


# ---------------------------------------------------------------------------- #
#                                  TRAIN MODEL                                 #
# ---------------------------------------------------------------------------- #

# ------------------------------ DEFINE FEATURES ----------------------------- #
# game_season is deliberately excluded from NUM_FEATS: val/test seasons are always
# chronologically after every fit season, so a raw year feature would ask a tree to
# extrapolate past the range it was trained on rather than encode anything that
# generalizes. Numeric/categorical features are split by dtype rather than by
# hardcoded prefix lists — model_df's stat columns come from several modules
# (season_stats, rolling_stats) with different naming conventions, so dtype is the
# one thing that reliably holds across all of them.
NON_FEATURE_COLS = {
    'gamepk', 'play_id', 'batter_id', 'pitcher_id', 'pitcher_team_id', 'pitcher_key_id',
    'batter_team_name', 'pitcher_name', 'batter_name',
    'game_date', 'game_season', TARGET,
}
CAT_FEATS = ['pitcher_throw_hand', 'batter_bat_side', 'weather_condition', 'pitcher_role']
NUM_FEATS = [
    c for c in model_df.columns
    if c not in NON_FEATURE_COLS and c not in CAT_FEATS
    and pd.api.types.is_numeric_dtype(model_df[c])
]

# -------------------------------- SPLIT DATA -------------------------------- #
train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()

X_train_raw = train_df[NUM_FEATS + CAT_FEATS].copy()
y_train     = train_df[TARGET].copy()
X_val_raw   = val_df[NUM_FEATS + CAT_FEATS].copy()
y_val       = val_df[TARGET].copy()

# ---------------------------- MISSING INDICATORS ----------------------------- #
# A rookie batter/pitcher with no last-season stats is systematically different
# from a veteran — imputing the median alone throws that fact away. Indicator
# columns are derived from train only (get_columns_with_nulls) and applied
# identically to val, so the two splits never diverge on which indicators exist.
indicator_cols = get_columns_with_nulls(X_train_raw, NUM_FEATS)
indicator_col_names = [f"{c}_isnull" for c in indicator_cols]
X_train_ind = add_missing_indicators(X_train_raw, indicator_cols)
X_val_ind   = add_missing_indicators(X_val_raw, indicator_cols)

# ------------------------------ IMPUTE + ENCODE ------------------------------ #
# No StandardScaler — Random Forest doesn't need feature scaling, and skipping it
# avoids the class of bug where one column (e.g. a raw year) rides along unscaled
# next to standardized ones.
X_train, X_val = impute_and_encode(X_train_ind, X_val_ind, num_cols=NUM_FEATS, cat_cols=CAT_FEATS)
X_train = pd.concat([X_train, X_train_ind[indicator_col_names]], axis=1)
X_val   = pd.concat([X_val, X_val_ind[indicator_col_names]], axis=1)
FEATURE_NAMES = list(X_train.columns)


# ------------------------------ TRAIN MODELS -------------------------------- #
models = {
    "Naive baseline": DummyClassifier(strategy="prior", random_state=42),
    # min_samples_leaf/max_features are a standard RF-first starting point (prevents
    # single-observation leaves, decorrelates trees); oob_score gives a free internal
    # generalization check without spending any extra held-out data.
    "Random Forest": RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, max_features="sqrt",
        n_jobs=-1, oob_score=True, random_state=42,
    ),
}

N_BINS = 10
MIN_N = 500

PLOT_DIR = BASE_DIR / "plots" / "v2_rolling_features"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_RESULTS_MD = BASE_DIR / "baseline_results.md"

MODEL_TYPE_TAGS = {
    "Naive baseline": "naive_baseline",
    "Random Forest": "random_forest",
}

fitted_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    fitted_models[name] = model

print(f"Random Forest OOB score: {fitted_models['Random Forest'].oob_score_:.4f}"
      "  (sanity check against val log loss below)")


for name, model in fitted_models.items():
    proba = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_hit_predictor(
        y_true=y_val, y_prob=proba, n_bins=N_BINS, min_n=MIN_N,
    )

    calibration_plot_path = PLOT_DIR / f"{MODEL_TYPE_TAGS[name]}_calibration_curve.png"
    plot_calibration_curve(
        y_val, {name: {"proba": proba}}, n_bins=N_BINS, min_n=MIN_N,
        save_path=calibration_plot_path,
    )

    artifact_paths = [str(calibration_plot_path)]
    if BASELINE_RESULTS_MD.exists():
        artifact_paths.append(str(BASELINE_RESULTS_MD))

    if name == "Random Forest":
        importance_plot_path = PLOT_DIR / "random_forest_feature_importance.png"
        plot_feature_importance(
            pd.Series(model.feature_importances_, index=FEATURE_NAMES),
            save_path=importance_plot_path,
        )
        artifact_paths.append(str(importance_plot_path))

    log_evaluation_to_mlflow(
        metrics=metrics,
        params={
            **model.get_params(),
            "FIT_SEASONS": FIT_SEASONS,
            "VAL_SEASON": VAL_SEASON,
            "TEST_SEASON": TEST_SEASON,
            "n_bins": N_BINS,
            "min_n": MIN_N,
        },
        tags={
            "model_type": MODEL_TYPE_TAGS[name],
            "stage": str(STAGE),
            "target": TARGET,
            "val_season": str(VAL_SEASON),
            "git_sha": get_git_sha(),
        },
        artifact_paths=artifact_paths,
    )