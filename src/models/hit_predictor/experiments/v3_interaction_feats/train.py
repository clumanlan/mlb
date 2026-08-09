
import time
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

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
from models.hit_predictor.processing.features import park_factors
from models.hit_predictor.processing.features import expected_role
from models.hit_predictor.processing.features import interaction_feats
from models.hit_predictor.processing.features.interaction_feats import (
    find_rolling_trend_pairs,
    find_sample_size_col,
)


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
# all_boxscore_seasons (not just TRAIN_SEASONS) — park factors need a prior
# season's schedule+boxscore data to compute the shifted factor that joins
# onto the earliest fit season, same reason batter/pitcher boxscore already
# load all_boxscore_seasons above.
schedule = read_parquet_seasons(
    "s3://{bucket}/processed_data/games/schedule/{season}/",
    all_boxscore_seasons,
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

# Realized pitcher_role reflects who ACTUALLY pitched to a PA — only knowable after the
# fact (a manager might pull his starter after 4 innings or let him go 8), so it leaks
# in-game information a pre-game model wouldn't have. expected_pitcher_role/
# expected_pitcher_key_id replace it: pre-game-knowable estimates built from the specific
# starter's own historical depth (how many batters he typically faces per start) compared
# against this batter's estimated position in the lineup. Applied to historical rows too,
# not just future ones, so training doesn't rely on precision (the true realized role)
# that won't be available at serving time.
pitcher_start_depth_stats = season_stats.build_pitcher_start_depth_stats(pbp)
league_avg_start_depth = season_stats.build_league_avg_start_depth(pitcher_start_depth_stats)
pa_outcome = expected_role.assign_expected_pitcher_role(
    pa_outcome, pitcher_start_depth_stats, league_avg_start_depth
)

batter_season_stats = season_stats.build_batter_stats(batter_boxscore).rename(columns={'personId': 'batter_id'})
# Role-tagged AND role-pooled: a swingman (starts some games, relieves others in the same
# season) gets separate 'sp'/'bullpen' rows rather than one blended aggregate, and the
# bullpen half is pooled by team (not kept per individual pitcher_id) since a specific
# reliever's identity isn't knowable pre-game. Both output a pitcher_key_id/pitcher_role
# pair, renamed to expected_pitcher_key_id/expected_pitcher_role immediately below so they
# join onto pa_outcome's new expected-role columns uniformly regardless of role — these
# tables are still built from REALIZED pitcher_role internally (a pitcher's own aggregate
# should reflect his true starts, not a guess), only the column names used to JOIN them
# onto model_df change.
pitcher_role_season_stats = season_stats.build_pbp_pitcher_feats_all_roles(pbp).rename(
    columns={'pitcher_key_id': 'expected_pitcher_key_id', 'pitcher_role': 'expected_pitcher_role'}
)
pitcher_season_stats = season_stats.build_pitcher_stats_all_roles(pitcher_boxscore, pbp).rename(
    columns={'pitcher_key_id': 'expected_pitcher_key_id', 'pitcher_role': 'expected_pitcher_role'}
)

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

ROLE_KEY_RENAME = {'pitcher_key_id': 'expected_pitcher_key_id', 'pitcher_role': 'expected_pitcher_role'}

# Role-tagged AND role-pooled, same rationale as pitcher_season_stats above.
pitcher_rolling_season_stats = rolling_stats.build_pitcher_rolling_stats_all_roles(
    pitcher_boxscore, pbp, window='season'
).rename(columns=ROLE_KEY_RENAME)
pitcher_rolling_short_stats = rolling_stats.build_pitcher_rolling_stats_all_roles(
    pitcher_boxscore, pbp, window=SHORT_WINDOW_GAMES
).rename(columns=ROLE_KEY_RENAME)

# Same role-tagged, role-pooled pattern as pitcher_role_season_stats above, rolling instead of season.
pitcher_role_rolling_season_stats = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(
    pbp, window='season'
).rename(columns=ROLE_KEY_RENAME)
pitcher_role_rolling_short_stats = rolling_stats.build_pbp_pitcher_rolling_feats_all_roles(
    pbp, window=SHORT_WINDOW_GAMES
).rename(columns=ROLE_KEY_RENAME)

# v1 park factor: single-season (trailing 1 year) venue hit-rate index, same
# _shift_to_last_season convention as every other season-level feature here.
# Keyed by (game_season, venue_id) rather than a player entity.
venue_park_factors = park_factors.build_park_factors(schedule, batter_boxscore)

# Phase 2: a starter's PA-outcome stats split by 1st/2nd/3rd+ time through the order
# (Lichtman TTOP research), plus a league-wide pooled companion for the thin tto3plus
# per-pitcher sample. Attached unconditionally (every column, every 'sp' PA) rather than
# picked per-PA — the model combines these with expected_times_through_order (phase 1)
# itself to learn which bucket applies to a given PA.
pitcher_tto_stats = season_stats.build_pbp_pitcher_feats_by_times_through_order(pbp).rename(
    columns=ROLE_KEY_RENAME
)
league_tto_stats = season_stats.build_league_times_through_order_stats(pbp)


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
# expected_pitcher_key_id + expected_pitcher_role join as the key: pa_outcome carries each
# PA's pre-game-estimated role and resolved key (sp -> starting_pitcher_id, bullpen ->
# pitcher_team_id), so this picks up exactly the sp/team-pooled-bullpen row matching that
# expected role.
model_df = model_df.merge(
    pitcher_season_stats, on=['game_season', 'expected_pitcher_key_id', 'expected_pitcher_role'], how='left'
)
model_df = model_df.merge(
    pitcher_role_season_stats, on=['game_season', 'expected_pitcher_key_id', 'expected_pitcher_role'], how='left'
)
model_df = model_df.merge(
    venue_park_factors, on=['game_season', 'venue_id'], how='left'
)
# TTO-split table is inherently sp-only (no bullpen rows exist to match), so bullpen-role
# PAs naturally get NaN here via the same 3-key merge — no special-casing needed.
model_df = model_df.merge(
    pitcher_tto_stats, on=['game_season', 'expected_pitcher_key_id', 'expected_pitcher_role'], how='left'
)
# Population-level table — no entity key, joins on game_season alone.
model_df = model_df.merge(league_tto_stats, on='game_season', how='left')

for rolling_df in (batter_rolling_season_stats, batter_rolling_short_stats):
    rolling_df = rolling_df.drop(columns=['game_date', 'game_season'])
    model_df = model_df.merge(rolling_df, on=['gamepk', 'batter_id'], how='left')

# Role-tagged (and for bullpen, role-pooled) rolling tables need expected_pitcher_key_id +
# expected_pitcher_role in the merge key, so they can't share the batter loop above.
for rolling_df in (
    pitcher_rolling_season_stats, pitcher_rolling_short_stats,
    pitcher_role_rolling_season_stats, pitcher_role_rolling_short_stats,
):
    rolling_df = rolling_df.drop(columns=['game_date', 'game_season'])
    model_df = model_df.merge(
        rolling_df, on=['gamepk', 'expected_pitcher_key_id', 'expected_pitcher_role'], how='left'
    )

# A specific reliever's throwing hand isn't knowable pre-game either (same reasoning as
# the stat pooling above) — null it for expected-bullpen rows so the existing
# missing-indicator + "missing"-sentinel-category machinery treats it as genuinely unknown
# rather than training on the actual (post-hoc-only-known) hand of whoever happened to relieve.
model_df.loc[model_df['expected_pitcher_role'] == 'bullpen', 'pitcher_throw_hand'] = None


# ---------------------------------------------------------------------------- #
#                         ENGINEERED INTERACTION FEATURES                      #
# ---------------------------------------------------------------------------- #
# An earlier run of this script trained a Random Forest on model_df's raw columns
# only, then PDP-checked (see git history / mlflow run_type=pdp_diagnostics) whether
# it was already learning (a) a hot/cold "recent form vs. season baseline" signal
# and (b) a "trust a rate less when its sample size is small" signal from the raw
# rolling columns alone. Neither showed up as a real interaction — both PDP surfaces
# looked close to additive — so both get built as explicit columns here instead.
trend_pairs = find_rolling_trend_pairs(list(model_df.columns))
model_df = interaction_feats.build_trend_features(model_df, trend_pairs)

rate_to_sample_col = {}
for short_col, _ in trend_pairs:
    sample_col = find_sample_size_col(list(model_df.columns), short_col)
    if sample_col:
        rate_to_sample_col[short_col] = sample_col
model_df = interaction_feats.build_shrinkage_weight_features(model_df, rate_to_sample_col)


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
    'gamepk', 'play_id', 'batter_id', 'pitcher_id', 'pitcher_team_id',
    'starting_pitcher_id', 'expected_pitcher_key_id',
    'batter_team_name', 'pitcher_name', 'batter_name',
    # venue_id is a join key for park_last_season_hit_factor/park_last_season_ab, not a
    # feature itself — as a raw numeric ID it has no ordinal meaning and would just let
    # a tree memorize venue identity instead of using the actual hit-factor signal.
    'venue_id',
    # realized pitcher_role is kept in the frame (needed to sanity-check
    # expected_pitcher_role's agreement rate against it) but excluded as a feature — it's
    # exactly the leak this refactor removes, so it can't ALSO ride along as a raw
    # categorical the model can see directly.
    'pitcher_role',
    'game_date', 'game_season', TARGET,
}
CAT_FEATS = ['pitcher_throw_hand', 'batter_bat_side', 'weather_condition', 'expected_pitcher_role']
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
        # n_jobs=-1 (one worker process per CPU core) OOM-killed this run on a
        # memory-constrained machine — each loky worker holds its own overhead
        # while fitting trees in parallel over the ~675k-row training set, on
        # top of everything else already resident. Capped to 2 to trade fit
        # speed for reliably finishing; bump back up once memory isn't tight.
        n_jobs=2, oob_score=True, random_state=42,
    ),
}

N_BINS = 10
MIN_N = 500

PLOT_DIR = BASE_DIR / "plots" / "v3_interaction_feats"
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


# ---------------------------------------------------------------------------- #
#                         PDP INTERACTION DIAGNOSTICS                          #
# ---------------------------------------------------------------------------- #
# Before hand-engineering an interaction feature, check whether the Random Forest
# fit above (raw features only, no engineered interactions) already learns it via
# 2-way partial dependence:
#   - "trend" pairs: a stat's short-window value vs. its season-long value (e.g.
#     batter_roll_last10g_ba x batter_roll_season_ba) — does the model already
#     combine these non-additively, the way an explicit trend_diff = short - season
#     feature would encode directly?
#   - "shrinkage" pairs: a short-window rate vs. its own sample-size denominator
#     (e.g. batter_roll_last10g_ba x batter_roll_last10g_plate_appearances, the
#     latter newly surviving into the feature set as of this experiment — see
#     rolling_stats.py) — does the model already learn to discount a noisy
#     small-sample short-window rate?
# A flat/axis-aligned 2-way PDP surface for a pair is evidence the model ISN'T
# capturing that interaction on its own and it's worth engineering explicitly in
# processing/features/interaction_feats.py; a visibly twisted/diagonal surface
# means it's already there and an explicit feature would likely be redundant.
# This step only produces diagnostic plots — no new features are added to the
# model in this run. Restricted to pairs touching the Random Forest's top-N most
# important raw features, since those are the only ones worth the (slow) cost of
# 2-way PDP computation.

rf_model = fitted_models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=FEATURE_NAMES)
TOP_N_IMPORTANT = 20
top_features = set(importances.sort_values(ascending=False).head(TOP_N_IMPORTANT).index)

trend_pairs = [
    (short_col, season_col)
    for short_col, season_col in find_rolling_trend_pairs(FEATURE_NAMES)
    if short_col in top_features or season_col in top_features
]

PDP_DIR = PLOT_DIR / "pdp"
PDP_DIR.mkdir(parents=True, exist_ok=True)

# RandomForestClassifier only supports sklearn's brute-force PDP method (no
# 'recursion' shortcut like GradientBoosting gets), which re-predicts over
# every row of whatever X is passed in for every grid point — on the full
# multi-million-row X_train that's hours per pair. A few thousand rows is
# plenty to estimate the average partial-dependence surface; grid_resolution
# is similarly capped well below sklearn's default of 100 per axis.
PDP_SAMPLE_SIZE = 3000
PDP_GRID_RESOLUTION = 20
pdp_background = X_train.sample(n=min(PDP_SAMPLE_SIZE, len(X_train)), random_state=42)
print(f"\nPDP background sample: {len(pdp_background)} rows (of {len(X_train)} total train rows), "
      f"grid_resolution={PDP_GRID_RESOLUTION}")
print(f"Trend pairs to check: {len(trend_pairs)}")

pdp_artifact_paths = []
for i, (short_col, season_col) in enumerate(trend_pairs, start=1):
    t0 = time.time()
    fig, ax = plt.subplots(figsize=(6, 5))
    PartialDependenceDisplay.from_estimator(
        rf_model, pdp_background, [(short_col, season_col)], ax=ax,
        grid_resolution=PDP_GRID_RESOLUTION,
    )
    trend_plot_path = PDP_DIR / f"trend_{short_col}_x_{season_col}.png"
    fig.savefig(trend_plot_path, bbox_inches="tight")
    plt.close(fig)
    pdp_artifact_paths.append(str(trend_plot_path))
    print(f"  [{i}/{len(trend_pairs)}] trend {short_col} x {season_col} -> {trend_plot_path.name} "
          f"({time.time() - t0:.1f}s)", flush=True)

    sample_col = find_sample_size_col(FEATURE_NAMES, short_col)
    if sample_col:
        t0 = time.time()
        fig, ax = plt.subplots(figsize=(6, 5))
        PartialDependenceDisplay.from_estimator(
            rf_model, pdp_background, [(short_col, sample_col)], ax=ax,
            grid_resolution=PDP_GRID_RESOLUTION,
        )
        shrinkage_plot_path = PDP_DIR / f"shrinkage_{short_col}_x_{sample_col}.png"
        fig.savefig(shrinkage_plot_path, bbox_inches="tight")
        plt.close(fig)
        pdp_artifact_paths.append(str(shrinkage_plot_path))
        print(f"  [{i}/{len(trend_pairs)}] shrinkage {short_col} x {sample_col} -> {shrinkage_plot_path.name} "
              f"({time.time() - t0:.1f}s)", flush=True)

print(f"\nSaved {len(pdp_artifact_paths)} PDP diagnostic plots to {PDP_DIR}")
print("Next step: inspect these plots. For any pair whose 2-way PDP surface is")
print("flat/axis-aligned (no visible interaction), that trend or shrinkage signal")
print("likely isn't being learned implicitly by the tree — add it as an explicit")
print("engineered feature in processing/features/interaction_feats.py and re-train.")
print("Skip pairs where the surface already shows the effect (redundant to add).")

if pdp_artifact_paths:
    with mlflow.start_run():
        mlflow.set_tags({
            "stage": str(STAGE),
            "run_type": "pdp_diagnostics",
            "git_sha": get_git_sha(),
        })
        for path in pdp_artifact_paths:
            mlflow.log_artifact(path, artifact_path="pdp")
