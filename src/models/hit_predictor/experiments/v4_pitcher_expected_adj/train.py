
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from models.hit_predictor.utils.eval import (
    plot_calibration_curve,
    run_pa_vs_game_grain_check,
)
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
from models.hit_predictor.processing.features import game_context


STAGE = Path(__file__).parent

# ---------------------------------------------------------------------------- #
#                                    v4 SUMMARY                                #
# ---------------------------------------------------------------------------- #
# Same base pipeline as v2_rolling_features (season + rolling batter/pitcher stats,
# park factors, TTO splits), PLUS everything built in the "game context" work this
# session:
#
#   1. Pitcher-role gating is now INNING-BASED, not PA-position-based. v2/v3 (and
#      the still-untouched season_stats.py/rolling_stats.py builders) gate every
#      pitcher-side merge on expected_pitcher_role/expected_pitcher_key_id, produced
#      by comparing a batter's estimated position in the lineup against the
#      starter's average BATTERS FACED per start (expected_role.assign_expected_
#      pitcher_role). v4 replaces that call with assign_expected_pitcher_role_by_inning:
#      the batter's estimated lineup position is converted into an estimated INNING
#      (a fixed avg-team-PAs-per-inning approximation), compared against the
#      starter's expected_start_innings — itself a shrinkage blend of his own
#      last-season baseline and this-season rolling average, not a single static
#      number. See processing/features/expected_role.py and game_context.py's
#      docstrings for the full reasoning; this is a genuinely different (and IMO
#      more principled) estimate, not a proven-better one yet — that's what this
#      experiment is for.
#
#   2. New game-context features, merged onto model_df same as park_factors already
#      is (via join keys, no grain changes needed):
#        - game_dt_* — calendar/time-of-day decomposition of game_datetime
#          (fastai add_datepart style: year/month/week/day/dayofweek/dayofyear,
#          month/quarter/year start-end flags, hour, minute). No day/night flag —
#          not in raw ingestion, and a UTC-hour heuristic would need a venue-
#          timezone lookup we don't have; deferred (see FEATURE_GLOSSARY.md).
#        - is_doubleheader — ground-truth from raw schedule's 'doubleheader' code
#          (not yet in processed_data/games/schedule's SCHEDULE_COLUMNS, so pulled
#          via a light, column-pruned raw read here rather than a full historical
#          reprocessing job).
#        - batting_team_roll_*/pitching_team_roll_* — team win/loss record and
#          run differential, season-to-date AND trailing-SHORT_WINDOW_GAMES, for
#          BOTH the batting team's and the pitching team's perspective (a team's
#          own record/form is a different signal depending on whether they're the
#          ones at bat or on the mound in this PA).
#        - batting_team_days_since_last_game/pitching_team_days_since_last_game —
#          rest days for each side.
#
# Everything else (batter/pitcher season+rolling stats, park factors, TTO splits,
# missing-indicator/impute/encode, model training, eval, MLflow logging) is
# unchanged from v2 — this experiment isolates the effect of (1) the gating swap
# and (2) the new game-context features together, against v2's raw-features
# baseline (see mlflow run history / prior session for v2's val metrics).

# Short rolling window size, in games. Change this one constant to try a different
# recent-form window — it flows through every build_*_rolling_stats(window=...) call below.
SHORT_WINDOW_GAMES = 10

# Shrinkage constant for build_expected_start_innings: weight = starts_n / (starts_n + k).
# At starts_n == k the blend is 50/50 between last season's baseline and this season's
# own emerging average. Same order of magnitude as SHORT_WINDOW_GAMES — roughly "half a
# rolling-window's worth of starts" before this season's own number carries equal weight.
STARTER_IP_SHRINKAGE_K = 5.0

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

# have to drop 2017 bc of lagged stats - can fill in later. Applies equally to the new
# starter-IP-based last-season stat (build_pitcher_start_ip_stats) — same root cause
# (needs a prior season's pbp, which isn't loaded for 2016) as the existing
# batters-faced depth stat this comment originally referred to.
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


def read_doubleheader_codes(seasons):
    """Light, single-purpose raw-schedule read: just gamepk + the 'doubleheader'
    code ('Y'/'N'/'S'), not the full raw table (~2 columns vs ~30, column-pruned
    at the parquet level — a few seconds/season, not the multi-GB full-table
    pulls that caused OOM issues earlier this session). See
    game_context.build_doubleheader_flag's docstring for why this comes from
    raw, not processed, schedule. Deduplicated the same way process_schedule()
    resolves rescheduled-game duplicate gamepks (sort by date, keep last).
    """
    frames = []
    for season in seasons:
        df = wr.s3.read_parquet(
            path=f"s3://{BUCKET}/raw_data/games/schedule/{season}/",
            boto3_session=boto_session,
            columns=["game_id", "game_date", "doubleheader"],
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).rename(columns={"game_id": "gamepk"})
    df["game_date"] = pd.to_datetime(df["game_date"])
    return (
        df.sort_values("game_date")
        .drop_duplicates(subset="gamepk", keep="last")
        [["gamepk", "doubleheader"]]
    )


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

print("\nLoading doubleheader codes (raw schedule, column-pruned)...")
doubleheader_codes = read_doubleheader_codes(all_boxscore_seasons)


game_info = pipeline.process_game_info(game_info)
pitcher_boxscore = pipeline.process_pitcher_boxscore(pitcher_boxscore)
schedule = pipeline.process_schedule(schedule)

pbp = pipeline.build_pbp_features(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

# ---------------------------------------------------------------------------- #
#                    INNING-BASED EXPECTED PITCHER ROLE (v4 change #1)         #
# ---------------------------------------------------------------------------- #
# Replaces v2/v3's assign_expected_pitcher_role (batters-faced-based) with
# assign_expected_pitcher_role_by_inning (innings-based) — see the v4 SUMMARY
# comment at the top of this file.

# Last-season fixed baseline (avg IP/start) + league-wide fallback — same
# fallback-chain role as season_stats.build_pitcher_start_depth_stats, but
# aggregating pitcher_boxscore's own ip column instead of pbp PA counts.
pitcher_start_ip_last_season = season_stats.build_pitcher_start_ip_stats(pitcher_boxscore, pbp)
league_avg_start_ip = season_stats.build_league_avg_start_ip(pitcher_start_ip_last_season)

# This-season rolling avg IP/start — thin/empty early in a season, which is
# exactly why the blend below shrinks toward it only as it accumulates.
pitcher_start_ip_this_season = game_context.build_pitcher_start_ip_this_season(pitcher_boxscore, pbp)

expected_start_innings = game_context.build_expected_start_innings(
    pitcher_start_ip_last_season, pitcher_start_ip_this_season, league_avg_start_ip,
    k=STARTER_IP_SHRINKAGE_K,
)

pa_outcome = expected_role.assign_expected_pitcher_role_by_inning(pa_outcome, expected_start_innings)

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
#                    NEW GAME-CONTEXT FEATURES (v4 change #2)                  #
# ---------------------------------------------------------------------------- #

# ------------------------------ DATETIME FEATURES ---------------------------- #
datetime_feats = game_context.build_datetime_features(schedule[['gamepk', 'game_datetime']])
dt_cols = [c for c in datetime_feats.columns if c.startswith('game_dt_')]
model_df = model_df.merge(datetime_feats[['gamepk'] + dt_cols], on='gamepk', how='left')

# ----------------------------- DOUBLEHEADER FLAG ----------------------------- #
model_df = model_df.merge(doubleheader_codes, on='gamepk', how='left')
model_df = game_context.build_doubleheader_flag(model_df)
# Raw 3-value code ('Y'/'N'/'S') dropped — is_doubleheader (bool) already carries
# the feature-relevant signal, and CAT_FEATS below doesn't include it, so it would
# otherwise just ride along unused.
model_df = model_df.drop(columns=['doubleheader'])

# ------------------------- TEAM WIN/LOSS RECORD + REST ----------------------- #
# Built once on the full schedule (every team, every game), then merged onto
# model_df TWICE — once for the batting team's perspective, once for the
# pitching team's — since a team's own record/rest is a different signal
# depending on which side of the ball they're on for this specific PA.
team_win_loss_season = game_context.build_team_win_loss_record(schedule, window='season')
team_win_loss_short = game_context.build_team_win_loss_record(schedule, window=SHORT_WINDOW_GAMES)
team_rest_days = game_context.build_team_rest_days(schedule)

team_context = team_win_loss_season.merge(
    team_win_loss_short.drop(columns=['game_date', 'game_datetime', 'game_season']),
    on=['team_id', 'gamepk'], how='left',
).merge(
    team_rest_days.drop(columns=['game_date', 'game_datetime', 'game_season']),
    on=['team_id', 'gamepk'], how='left',
).drop(columns=['game_date', 'game_datetime', 'game_season'])

for side, team_id_col in (('batting', 'batter_team_id'), ('pitching', 'pitcher_team_id')):
    side_context = team_context.rename(columns={'team_id': team_id_col})
    side_context = side_context.rename(columns={
        c: f'{side}_{c}' for c in side_context.columns if c not in (team_id_col, 'gamepk')
    })
    model_df = model_df.merge(side_context, on=[team_id_col, 'gamepk'], how='left')


# ---------------------------------------------------------------------------- #
#                                  TRAIN MODEL                                 #
# ---------------------------------------------------------------------------- #

# ------------------------------ DEFINE FEATURES ----------------------------- #
# game_season is deliberately excluded from NUM_FEATS: val/test seasons are always
# chronologically after every fit season, so a raw year feature would ask a tree to
# extrapolate past the range it was trained on rather than encode anything that
# generalizes. Same reasoning applies to the new game_dt_year column below — it's
# not hardcoded out (NUM_FEATS is dtype-based, not a hand-picked list), so if this
# turns out to matter, exclude it explicitly; noted here rather than silently
# assumed away, since unlike game_season it's easy to miss.
NON_FEATURE_COLS = {
    'gamepk', 'play_id', 'batter_id', 'pitcher_id', 'pitcher_team_id',
    'starting_pitcher_id', 'expected_pitcher_key_id',
    'batter_team_id', 'batter_team_name', 'pitcher_name', 'batter_name',
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
    # generalization check without spending any extra held-out data. n_jobs=2 (not -1)
    # — see v3_interaction_feats/train.py's comment: -1 OOM-killed a run on a
    # memory-constrained machine via too many parallel loky workers.
    "Random Forest": RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, max_features="sqrt",
        n_jobs=2, oob_score=True, random_state=42,
    ),
}

N_BINS = 10
MIN_N = 500
# Smaller than PA-grain MIN_N since there are ~4x fewer rows at game grain
# (~4.02 PA/game per BENCHMARKS.md §4.5) — same 500/200 split
# per_game_aggregation_check.py used for v2's game-grain check.
GAME_MIN_N = 200

PLOT_DIR = BASE_DIR / "plots" / "v4_pitcher_expected_adj"
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

# Sanity check: how often does the new inning-based estimate agree with the
# realized (post-hoc) pitcher_role? Same diagnostic role as the equivalent
# check implicitly available in v2/v3 via the retained-but-unused pitcher_role
# column — surfaced explicitly here since v4's whole point is testing whether
# this estimate is any good.
agreement_rate = (model_df['expected_pitcher_role'] == model_df['pitcher_role']).mean()
print(f"expected_pitcher_role (inning-based) vs realized pitcher_role agreement: {agreement_rate:.4f}")

for name, model in fitted_models.items():
    proba = model.predict_proba(X_val)[:, 1]

    # PA grain (existing eval) + game grain (P(1+ hits) via aggregation, see
    # BENCHMARKS.md §4.5) in one call — game grain is now the primary
    # evaluation target per ROADMAP.md's 2026-08-19 decision.
    metrics, game_metrics, game_results = run_pa_vs_game_grain_check(
        val_df, y_val, proba, n_bins=N_BINS, pa_min_n=MIN_N, game_min_n=GAME_MIN_N,
    )
    # Merged into the same metrics dict so log_evaluation_to_mlflow logs both
    # grains on one run; game-grain's own calibration_df is dropped here (the
    # PA-grain one is already logged as a CSV artifact below) rather than
    # colliding with the top-level 'calibration_df' key that function expects.
    metrics.update({
        f"game_grain_{k}": v for k, v in game_metrics.items() if k != "calibration_df"
    })

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
            "STARTER_IP_SHRINKAGE_K": STARTER_IP_SHRINKAGE_K,
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


