"""
k_predictor: total-strikeout count-distribution check ("does the plumbing work").
Run from src/models/k_predictor/ with: python experiments/count_distribution_check/run.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    SUMMARY                                   #
# ---------------------------------------------------------------------------- #
# k_predictor v2's PA classifier only ever scores REALIZED plate appearances —
# fine for training, useless for the real production target: a pitcher's TOTAL
# strikeout count this game vs. a DK over/under line. This script is the first,
# deliberately simple pass at closing that gap, per explicit direction to build
# the fixed-N version first and see where the gap actually is before paying for
# the more sophisticated one (simulating batters-faced uncertainty).
#
# Pipeline: for every 2024 SP start, take expected_batters_faced (this session's
# pitcher -> team -> league shrinkage cascade) as a FIXED point estimate, expand
# it into synthetic batter-slots cycling through that start's REALIZED lineup
# (game_context.build_batter_slot_expansion — a stand-in for real pregame lineup
# data, which this pipeline doesn't ingest anywhere yet), score each slot with
# v2's fitted LR model, and combine the N per-slot probabilities into an exact
# total-K distribution (count_distribution.poisson_binomial_pmf — closed-form,
# not simulation). Compare the predicted distribution against each start's
# REALIZED total K (pitcher_boxscore's own k column), and specifically check
# whether error grows with |realized batters faced - expected_batters_faced| —
# that's the direct evidence for whether the deferred N-simulation is worth
# building next.
#
# Deliberately NOT built here (flagged, not dropped): simulating N from its own
# distribution; real pregame lineup ingestion; de-vig/CLV vs actual DK lines.
# ---------------------------------------------------------------------------- #
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import expected_role
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.utils.eval import evaluate_hit_predictor

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip
from models.hit_predictor.utils.count_distribution import poisson_binomial_pmf, poisson_binomial_mixture_pmf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WHIP_SHRINKAGE_K = 20.0
PA_SHRINKAGE_K = 5.0
MAX_SLOTS = 45


# ── 1. Config + load (identical shape to v2_batter_and_team_features/train.py) ─
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


# ── 2. Build PA-grain frame + fit v2's feature set (same as v2's train.py) ─────
print("\nBuilding PA-grain DataFrame...")
schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)
# build_pitcher_start_pa_this_season (below) needs game_datetime to order same-date
# doubleheaders correctly, same reason every _rolling_sum caller does — pbp's own
# pipeline only ever adds game_date (see _add_pbp_game_date), never game_datetime,
# since every OTHER caller of this sort_col gets it from pitcher_boxscore instead
# (which already carries it as a prepared column). This is the first pbp-only
# caller, so it needs its own explicit merge from schedule here.
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
pbp_rolling_cols = ["pitcher_roll_season_pa_total", "pitcher_roll_season_pa_strikeout_rate"]
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
opp_team_rolling = team_batter_rolling.rename(
    columns={"team_roll_season_pa_strikeout_rate": "opp_team_roll_season_pa_strikeout_rate"}
)[["batter_team_id", "gamepk", "opp_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")
pitching_team_rolling = team_batter_rolling.rename(columns={
    "batter_team_id": "pitcher_team_id",
    "team_roll_season_pa_strikeout_rate": "pitching_team_roll_season_pa_strikeout_rate",
})[["pitcher_team_id", "gamepk", "pitching_team_roll_season_pa_strikeout_rate"]]
pa_outcome = pa_outcome.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")

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
    "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight", "weather_condition", "weather_temp", "expected_times_through_order",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in pa_outcome.columns]
model_df = pa_outcome[FEATURE_COLS + [TARGET, "game_season"]].copy()
model_df["game_season"] = model_df["game_season"].astype(int)
train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()

num_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(train_df[c])]
cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

num_imp = SimpleImputer(strategy="median")
cat_imp = SimpleImputer(strategy="most_frequent")
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
scaler = StandardScaler()

X_num = num_imp.fit_transform(pd.DataFrame(train_df[num_cols]).apply(pd.to_numeric, errors="coerce")) if num_cols else np.empty((len(train_df), 0))
if cat_cols:
    X_cat_imp = cat_imp.fit_transform(train_df[cat_cols].astype(object).fillna(np.nan))
    X_cat = enc.fit_transform(X_cat_imp)
else:
    X_cat = np.empty((len(train_df), 0))
Xtr = scaler.fit_transform(np.hstack([X_num, X_cat]))

print(f"\nFitting logistic regression on {len(train_df):,} PAs ({FIT_SEASONS})...")
lr = LogisticRegression(max_iter=1000)
lr.fit(Xtr, train_df[TARGET])


def score(df_feat):
    """Apply the SAME fitted imputers/encoder/scaler/model to any frame with
    FEATURE_COLS columns — real PA rows or synthetic slot rows alike."""
    df_feat = df_feat[FEATURE_COLS].copy()
    x_num = num_imp.transform(pd.DataFrame(df_feat[num_cols]).apply(pd.to_numeric, errors="coerce")) if num_cols else np.empty((len(df_feat), 0))
    if cat_cols:
        x_cat_imp = cat_imp.transform(df_feat[cat_cols].astype(object).fillna(np.nan))
        x_cat = enc.transform(x_cat_imp)
    else:
        x_cat = np.empty((len(df_feat), 0))
    x = scaler.transform(np.hstack([x_num, x_cat]))
    return lr.predict_proba(x)[:, 1]


# ── 3. Expected batters faced, val season 2024 SP starts only ─────────────────
print("\nBuilding expected_batters_faced cascade...")
pitcher_start_pa_last_season = season_stats.build_pitcher_start_pa_stats(pbp)
league_avg_start_pa = season_stats.build_league_avg_start_pa(pitcher_start_pa_last_season)
team_avg_start_pa = season_stats.build_team_avg_start_pa(pitcher_start_pa_last_season)
pitcher_start_pa_this_season = game_context.build_pitcher_start_pa_this_season(pbp)

expected_pa = game_context.build_expected_batters_faced(
    pitcher_start_pa_last_season, pitcher_start_pa_this_season, team_avg_start_pa, league_avg_start_pa, k=PA_SHRINKAGE_K,
)
pitcher_starts_2024 = expected_pa[expected_pa["game_season"] == VAL_SEASON].copy()
print(f"2024 SP starts with an expected_batters_faced estimate: {len(pitcher_starts_2024):,}")

# Game-level features (identical across every slot in a start): pitcher season/
# rolling/shrunk WHIP+K-rate, team K-rate (both sides), weather, throw hand.
pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_role_season_stats.rename(columns={"expected_pitcher_key_id": "personId"})[
        ["personId", "game_season", "pitcher_last_season_pa_strikeout_rate"]
    ],
    on=["personId", "game_season"], how="left",
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_box_season_stats.rename(columns={"expected_pitcher_key_id": "personId"})[["personId", "game_season", "pitcher_last_season_whip"]],
    on=["personId", "game_season"], how="left",
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_box_rolling.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + box_rolling_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_2024 = pitcher_starts_2024.merge(
    pitcher_pbp_rolling.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId"] + pbp_rolling_cols],
    on=["gamepk", "personId"], how="left",
)
pitcher_starts_2024["pitcher_roll_season_avg_ip_per_game"] = (
    pitcher_starts_2024["pitcher_roll_season_ip"] / pitcher_starts_2024["pitcher_roll_season_games_n"].replace(0, np.nan)
)
pitcher_starts_2024 = pitcher_starts_2024.merge(pitching_team_rolling, on=["pitcher_team_id", "gamepk"], how="left")
pitcher_starts_2024 = pitcher_starts_2024.merge(
    shrunk_whip.rename(columns={"pitcher_key_id": "personId"})[["gamepk", "personId", "pitcher_shrunk_whip", "pitcher_shrunk_whip_weight"]],
    on=["gamepk", "personId"], how="left",
)
throw_hand = pbp[["pitcher_id", "gamepk", "pitcher_throw_hand"]].drop_duplicates().rename(columns={"pitcher_id": "personId"})
pitcher_starts_2024 = pitcher_starts_2024.merge(throw_hand, on=["personId", "gamepk"], how="left")
weather = game_info[["gamepk", "weather_condition", "weather_temp"]].drop_duplicates("gamepk")
pitcher_starts_2024 = pitcher_starts_2024.merge(weather, on="gamepk", how="left")
pitcher_starts_2024["expected_pitcher_role"] = "sp"  # by construction — these ARE the realized starters


# ── 4. Expand to synthetic batter slots, attach batter + opp-team features ────
# FIXED 2026-09-02 (same bug/fix as score_2026_test_dates.py, see ROADMAP.md
# item 6(e) / v13_results.md): build_batter_slot_expansion merges
# batting_order onto (gamepk, lineup_position) with NO team-awareness. Every
# gamepk here has TWO starts (home SP, away SP), each needing a DIFFERENT
# team's 9 batters -- passing the full unscoped batting_order (as this script
# always has) let roughly half of all synthetic slots collide against the
# WRONG team (a pitcher's own teammates, whom he never actually faces).
# pitcher_starts_2024 (from the expected_batters_faced cascade) only carries
# pitcher_team_id, not home_id/away_id/opp_team_id -- derive those first, the
# same way score_2026_test_dates.py already does, then team-scope
# batting_order per start (expand home-team and away-team starters
# separately, each unambiguous against the correctly opposing lineup).
print("Expanding to synthetic batter slots...")
batting_order = hp_pipeline._create_batting_order(batter_boxscore)
batter_team_lookup = batter_boxscore[["gamepk", "personId", "team_id"]].rename(
    columns={"personId": "batter_id", "team_id": "batter_team_id"}
).drop_duplicates()

schedule_teams = schedule[["gamepk", "home_id", "away_id"]].drop_duplicates("gamepk")
pitcher_starts_2024 = pitcher_starts_2024.merge(schedule_teams, on="gamepk", how="left")
pitcher_starts_2024["opp_team_id"] = np.where(
    pitcher_starts_2024["pitcher_team_id"] == pitcher_starts_2024["home_id"],
    pitcher_starts_2024["away_id"], pitcher_starts_2024["home_id"],
)

batting_order_with_team = batting_order.merge(
    batter_team_lookup.rename(columns={"batter_team_id": "team_id"}), on=["gamepk", "batter_id"], how="left",
)


def opp_scoped_batting_order(starts_subset):
    context = starts_subset[["gamepk", "opp_team_id"]].drop_duplicates()
    scoped = context.merge(batting_order_with_team, on="gamepk", how="left")
    scoped = scoped[scoped["team_id"] == scoped["opp_team_id"]]
    return scoped[["gamepk", "batter_id", "batting_order"]]


starts_home_sp = pitcher_starts_2024[pitcher_starts_2024["pitcher_team_id"] == pitcher_starts_2024["home_id"]]
starts_away_sp = pitcher_starts_2024[pitcher_starts_2024["pitcher_team_id"] == pitcher_starts_2024["away_id"]]

slots_home = game_context.build_batter_slot_expansion(starts_home_sp, opp_scoped_batting_order(starts_home_sp), max_slots=MAX_SLOTS)
slots_away = game_context.build_batter_slot_expansion(starts_away_sp, opp_scoped_batting_order(starts_away_sp), max_slots=MAX_SLOTS)
slots = pd.concat([slots_home, slots_away], ignore_index=True)
slots = slots.merge(batter_team_lookup, on=["gamepk", "batter_id"], how="left")

bat_side = pbp[["batter_id", "gamepk", "batter_bat_side"]].drop_duplicates()
slots = slots.merge(bat_side, on=["batter_id", "gamepk"], how="left")
slots = slots.merge(
    batter_season_stats, on=["batter_id", "game_season"], how="left",
)
slots = slots.merge(
    batter_pbp_rolling[["batter_id", "gamepk", "batter_roll_season_pa_strikeout_rate"]],
    on=["batter_id", "gamepk"], how="left",
)
slots = slots.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")

print(f"{len(pitcher_starts_2024):,} starts -> {len(slots):,} synthetic batter-slots "
      f"(mean {len(slots) / max(len(pitcher_starts_2024), 1):.1f} slots/start)")


# ── 5. Score each slot, combine into a total-K distribution per start ─────────
print("Scoring synthetic slots...")
slots["k_prob"] = score(slots)

print("Combining via exact Poisson-binomial...")
results = []
for (gamepk, person_id), grp in slots.groupby(["gamepk", "personId"]):
    probs = grp["k_prob"].to_numpy()
    pmf = poisson_binomial_pmf(list(probs))
    results.append({
        "gamepk": gamepk, "personId": person_id, "n_slots": len(probs),
        "predicted_mean_k": probs.sum(), "pmf": pmf,
    })
pred_df = pd.DataFrame(results)


# ── 6. Compare against realized total K + realized batters faced ──────────────
print("Comparing against realized outcomes...")
role_lookup = season_stats._pitcher_role_lookup(pbp)[["gamepk", "pitcher_id", "pitcher_role"]].rename(columns={"pitcher_id": "personId"})
pitcher_box_tagged = pitcher_boxscore.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
).merge(
    role_lookup.assign(personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str)),
    on=["gamepk", "personId"], how="left",
)
realized_k = pitcher_box_tagged[
    (pitcher_box_tagged["pitcher_role"] == "sp") & (pitcher_box_tagged["game_season"] == VAL_SEASON)
][["personId", "gamepk", "k"]].rename(columns={"k": "realized_k"})

sp_pbp = pbp[pbp["pitcher_role"] == "sp"]
realized_pa = rolling_stats._pitcher_pa_outcome_per_game(sp_pbp, entity_col="pitcher_id")[
    ["pitcher_id", "gamepk", "game_season", "pa_total"]
].rename(columns={"pitcher_id": "personId", "pa_total": "realized_batters_faced"})
realized_pa = realized_pa[realized_pa["game_season"] == VAL_SEASON]

pred_df = pred_df.merge(realized_k, on=["personId", "gamepk"], how="inner")
pred_df = pred_df.merge(realized_pa[["personId", "gamepk", "realized_batters_faced"]], on=["personId", "gamepk"], how="left")
pred_df = pred_df.merge(
    pitcher_starts_2024[["personId", "gamepk", "expected_batters_faced", "expected_batters_faced_weight"]],
    on=["personId", "gamepk"], how="left",
)
pred_df["bf_gap"] = (pred_df["realized_batters_faced"] - pred_df["expected_batters_faced"]).abs()
pred_df["abs_error"] = (pred_df["predicted_mean_k"] - pred_df["realized_k"]).abs()

# ── Does expected_batters_faced_weight (starts_n / (starts_n + k), 0=season-
# opener/pure-baseline, ->1 as this pitcher's own current-season sample grows)
# actually predict bf_gap size? If low-weight (thin-sample) starts show bigger
# misses than high-weight starts, that validates using this weight as the
# conditioning variable for a future N-simulation's per-start spread, rather
# than resampling the pooled/unconditional realized-BF distribution (which
# would just recover the population's average uncertainty for every start).
print(f"\n{'=' * 72}\nDOES expected_batters_faced_weight PREDICT bf_gap SIZE?\n"
      f"(low weight = thin this-season sample, leans on last-season/team/league\n"
      f"baseline -- if bf_gap is bigger there, the weight is a real conditioning\n"
      f"signal for a future N-simulation's per-start spread, not just noise)\n{'=' * 72}")
corr = pred_df[["expected_batters_faced_weight", "bf_gap"]].corr().iloc[0, 1]
print(f"Correlation(expected_batters_faced_weight, bf_gap): {corr:.3f}  (negative = weight predicts smaller gap, as expected)")

print(f"\n{'weight quartile':<20} {'n':>6} {'mean_weight':>12} {'mean_bf_gap':>12} {'mean_abs_error':>15}")
pred_df["weight_q"] = pd.qcut(pred_df["expected_batters_faced_weight"], 4, labels=["Q1 (thinnest)", "Q2", "Q3", "Q4 (most reliable)"], duplicates="drop")
for q, grp in pred_df.groupby("weight_q", observed=True):
    print(f"{str(q):<20} {len(grp):>6} {grp['expected_batters_faced_weight'].mean():>12.3f} {grp['bf_gap'].mean():>12.3f} {grp['abs_error'].mean():>15.3f}")


# ── LAYER A METRICS — point-estimate accuracy of expected_batters_faced itself ─
# Story 0 of the batters-faced-distribution plan (see ROADMAP.md): the checks
# above all measure error in the DOWNSTREAM total-K prediction. This measures
# the batters-faced ESTIMATE directly against what actually happened -- MAE,
# RMSE (penalizes large misses more than MAE), Bias (signed -- systematic
# over/under-prediction, not just magnitude), and Pearson r, overall and by
# expected_batters_faced_weight quartile. This is the "before" baseline the
# new empirical-residual-distribution method (Stories 1-3) will be compared
# against.
print(f"\n{'=' * 72}\nLAYER A METRICS — expected_batters_faced point-estimate accuracy vs.\nrealized_batters_faced (batters-faced-distribution plan, Story 0 baseline)\n{'=' * 72}")

bf_valid = pred_df.dropna(subset=["realized_batters_faced"]).copy()
bf_valid["bf_error"] = bf_valid["realized_batters_faced"] - bf_valid["expected_batters_faced"]
bf_valid["bf_abs_error"] = bf_valid["bf_error"].abs()


def layer_a_metrics(df):
    return {
        "n": len(df),
        "mae": df["bf_abs_error"].mean(),
        "rmse": np.sqrt((df["bf_error"] ** 2).mean()),
        "bias": df["bf_error"].mean(),
        "pearson_r": df[["expected_batters_faced", "realized_batters_faced"]].corr().iloc[0, 1],
    }


overall_a = layer_a_metrics(bf_valid)
print(f"\nOverall (n={overall_a['n']:,}): MAE={overall_a['mae']:.3f}  RMSE={overall_a['rmse']:.3f}  "
      f"Bias={overall_a['bias']:+.3f}  Pearson r={overall_a['pearson_r']:.3f}")

print(f"\n{'weight quartile':<20} {'n':>6} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Pearson r':>10}")
for q, grp in bf_valid.groupby("weight_q", observed=True):
    m = layer_a_metrics(grp)
    print(f"{str(q):<20} {m['n']:>6} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['bias']:>+8.3f} {m['pearson_r']:>10.3f}")

print(f"\n{'=' * 72}\nTOTAL-K COUNT-DISTRIBUTION CHECK — val season {VAL_SEASON}, {len(pred_df):,} starts\n{'=' * 72}")
print(f"Predicted mean K:  {pred_df['predicted_mean_k'].mean():.3f}")
print(f"Realized mean K:   {pred_df['realized_k'].mean():.3f}")
print(f"MAE (predicted mean vs. realized): {pred_df['abs_error'].mean():.3f}")
print(f"Mean |realized batters faced - expected_batters_faced|: {pred_df['bf_gap'].mean():.3f}")

print(f"\n{'bf_gap quartile':<20} {'n':>6} {'MAE':>8}")
pred_df["bf_gap_q"] = pd.qcut(pred_df["bf_gap"], 4, labels=["Q1 (closest)", "Q2", "Q3", "Q4 (furthest)"], duplicates="drop")
for q, grp in pred_df.groupby("bf_gap_q", observed=True):
    print(f"{str(q):<20} {len(grp):>6} {grp['abs_error'].mean():>8.3f}")

# Reuse evaluate_hit_predictor's reliability/resolution decomposition on a
# concrete threshold question — P(total K > line) vs. realized outcome — the
# same two-tier evaluation posture (PR-AUC-style diagnostics + reliability/
# resolution decision metrics) this project uses everywhere else.
LINE = round(pred_df["realized_k"].median()) + 0.5
print(f"\n{'=' * 72}\nTHRESHOLD CHECK — P(total K > {LINE}) vs. realized, same decomposition as every\nother model in this project (reliability/resolution, not just MAE)\n{'=' * 72}")


def p_over_line(pmf, line):
    k_over = int(np.floor(line)) + 1
    return pmf[k_over:].sum() if k_over < len(pmf) else 0.0


pred_df["p_over_line"] = pred_df["pmf"].apply(lambda pmf: p_over_line(pmf, LINE))
pred_df["realized_over_line"] = (pred_df["realized_k"] > LINE).astype(int)

metrics = evaluate_hit_predictor(
    y_true=pred_df["realized_over_line"], y_prob=pred_df["p_over_line"],
    n_bins=8, min_n=30, base_rate=pred_df["realized_over_line"].mean(),
)


# ── 7. Coverage check — is the predicted distribution honestly wide? ──────────
# Everything above tests whether the CENTER of the predicted pmf is right
# (predicted mean K vs realized, threshold calibration). None of it tests
# whether the SPREAD is right. This pmf is built by treating N (batters
# faced) as certain -- all of its spread comes from batter-to-batter K
# randomness, none from "did he get pulled early or go deep." If that's a
# real gap, the predicted pmf should be too NARROW: a nominal 80% interval
# should capture realized K less than 80% of the time. Cheap to check --
# reuses the pmf we already computed, no new modeling.
#
# Two things worth keeping straight about what this pmf actually is (see
# ROADMAP.md's 2026-08-26 note for the fuller version): (1) N is a fixed
# point estimate going in -- every bit of spread in the pmf comes from not
# knowing WHICH batters whiff, none of it from not knowing HOW MANY at-bats
# there will be, which is precisely the gap this check is testing for; (2)
# poisson_binomial_pmf is pure combinatorial bookkeeping -- it adds no
# information beyond the N per-slot probabilities score() already produced,
# so the pmf's honesty here is really a statement about score()'s honesty.
print(f"\n{'=' * 72}\nCOVERAGE CHECK — is the predicted total-K distribution honestly wide?\n"
      f"(for each start, does the nominal X% predicted interval actually contain\n"
      f"the realized K about X% of the time? if the true rate is well BELOW\n"
      f"nominal, the pmf is too narrow -- overconfident -- because it treats N\n"
      f"as certain when we've shown it often isn't)\n{'=' * 72}")


def interval_bounds(pmf, level):
    """Smallest [lower, upper] such that the pmf's cumulative mass covers
    at least `level` of probability, centered (equal tail mass each side)."""
    cdf = np.cumsum(pmf)
    alpha = (1 - level) / 2
    lower = int(np.searchsorted(cdf, alpha, side="left"))
    upper = int(np.searchsorted(cdf, 1 - alpha, side="left"))
    return lower, upper


LEVELS = [0.50, 0.80, 0.95]
print(f"{'level (nominal)':<18} {'n':>6} {'empirical coverage':>20} {'gap (empirical - nominal)':>28}")
for level in LEVELS:
    bounds = pred_df["pmf"].apply(lambda pmf: interval_bounds(pmf, level))
    lower = bounds.apply(lambda t: t[0])
    upper = bounds.apply(lambda t: t[1])
    covered = (pred_df["realized_k"] >= lower) & (pred_df["realized_k"] <= upper)
    pred_df[f"covered_{int(level * 100)}"] = covered
    emp = covered.mean()
    print(f"{level:<18.0%} {len(pred_df):>6} {emp:>20.1%} {emp - level:>+28.1%}")

print(f"\n80% interval coverage by expected_batters_faced_weight quartile "
      f"(thin-sample starts should under-cover more if N-uncertainty is the driver):")
print(f"{'weight quartile':<20} {'n':>6} {'80% coverage':>14}")
for q, grp in pred_df.groupby("weight_q", observed=True):
    print(f"{str(q):<20} {len(grp):>6} {grp['covered_80'].mean():>14.1%}")


# ── 8. Dump a few real worked examples for a visual walkthrough (not part of ──
# the diagnostic itself — just exports real numbers from real 2024 starts so
# an explanation of the coverage check can use an actual example instead of a
# made-up one). Not saved anywhere the pipeline reads back.
import json

name_lookup = pitcher_boxscore[["personId", "player_name"]].drop_duplicates("personId")
examples_df = pred_df.merge(name_lookup, on="personId", how="left")
examples_df = examples_df.merge(
    pitcher_starts_2024[["personId", "gamepk", "expected_batters_faced_weight"]],
    on=["personId", "gamepk"], how="left", suffixes=("", "_dup"),
)

interval_80 = examples_df["pmf"].apply(lambda pmf: interval_bounds(pmf, 0.80))
examples_df["lower_80"] = interval_80.apply(lambda t: t[0])
examples_df["upper_80"] = interval_80.apply(lambda t: t[1])

def pick_example(df, covered, weight_bucket=None):
    pool = df[df["covered_80"] == covered]
    if weight_bucket == "thin":
        pool = pool[pool["expected_batters_faced_weight"] < 0.3]
    elif weight_bucket == "reliable":
        pool = pool[pool["expected_batters_faced_weight"] > 0.7]
    pool = pool[pool["n_slots"].between(18, 26)]  # typical slot count, easier to visualize
    row = pool.sample(1, random_state=7).iloc[0]
    return {
        "player_name": row["player_name"], "gamepk": str(row["gamepk"]), "personId": str(row["personId"]),
        "n_slots": int(row["n_slots"]), "expected_batters_faced": round(float(row["expected_batters_faced"]), 2),
        "expected_batters_faced_weight": round(float(row["expected_batters_faced_weight"]), 3),
        "realized_batters_faced": int(row["realized_batters_faced"]) if pd.notnull(row["realized_batters_faced"]) else None,
        "realized_k": int(row["realized_k"]), "predicted_mean_k": round(float(row["predicted_mean_k"]), 2),
        "lower_80": int(row["lower_80"]), "upper_80": int(row["upper_80"]), "covered_80": bool(row["covered_80"]),
        "pmf": [round(float(p), 5) for p in row["pmf"]],
    }

worked_examples = {
    "covered_example": pick_example(examples_df, covered=True, weight_bucket="reliable"),
    "miss_example": pick_example(examples_df, covered=False),
    "aggregate": {
        "n_starts": int(len(pred_df)),
        "coverage_by_level": {str(int(l * 100)): round(float(pred_df[f"covered_{int(l*100)}"].mean()), 4) for l in LEVELS},
        "coverage_by_weight_quartile_80": {
            str(q): round(float(grp["covered_80"].mean()), 4) for q, grp in pred_df.groupby("weight_q", observed=True)
        },
        "bf_gap_mae_by_quartile": {
            str(q): round(float(grp["abs_error"].mean()), 4) for q, grp in pred_df.groupby("bf_gap_q", observed=True)
        },
    },
}
out_path = "/private/tmp/claude-501/-Users-clumanlan-projects-mlb-src-models/cdc23926-c06e-4064-8647-e9bddfcdf79a/scratchpad/coverage_check_examples.json"
with open(out_path, "w") as f:
    json.dump(worked_examples, f, indent=2)
print(f"\nWorked examples written to {out_path}")

print(f"\n{'=' * 72}\nSaved nothing (diagnostic-only check) — record the outcome in ROADMAP.md.\n{'=' * 72}")
