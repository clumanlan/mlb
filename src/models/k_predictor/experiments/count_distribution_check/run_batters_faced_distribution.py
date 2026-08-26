"""
k_predictor: total-strikeout count-distribution check, DISTRIBUTION-based batters faced.
Run from src/models/k_predictor/ with: python experiments/count_distribution_check/run_batters_faced_distribution.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    SUMMARY                                   #
# ---------------------------------------------------------------------------- #
# Sibling to run.py (the fixed-N version) — a separate frozen script, not an edit
# in place, so both results stay independently re-runnable/diffable, per this
# repo's versioned-experiment convention. Reuses everything about run.py's data
# load and v2 LR fit verbatim, but replaces the FIXED expected_batters_faced point
# estimate with a real distribution (game_context.build_batters_faced_residual_bins
# + build_batters_faced_distribution — empirical residual histograms binned by
# expected_batters_faced_weight, fit on FIT_SEASONS, applied to VAL_SEASON), and
# combines per-slot K probabilities via poisson_binomial_mixture_pmf instead of
# the fixed-N poisson_binomial_pmf. See ROADMAP.md's batters-faced-distribution
# plan and run.py's own Story-0 baseline run for the "before" numbers this
# compares against.
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
from models.hit_predictor.utils.count_distribution import poisson_binomial_mixture_pmf

import models.k_predictor.processing.pipeline as pipeline
from models.k_predictor.processing.features.pitcher_workload import build_pitcher_shrunk_whip

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WHIP_SHRINKAGE_K = 20.0
PA_SHRINKAGE_K = 5.0
MAX_SLOTS = 45
N_RESIDUAL_BINS = 4


# ── 1. Config + load (identical shape to run.py / v2_batter_and_team_features) ─
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


# ── 2. Build PA-grain frame + fit v2's feature set (same as run.py) ────────────
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
    df_feat = df_feat[FEATURE_COLS].copy()
    x_num = num_imp.transform(pd.DataFrame(df_feat[num_cols]).apply(pd.to_numeric, errors="coerce")) if num_cols else np.empty((len(df_feat), 0))
    if cat_cols:
        x_cat_imp = cat_imp.transform(df_feat[cat_cols].astype(object).fillna(np.nan))
        x_cat = enc.transform(x_cat_imp)
    else:
        x_cat = np.empty((len(df_feat), 0))
    x = scaler.transform(np.hstack([x_num, x_cat]))
    return lr.predict_proba(x)[:, 1]


# ── 3. Expected batters faced cascade, ALL seasons (need FIT_SEASONS to fit ────
#      residual bins + VAL_SEASON to apply them) ───────────────────────────────
print("\nBuilding expected_batters_faced cascade...")
pitcher_start_pa_last_season = season_stats.build_pitcher_start_pa_stats(pbp)
league_avg_start_pa = season_stats.build_league_avg_start_pa(pitcher_start_pa_last_season)
team_avg_start_pa = season_stats.build_team_avg_start_pa(pitcher_start_pa_last_season)
pitcher_start_pa_this_season = game_context.build_pitcher_start_pa_this_season(pbp)

expected_pa = game_context.build_expected_batters_faced(
    pitcher_start_pa_last_season, pitcher_start_pa_this_season, team_avg_start_pa, league_avg_start_pa, k=PA_SHRINKAGE_K,
)

sp_pbp = pbp[pbp["pitcher_role"] == "sp"]
realized_pa_all = rolling_stats._pitcher_pa_outcome_per_game(sp_pbp, entity_col="pitcher_id")[
    ["pitcher_id", "gamepk", "game_season", "pa_total"]
].rename(columns={"pitcher_id": "personId", "pa_total": "realized_batters_faced"})

expected_pa = expected_pa.merge(
    realized_pa_all[["personId", "gamepk", "realized_batters_faced"]], on=["personId", "gamepk"], how="left",
)

# ── 4. Fit residual bins on FIT_SEASONS (no leakage into VAL_SEASON) ───────────
fit_starts = expected_pa[
    expected_pa["game_season"].isin(FIT_SEASONS) & expected_pa["realized_batters_faced"].notna()
].copy()
print(f"\nFitting batters-faced residual bins on {len(fit_starts):,} {FIT_SEASONS} SP starts "
      f"({N_RESIDUAL_BINS} bins)...")
residual_bins = game_context.build_batters_faced_residual_bins(fit_starts, n_bins=N_RESIDUAL_BINS)
print(residual_bins.groupby("weight_bin").agg(
    n_residuals=("residual", "size"), lower=("weight_bin_lower", "first"), upper=("weight_bin_upper", "first"),
))

pitcher_starts_2024 = expected_pa[expected_pa["game_season"] == VAL_SEASON].copy()
print(f"\n2024 SP starts with an expected_batters_faced estimate: {len(pitcher_starts_2024):,}")

bf_distribution = game_context.build_batters_faced_distribution(pitcher_starts_2024, residual_bins, max_slots=MAX_SLOTS)
bf_distribution = bf_distribution.set_index(["personId", "gamepk"])["batters_faced_pmf"]

# Game-level features (identical across every slot in a start) — same merges as run.py.
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
pitcher_starts_2024["expected_pitcher_role"] = "sp"


# ── 5. Expand to the FULL max_slots (not the point estimate) so every slot the ─
#      mixture might need is scored, attach batter + opp-team features, score ──
print("\nExpanding to synthetic batter slots (full max_slots cap)...")
batting_order = hp_pipeline._create_batting_order(batter_boxscore)
batter_team_lookup = batter_boxscore[["gamepk", "personId", "team_id"]].rename(
    columns={"personId": "batter_id", "team_id": "batter_team_id"}
).drop_duplicates()

pitcher_starts_for_slots = pitcher_starts_2024.assign(expected_batters_faced=MAX_SLOTS)
slots = game_context.build_batter_slot_expansion(pitcher_starts_for_slots, batting_order, max_slots=MAX_SLOTS)
slots = slots.merge(batter_team_lookup, on=["gamepk", "batter_id"], how="left")

bat_side = pbp[["batter_id", "gamepk", "batter_bat_side"]].drop_duplicates()
slots = slots.merge(bat_side, on=["batter_id", "gamepk"], how="left")
slots = slots.merge(batter_season_stats, on=["batter_id", "game_season"], how="left")
slots = slots.merge(
    batter_pbp_rolling[["batter_id", "gamepk", "batter_roll_season_pa_strikeout_rate"]],
    on=["batter_id", "gamepk"], how="left",
)
slots = slots.merge(opp_team_rolling, on=["batter_team_id", "gamepk"], how="left")

print(f"{len(pitcher_starts_2024):,} starts -> {len(slots):,} synthetic batter-slots "
      f"(mean {len(slots) / max(len(pitcher_starts_2024), 1):.1f} slots/start)")

print("Scoring synthetic slots...")
slots["k_prob"] = score(slots)


# ── 6. Combine via the MIXTURE (uncertain N), not the fixed-N combinator ───────
print("Combining via Poisson-binomial MIXTURE (batters-faced distribution)...")
results = []
for (gamepk, person_id), grp in slots.groupby(["gamepk", "personId"]):
    probs = grp.sort_values("slot")["k_prob"].to_numpy()
    n_pmf = bf_distribution.loc[(person_id, gamepk)]
    pmf = poisson_binomial_mixture_pmf(list(probs), n_pmf)
    predicted_mean_k = float((pmf * np.arange(len(pmf))).sum())
    results.append({
        "gamepk": gamepk, "personId": person_id, "n_slots": len(probs),
        "predicted_mean_k": predicted_mean_k, "pmf": pmf,
    })
pred_df = pd.DataFrame(results)


# ── 7. Compare against realized total K + realized batters faced ──────────────
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

pred_df = pred_df.merge(realized_k, on=["personId", "gamepk"], how="inner")
pred_df = pred_df.merge(
    pitcher_starts_2024[["personId", "gamepk", "expected_batters_faced", "expected_batters_faced_weight", "realized_batters_faced"]],
    on=["personId", "gamepk"], how="left",
)
pred_df["bf_gap"] = (pred_df["realized_batters_faced"] - pred_df["expected_batters_faced"]).abs()
pred_df["abs_error"] = (pred_df["predicted_mean_k"] - pred_df["realized_k"]).abs()

print(f"\n{'=' * 72}\nTOTAL-K COUNT-DISTRIBUTION CHECK (DISTRIBUTION-BASED) — val season {VAL_SEASON}, "
      f"{len(pred_df):,} starts\n{'=' * 72}")
print(f"Predicted mean K:  {pred_df['predicted_mean_k'].mean():.3f}")
print(f"Realized mean K:   {pred_df['realized_k'].mean():.3f}")
print(f"MAE (predicted mean vs. realized): {pred_df['abs_error'].mean():.3f}")

print(f"\n{'bf_gap quartile':<20} {'n':>6} {'MAE':>8}")
pred_df["bf_gap_q"] = pd.qcut(pred_df["bf_gap"], 4, labels=["Q1 (closest)", "Q2", "Q3", "Q4 (furthest)"], duplicates="drop")
for q, grp in pred_df.groupby("bf_gap_q", observed=True):
    print(f"{str(q):<20} {len(grp):>6} {grp['abs_error'].mean():>8.3f}")

LINE = round(pred_df["realized_k"].median()) + 0.5
print(f"\n{'=' * 72}\nTHRESHOLD CHECK — P(total K > {LINE}) vs. realized\n{'=' * 72}")


def p_over_line(pmf, line):
    k_over = int(np.floor(line)) + 1
    return pmf[k_over:].sum() if k_over < len(pmf) else 0.0


pred_df["p_over_line"] = pred_df["pmf"].apply(lambda pmf: p_over_line(pmf, LINE))
pred_df["realized_over_line"] = (pred_df["realized_k"] > LINE).astype(int)

metrics = evaluate_hit_predictor(
    y_true=pred_df["realized_over_line"], y_prob=pred_df["p_over_line"],
    n_bins=8, min_n=30, base_rate=pred_df["realized_over_line"].mean(),
)


# ── 8. Coverage check ───────────────────────────────────────────────────────
print(f"\n{'=' * 72}\nCOVERAGE CHECK (DISTRIBUTION-BASED) — is the predicted total-K distribution\n"
      f"honestly wide?\n{'=' * 72}")


def interval_bounds(pmf, level):
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

pred_df["weight_q"] = pd.qcut(pred_df["expected_batters_faced_weight"], 4, labels=["Q1 (thinnest)", "Q2", "Q3", "Q4 (most reliable)"], duplicates="drop")
print(f"\n80% interval coverage by expected_batters_faced_weight quartile:")
print(f"{'weight quartile':<20} {'n':>6} {'80% coverage':>14}")
for q, grp in pred_df.groupby("weight_q", observed=True):
    print(f"{str(q):<20} {len(grp):>6} {grp['covered_80'].mean():>14.1%}")

print(f"\n{'=' * 72}\nSaved nothing (diagnostic-only check) — record the outcome in ROADMAP.md.\n{'=' * 72}")
