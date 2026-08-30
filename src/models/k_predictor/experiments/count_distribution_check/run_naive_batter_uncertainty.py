"""
k_predictor: total-strikeout count-distribution + uncertainty report, scored by
a NAIVE per-slot probability -- the batter's own shrunk strikeout rate, with
zero pitcher signal at all -- instead of v6's tuned XGBoost.

Run from src/models/k_predictor/ with:
    python experiments/count_distribution_check/run_naive_batter_uncertainty.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
# ---------------------------------------------------------------------------- #
#                                    SUMMARY                                   #
# ---------------------------------------------------------------------------- #
# run_xgboost_uncertainty.py found that the predicted 80% interval is honest in
# aggregate (79.9% empirical) but its WIDTH barely varies across starts (corr
# with actual error = 0.07) and its worst misses are all big under-predictions
# on unusually dominant starts. Open question: is that a property of the
# Poisson-binomial-of-independent-slots APPROACH, or specific to what XGBoost's
# 42 features happen to capture? This sibling script re-runs the exact same
# downstream pipeline (batters-faced cascade -> slot expansion -> Poisson-
# binomial combine -> coverage check -> residual-ranked examples) but replaces
# the per-slot score with a dead-simple per-BATTER-only number: their own
# strikeout rate, shrunk from last season toward this season's emerging rate
# as PA accumulate (shrinkage_weight = pa_n / (pa_n + k), same formula shape as
# k_predictor's pitcher_shrunk_whip, applied here to the batter side instead).
# No pitcher, no matchup, no game context at all goes into this score -- it's
# the simplest per-slot number that's still season/form-aware, one step above
# a constant. If it produces the SAME coverage/width-uninformativeness/thin-
# tail pattern as XGBoost, that's evidence the pattern is structural to the
# approach; if it looks meaningfully different, XGBoost's features are doing
# real work on the uncertainty shape, not just the point estimate.
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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import rolling_stats
from models.hit_predictor.processing.features import game_context
from models.hit_predictor.utils.eval import evaluate_hit_predictor

import models.k_predictor.processing.pipeline as pipeline
from models.hit_predictor.utils.count_distribution import poisson_binomial_pmf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PA_SHRINKAGE_K = 5.0
BATTER_K_SHRINKAGE_K = 50.0  # PA-based, not games-based (batters accumulate PA fast) -- a diagnostic choice, not an established repo convention
MAX_SLOTS = 45
N_EXAMPLES_PER_TAIL = 5

OUT_DIR = Path(__file__).parent / "naive_batter_uncertainty"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Config + load (identical to run_xgboost_uncertainty.py) ────────────────
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


# ── 2. Minimal processing — no PA-grain frame, no model, batter stats only ────
print("\nProcessing...")
schedule = hp_pipeline.process_schedule(schedule)
game_info = hp_pipeline.process_game_info(game_info)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pbp = pipeline.build_pbp_features_strikeout(pbp, schedule, player_info)
# build_pitcher_start_pa_this_season (below) needs game_datetime for correct
# same-date doubleheader ordering -- same reason run_xgboost_uncertainty.py needs it.
pbp = pbp.merge(schedule[["gamepk", "game_datetime"]], on="gamepk", how="left")

batter_season_stats = season_stats.build_pbp_batter_feats(pbp)[
    ["batter_id", "game_season", "batter_last_season_pa_strikeout_rate"]
]
batter_pbp_rolling = rolling_stats.build_pbp_batter_rolling_feats(pbp, window="season")[
    ["batter_id", "gamepk", "batter_roll_season_pa_strikeout_rate", "batter_roll_season_pa_total"]
]

league_avg_batter_k_rate = batter_season_stats["batter_last_season_pa_strikeout_rate"].mean()
print(f"League-average batter last-season K rate (fallback only): {league_avg_batter_k_rate:.4f}")


# ── 3. Expected batters faced, val season 2024 SP starts only (unchanged) ─────
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


# ── 4. Expand to synthetic batter slots, attach ONLY batter-side features ─────
print("Expanding to synthetic batter slots...")
batting_order = hp_pipeline._create_batting_order(batter_boxscore)
batter_team_lookup = batter_boxscore[["gamepk", "personId", "team_id"]].rename(
    columns={"personId": "batter_id", "team_id": "batter_team_id"}
).drop_duplicates()

slots = game_context.build_batter_slot_expansion(pitcher_starts_2024, batting_order, max_slots=MAX_SLOTS)
slots = slots.merge(batter_team_lookup, on=["gamepk", "batter_id"], how="left")
slots = slots.merge(batter_season_stats, on=["batter_id", "game_season"], how="left")
slots = slots.merge(batter_pbp_rolling, on=["batter_id", "gamepk"], how="left")

print(f"{len(pitcher_starts_2024):,} starts -> {len(slots):,} synthetic batter-slots "
      f"(mean {len(slots) / max(len(pitcher_starts_2024), 1):.1f} slots/start)")

# Shrunk batter K rate — same formula shape as pitcher_workload.build_pitcher_shrunk_whip:
# baseline = last season (falls back to this-season rolling for a rookie with no
# last-season row), rolling_safe = this-season rolling (falls back to baseline
# early in the season before any PA), weight = pa_n / (pa_n + k) rises from 0 at
# a season opener toward 1 as this season's own sample grows. Final league-average
# fallback only covers the rare case where BOTH are missing (a true debut).
pa_n = slots["batter_roll_season_pa_total"].fillna(0)
weight = pa_n / (pa_n + BATTER_K_SHRINKAGE_K)
baseline = slots["batter_last_season_pa_strikeout_rate"].fillna(slots["batter_roll_season_pa_strikeout_rate"])
rolling_safe = slots["batter_roll_season_pa_strikeout_rate"].fillna(baseline)
slots["shrunk_batter_k_rate"] = ((1 - weight) * baseline + weight * rolling_safe).fillna(league_avg_batter_k_rate)


# ── 5. Score each slot (naive), combine into a total-K distribution per start ─
print("Scoring synthetic slots with the naive shrunk batter K rate...")
slots["k_prob"] = slots["shrunk_batter_k_rate"]

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


# ── 6. Compare against realized total K + realized batters faced (unchanged) ──
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
    pitcher_starts_2024[["personId", "gamepk", "expected_batters_faced", "expected_batters_faced_weight"]],
    on=["personId", "gamepk"], how="left",
)
pred_df["residual"] = pred_df["predicted_mean_k"] - pred_df["realized_k"]
pred_df["abs_error"] = pred_df["residual"].abs()
pred_df["weight_q"] = pd.qcut(pred_df["expected_batters_faced_weight"], 4, labels=["Q1 (thinnest)", "Q2", "Q3", "Q4 (most reliable)"], duplicates="drop")

print(f"\n{'=' * 72}\nTOTAL-K COUNT-DISTRIBUTION CHECK (naive batter-only) — val season {VAL_SEASON}, {len(pred_df):,} starts\n{'=' * 72}")
print(f"Predicted mean K:  {pred_df['predicted_mean_k'].mean():.3f}")
print(f"Realized mean K:   {pred_df['realized_k'].mean():.3f}")
print(f"MAE (predicted mean vs. realized): {pred_df['abs_error'].mean():.3f}")

LINE = round(pred_df["realized_k"].median()) + 0.5


def p_over_line(pmf, line):
    k_over = int(np.floor(line)) + 1
    return pmf[k_over:].sum() if k_over < len(pmf) else 0.0


pred_df["p_over_line"] = pred_df["pmf"].apply(lambda pmf: p_over_line(pmf, LINE))
pred_df["realized_over_line"] = (pred_df["realized_k"] > LINE).astype(int)

print(f"\n{'=' * 72}\nTHRESHOLD CHECK — P(total K > {LINE}) vs. realized\n{'=' * 72}")
threshold_metrics = evaluate_hit_predictor(
    y_true=pred_df["realized_over_line"], y_prob=pred_df["p_over_line"],
    n_bins=8, min_n=30, base_rate=pred_df["realized_over_line"].mean(),
)


# ── 7. Coverage check (unchanged) ──────────────────────────────────────────────
def interval_bounds(pmf, level):
    cdf = np.cumsum(pmf)
    alpha = (1 - level) / 2
    lower = int(np.searchsorted(cdf, alpha, side="left"))
    upper = int(np.searchsorted(cdf, 1 - alpha, side="left"))
    return lower, upper


LEVELS = [0.50, 0.80, 0.95]
print(f"\n{'=' * 72}\nCOVERAGE CHECK\n{'=' * 72}")
print(f"{'level (nominal)':<18} {'n':>6} {'empirical coverage':>20} {'gap (empirical - nominal)':>28}")
for level in LEVELS:
    bounds = pred_df["pmf"].apply(lambda pmf: interval_bounds(pmf, level))
    lower = bounds.apply(lambda t: t[0])
    upper = bounds.apply(lambda t: t[1])
    covered = (pred_df["realized_k"] >= lower) & (pred_df["realized_k"] <= upper)
    pred_df[f"covered_{int(level * 100)}"] = covered
    pred_df[f"lower_{int(level * 100)}"] = lower
    pred_df[f"upper_{int(level * 100)}"] = upper
    emp = covered.mean()
    print(f"{level:<18.0%} {len(pred_df):>6} {emp:>20.1%} {emp - level:>+28.1%}")

pred_df["interval_width_80"] = pred_df["upper_80"] - pred_df["lower_80"]

coverage_by_level = {str(int(l * 100)): round(float(pred_df[f"covered_{int(l*100)}"].mean()), 4) for l in LEVELS}
coverage_by_weight_quartile_80 = {
    str(q): round(float(grp["covered_80"].mean()), 4) for q, grp in pred_df.groupby("weight_q", observed=True)
}


# ── 8. Population-level plots (unchanged shape from run_xgboost_uncertainty.py) ─
print("\nBuilding population-level plots...")

fig, ax = plt.subplots(figsize=(7, 7))
colors = {"Q1 (thinnest)": "#d62728", "Q2": "#ff7f0e", "Q3": "#1f77b4", "Q4 (most reliable)": "#2ca02c"}
for q, grp in pred_df.groupby("weight_q", observed=True):
    ax.scatter(grp["predicted_mean_k"], grp["realized_k"], s=14, alpha=0.5,
               label=str(q), color=colors.get(str(q), "gray"))
lims = [0, max(pred_df["predicted_mean_k"].max(), pred_df["realized_k"].max()) + 1]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Predicted mean K (pmf expectation)")
ax.set_ylabel("Realized K")
ax.set_title(f"Predicted vs. realized strikeouts — every {VAL_SEASON} SP start (naive batter-only)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "predicted_vs_realized_scatter.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(pred_df["interval_width_80"], pred_df["abs_error"], s=14, alpha=0.4, color="#1f77b4")
corr_width_error = pred_df[["interval_width_80", "abs_error"]].corr().iloc[0, 1]
ax.set_xlabel("Predicted 80% interval width (K)")
ax.set_ylabel("|residual| = |predicted mean K - realized K|")
ax.set_title(f"Does a wider predicted distribution track a bigger miss?\ncorr = {corr_width_error:.3f}")
plt.tight_layout()
plt.savefig(PLOT_DIR / "residual_vs_interval_width.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(pred_df["residual"], bins=30, color="#1f77b4", edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Residual (predicted mean K - realized K)")
ax.set_ylabel("Starts")
ax.set_title(f"Residual distribution — {VAL_SEASON} SP starts (naive batter-only)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "residual_histogram.png", dpi=130)
plt.close()

print(f"corr(interval_width_80, abs_error) = {corr_width_error:.3f}")


# ── 9. Down to examples — ranked by residual, both tails (unchanged) ──────────
print("\nSelecting worked examples by residual (best + worst)...")
name_lookup = pitcher_boxscore[["personId", "player_name"]].drop_duplicates("personId")
examples_df = pred_df.merge(name_lookup, on="personId", how="left")
examples_df = examples_df[examples_df["n_slots"] >= 10].copy()

best = examples_df.nsmallest(N_EXAMPLES_PER_TAIL, "abs_error")
worst = examples_df.nlargest(N_EXAMPLES_PER_TAIL, "abs_error")


def render_example(row, tag):
    pmf = np.array(row["pmf"])
    ks = np.arange(len(pmf))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ks, pmf, color="#1f77b4", alpha=0.8)
    ax.axvline(row["realized_k"], color="#d62728", linewidth=2, label=f"Realized K = {int(row['realized_k'])}")
    ax.axvline(row["predicted_mean_k"], color="black", linestyle="--", linewidth=1,
               label=f"Predicted mean = {row['predicted_mean_k']:.2f}")
    ax.set_xlabel("Total strikeouts (K)")
    ax.set_ylabel("Predicted probability")
    ax.set_title(f"{row['player_name']} — {row['gamepk']} (naive batter-only)\n"
                 f"|residual|={row['abs_error']:.2f}, 80% interval=[{int(row['lower_80'])},{int(row['upper_80'])}], "
                 f"weight={row['expected_batters_faced_weight']:.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"example_{tag}_{row['personId']}_{row['gamepk']}.png"
    plt.savefig(PLOT_DIR / fname, dpi=130)
    plt.close()
    return fname


def example_record(row, tag, fname):
    return {
        "tag": tag, "plot": fname, "player_name": row["player_name"],
        "gamepk": str(row["gamepk"]), "personId": str(row["personId"]),
        "n_slots": int(row["n_slots"]), "predicted_mean_k": round(float(row["predicted_mean_k"]), 2),
        "realized_k": int(row["realized_k"]), "abs_error": round(float(row["abs_error"]), 3),
        "lower_80": int(row["lower_80"]), "upper_80": int(row["upper_80"]),
        "covered_80": bool(row["covered_80"]),
        "expected_batters_faced_weight": round(float(row["expected_batters_faced_weight"]), 3),
    }


example_records = []
for i, row in enumerate(best.itertuples(index=False), start=1):
    row = dict(zip(best.columns, row))
    fname = render_example(row, f"best{i}")
    example_records.append(example_record(row, "best", fname))
for i, row in enumerate(worst.itertuples(index=False), start=1):
    row = dict(zip(worst.columns, row))
    fname = render_example(row, f"worst{i}")
    example_records.append(example_record(row, "worst", fname))


# ── 10. Persist everything the report needs ────────────────────────────────────
summary = {
    "val_season": VAL_SEASON,
    "n_starts": int(len(pred_df)),
    "batter_k_shrinkage_k": BATTER_K_SHRINKAGE_K,
    "predicted_mean_k": round(float(pred_df["predicted_mean_k"].mean()), 3),
    "realized_mean_k": round(float(pred_df["realized_k"].mean()), 3),
    "mae": round(float(pred_df["abs_error"].mean()), 3),
    "coverage_by_level": coverage_by_level,
    "coverage_by_weight_quartile_80": coverage_by_weight_quartile_80,
    "corr_interval_width_vs_abs_error": round(float(corr_width_error), 4),
    "threshold_check": {
        "line": LINE,
        "reliability": round(float(threshold_metrics["reliability"]), 4),
        "resolution": round(float(threshold_metrics["resolution"]), 4),
        "roc_auc": round(float(threshold_metrics["roc_auc"]), 4),
        "pr_auc": round(float(threshold_metrics["pr_auc"]), 4),
    },
    "examples": example_records,
}
out_path = OUT_DIR / "uncertainty_summary.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
pred_df.drop(columns=["pmf"]).to_parquet(OUT_DIR / "pred_df.parquet", index=False)
print(f"\nSummary written to {out_path}")
print(f"Plots written to {PLOT_DIR}")
