"""
Statistical shrinkage baseline — cascading empirical-Bayes estimator.
Run from models/hit_predictor/ with: python -m baseline.statistical.run_baseline
Requires AWS credentials with read access to s3://mlbdk (us-east-2).

Predicts P(hit) per PA as this-season BA-so-far (point-in-time safe, no
lookahead) shrunk toward last-season BA, falling back to that season's
league-average hit rate for batters with no prior-season data (rookies).
See shrinkage.py for the estimator itself.

Distinct from baseline/rules/run_baseline.py's fixed-weight 3-component
blend, which was only ever evaluated at PA grain. This script evaluates at
BOTH PA grain and player-game grain (via utils/eval.py::
run_pa_vs_game_grain_check), so results are directly comparable to
BENCHMARKS.md's naive-vs-model game-grain table.
"""
import yaml
from datetime import datetime
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd

import models.hit_predictor.processing.pipeline as pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.baseline.statistical.shrinkage import add_shrinkage_component
from models.hit_predictor.utils.eval import (
    run_pa_vs_game_grain_check,
    summarize_verdict,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── 1. Config ────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET          = cfg["bucket"]
REGION          = cfg["region"]
TRAIN_SEASONS   = cfg["train_seasons"]
FEATURE_SEASONS = cfg["feature_seasons"]
TARGET          = cfg["target_column"]
TEST_SEASON     = cfg["test_season"]
VAL_SEASON      = cfg["val_season"]
MODEL_NAME      = cfg["model_name"]

FIT_SEASONS = [s for s in TRAIN_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]

SHRINKAGE_K = 100.0

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))


# ── 2. Load data from S3 ─────────────────────────────────────────────────────
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

print("\nLoading player info...")
player_info = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session,
)

game_info = pipeline.process_game_info(game_info)
schedule = pipeline.process_schedule(schedule)
pbp = pipeline.build_pbp_features(pbp, schedule, player_info)

pa_outcome = pipeline.create_pa_outcome(pbp, batter_boxscore, game_info, schedule)

# ── 3. Attach last-season BA + compute the shrinkage estimator ───────────────
batter_season_stats = season_stats.build_batter_stats(batter_boxscore).rename(
    columns={'personId': 'batter_id'}
)[['batter_id', 'game_season', 'batter_last_season_ba']].rename(
    columns={'batter_last_season_ba': 'last_season_ba'}
)
pa_outcome = pa_outcome.merge(batter_season_stats, on=['batter_id', 'game_season'], how='left')

pa_outcome = add_shrinkage_component(pa_outcome, k=SHRINKAGE_K)

# ── 4. Split, evaluate at PA grain and game grain ─────────────────────────────
model_df = pa_outcome[
    ['batter_id', 'gamepk', 'play_id', 'game_season', TARGET, 'shrinkage_pred']
].dropna(subset=['shrinkage_pred']).copy()

train_df = model_df[model_df["game_season"].isin(FIT_SEASONS)].copy()
val_df   = model_df[model_df["game_season"] == VAL_SEASON].copy()

print(f"\nFit seasons:  {FIT_SEASONS}")
print(f"Val season:   {VAL_SEASON}  <- iterate against this")
print(f"Hit rate — train: {train_df[TARGET].mean():.3f}  val: {val_df[TARGET].mean():.3f}")

y_val = val_df[TARGET]
proba = np.clip(val_df['shrinkage_pred'].to_numpy(), 1e-6, 1 - 1e-6)

print("\n" + "#" * 75)
print("# STATISTICAL SHRINKAGE BASELINE — PA GRAIN vs GAME GRAIN")
print("#" * 75)
pa_metrics, game_metrics, game_results = run_pa_vs_game_grain_check(
    val_df, y_val, proba, group_cols=("batter_id", "gamepk"),
)

# ── 4b. Naive comparison + verdict ────────────────────────────────────────────
# Decision metrics are reliability+resolution at game grain, not log_loss/Brier
# eyeballed against BENCHMARKS.md's numbers — see BENCHMARKS.md §2. Naive is
# run through the identical harness here (not pulled from docs) so the
# comparison is always live, not stale.
naive_proba = np.full(len(y_val), np.clip(train_df[TARGET].mean(), 1e-6, 1 - 1e-6))
print("\n" + "#" * 75)
print("# NAIVE BASELINE — PA GRAIN vs GAME GRAIN (comparison point)")
print("#" * 75)
naive_pa_metrics, naive_game_metrics, _ = run_pa_vs_game_grain_check(
    val_df, y_val, naive_proba, group_cols=("batter_id", "gamepk"),
)

verdict = summarize_verdict(naive_game_metrics, game_metrics)
print("\n" + "#" * 75)
print("# VERDICT vs NAIVE (game grain)")
print("#" * 75)
print(f"  {verdict['verdict']}")
print(f"  trustworthy (reliability flat-or-better): {verdict['trustworthy']}"
      f"  (delta {verdict['reliability_delta']:+.4f})")
print(f"  differentiated (resolution up):           {verdict['differentiated']}"
      f"  (delta {verdict['resolution_delta']:+.4f})")

# ── 5. Write baseline_results.md ─────────────────────────────────────────────
md_lines = [
    f"# Statistical Shrinkage Baseline Results — {MODEL_NAME}",
    "",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
    f"**Estimator:** this-season BA-so-far shrunk toward last-season BA "
    f"(k={SHRINKAGE_K:.0f}), falling back to season league-average hit rate "
    f"for batters with no prior-season data.  ",
    f"**Data:** s3://{BUCKET}  ",
    "",
    "## Split",
    "",
    "| Split | Seasons | Rows | Hit rate |",
    "|-------|---------|------|----------|",
    f"| Train | {FIT_SEASONS} | {len(train_df):,} | {train_df[TARGET].mean():.3f} |",
    f"| Val   | {VAL_SEASON} | {len(val_df):,} | {y_val.mean():.3f} |",
    "",
    "## Results (val season, evaluated at both grains)",
    "",
    "| Metric | PA grain | Game grain |",
    "|--------|----------|------------|",
]
for key, label in [
    ("log_loss", "LogLoss"), ("brier", "Brier"), ("roc_auc", "ROC-AUC"),
    ("ece", "ECE"), ("reliability", "Reliability"), ("resolution", "Resolution"),
]:
    md_lines.append(f"| {label} | {pa_metrics[key]:.4f} | {game_metrics[key]:.4f} |")

md_lines += [
    "",
    "## Verdict vs naive (game grain)",
    "",
    "Per BENCHMARKS.md §2, the decision metrics are reliability + resolution "
    "together, not log_loss/Brier eyeballed against the docs — computed here "
    "via `utils/eval.py::summarize_verdict()` against a naive baseline run "
    "through the identical harness in this same script run (not a cached "
    "number).",
    "",
    "| | Naive | Shrinkage baseline | Delta |",
    "|---|---|---|---|",
    f"| Reliability | {naive_game_metrics['reliability']:.4f} | {game_metrics['reliability']:.4f} | {verdict['reliability_delta']:+.4f} |",
    f"| Resolution | {naive_game_metrics['resolution']:.4f} | {game_metrics['resolution']:.4f} | {verdict['resolution_delta']:+.4f} |",
    "",
    f"**Verdict: `{verdict['verdict']}`** — trustworthy={verdict['trustworthy']}, "
    f"differentiated={verdict['differentiated']}.",
    "",
    "## Setup",
    "",
    f"- shrinkage k = {SHRINKAGE_K:.0f}",
    "- last_season_ba: from season_stats.build_batter_stats (2016->2017, "
    "2019->2022 covid gap bridge, then year-over-year)",
]

with open(BASE_DIR / "baseline" / "statistical" / "baseline_results.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"\nSaved {BASE_DIR / 'baseline' / 'statistical' / 'baseline_results.md'}")
print("\nDone.")
