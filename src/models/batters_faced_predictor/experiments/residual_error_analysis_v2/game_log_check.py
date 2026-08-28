"""
Diagnostic (not a versioned experiment, same convention as run.py in this
folder): follow-up on error_analysis_v2.md's worked-example finding that the
dominant remaining over-prediction failure mode is established, high-weight
starters getting pulled after 0-2 clean innings with no decision recorded —
the same mechanism v2 targeted (trend/rest/workload_density) but didn't fully
close. Two names (Ranger Suarez, Frankie Montas) and one more (Yariel
Rodriguez) showed up in BOTH the worst-over AND worst-under prediction lists,
suggesting start-to-start VOLATILITY, not just level or trend, may be the
missing signal for some pitchers.

This script pulls full-2024 game logs for a widened set of flagged pitchers
(the both-lists names, plus other repeat/notable names from error_analysis_v2:
Skenes, Crochet, Yamamoto, Kershaw, J. Ryan, J. Steele, Gausman) and a random
control group of similarly-established starters (11+ starts, val season) NOT
flagged, then compares each pitcher's season-long coefficient of variation
(std/mean) of realized_batters_faced per start. If flagged pitchers show
materially higher CV than the control group, that's real evidence for a
"trailing-N start volatility" feature candidate (distinct from the existing
level/trend features) for a v4.

Run from src/models/batters_faced_predictor/ with:
  python experiments/residual_error_analysis_v2/game_log_check.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2).
"""
import random
import yaml
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd

import models.hit_predictor.processing.pipeline as hp_pipeline
from models.hit_predictor.processing.features import season_stats
from models.hit_predictor.processing.features import game_context

import models.batters_faced_predictor.processing.pipeline as pipeline

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

STAGE = Path(__file__).parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PA_SHRINKAGE_K = 5.0

with open(BASE_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET, REGION = cfg["bucket"], cfg["region"]
TRAIN_SEASONS, FEATURE_SEASONS = cfg["train_seasons"], cfg["feature_seasons"]
VAL_SEASON = cfg["val_season"]

boto_session = boto3.Session(region_name=REGION)
all_boxscore_seasons = sorted(set(FEATURE_SEASONS + TRAIN_SEASONS))

FLAGGED = {
    624133: "Ranger Suarez",       # appears in BOTH under- and over-prediction lists
    593423: "Frankie Montas",      # appears in BOTH under- and over-prediction lists
    684320: "Yariel Rodriguez",    # appears in BOTH under- and over-prediction lists
    694973: "Paul Skenes",         # repeat over-prediction, known 2024 innings-management storyline
    676979: "Garrett Crochet",     # repeat over-prediction, known 2024 innings-management storyline
    808967: "Yoshinobu Yamamoto",  # repeat over-prediction, 2024 shoulder-caution storyline
    477132: "Clayton Kershaw",     # repeat over-prediction, 2024 return-from-injury workload management
    657746: "Joe Ryan",            # repeat over-prediction
    657006: "Justin Steele",       # repeat over-prediction
    592332: "Kevin Gausman",       # repeat UNDER-prediction (opposite direction — goes long repeatedly)
}


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


print("Loading play-by-play...")
pbp = read_parquet_seasons("s3://{bucket}/processed_data/prepared/playbyplay/{season}/", TRAIN_SEASONS, chunked=True)
print("Loading schedule...")
schedule = read_parquet_seasons("s3://{bucket}/processed_data/games/schedule/{season}/", all_boxscore_seasons)
print("Loading pitcher boxscore (val season)...")
pitcher_boxscore_val = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/processed_data/prepared/pitcher_boxscore/{VAL_SEASON}/", boto3_session=boto_session,
)
print("Loading player info...")
player_info = wr.s3.read_parquet(path=f"s3://{BUCKET}/raw_data/reference/player_info/", boto3_session=boto_session)

print("\nBuilding start-grain DataFrame...")
schedule = hp_pipeline.process_schedule(schedule)
pbp = hp_pipeline.build_pbp_features(pbp, schedule, player_info)
pbp = pbp.merge(schedule[["gamepk", "game_datetime"]], on="gamepk", how="left")

start_outcome = pipeline.create_start_pa_outcome(pbp)

pitcher_start_pa_last_season = season_stats.build_pitcher_start_pa_stats(pbp)
league_avg_start_pa = season_stats.build_league_avg_start_pa(pitcher_start_pa_last_season)
team_avg_start_pa = season_stats.build_team_avg_start_pa(pitcher_start_pa_last_season)
pitcher_start_pa_this_season = game_context.build_pitcher_start_pa_this_season(pbp)

expected_pa = game_context.build_expected_batters_faced(
    pitcher_start_pa_last_season, pitcher_start_pa_this_season, team_avg_start_pa, league_avg_start_pa,
    k=PA_SHRINKAGE_K,
)
expected_pa["personId"] = expected_pa["personId"].astype(str)
expected_pa["gamepk"] = expected_pa["gamepk"].astype(str)
start_outcome = start_outcome.drop(columns=["game_date", "game_season"]).merge(
    expected_pa, on=["personId", "gamepk"], how="left",
)

# game_season was dropped by the merge above (expected_pa carries none) — recompute from schedule
schedule_season = schedule[["gamepk", "game_date", "game_datetime"]].drop_duplicates("gamepk").assign(
    gamepk=lambda x: x["gamepk"].astype(str),
)
schedule_season["game_season"] = pd.to_datetime(schedule_season["game_datetime"]).dt.year
start_outcome["gamepk"] = start_outcome["gamepk"].astype(str)
start_outcome = start_outcome.drop(columns=["game_date", "game_season"], errors="ignore").merge(
    schedule_season[["gamepk", "game_date", "game_season"]], on="gamepk", how="left",
)

val = start_outcome[start_outcome["game_season"] == VAL_SEASON].copy()
val["personId"] = val["personId"].astype(str)

pitcher_boxscore_val = pitcher_boxscore_val.assign(
    personId=lambda x: x["personId"].astype(str), gamepk=lambda x: x["gamepk"].astype(str),
)[["personId", "gamepk", "ip", "h", "r", "er", "bb", "k", "p", "outcome"]]

home_away = schedule[["gamepk", "home_id", "away_id", "home_name", "away_name"]].drop_duplicates("gamepk").assign(
    gamepk=lambda x: x["gamepk"].astype(str), home_id=lambda x: x["home_id"].astype(str), away_id=lambda x: x["away_id"].astype(str),
)

# pitcher_team_id already present on val, carried through from
# build_expected_batters_faced's own team_avg_start_pa merge — no re-merge needed.
val["pitcher_team_id"] = val["pitcher_team_id"].astype(str)
val = val.merge(home_away, on="gamepk", how="left")
val["opp_team_name"] = np.where(val["pitcher_team_id"] == val["home_id"], val["away_name"], val["home_name"])
val = val.merge(pitcher_boxscore_val, on=["personId", "gamepk"], how="left")

LOG_COLS = [
    "game_date", "opp_team_name", "realized_batters_faced", "expected_batters_faced",
    "expected_batters_faced_weight", "ip", "h", "r", "er", "bb", "k", "p", "outcome",
]

print(f"\n{'=' * 100}\nFULL 2024 GAME LOGS — FLAGGED PITCHERS\n{'=' * 100}")
flagged_stats = []
for pid, name in FLAGGED.items():
    log = val[val["personId"] == str(pid)].sort_values("game_date")
    if log.empty:
        print(f"\n--- {name} ({pid}): no {VAL_SEASON} starts found ---")
        continue
    bf = log["realized_batters_faced"]
    mean_bf, std_bf = bf.mean(), bf.std()
    cv = std_bf / mean_bf if mean_bf else np.nan
    print(f"\n--- {name} ({pid}) — {len(log)} starts, mean BF {mean_bf:.1f}, std {std_bf:.2f}, CV {cv:.3f} ---")
    print(log[LOG_COLS].round(2).to_string(index=False))
    flagged_stats.append({"personId": pid, "name": name, "n_starts": len(log), "mean_bf": mean_bf, "std_bf": std_bf, "cv": cv})

# ── Control group: established starters (>=15 starts in 2024) NOT flagged ──
starts_per_pitcher = val.groupby("personId").size()
established_ids = starts_per_pitcher[starts_per_pitcher >= 15].index.tolist()
established_ids = [pid for pid in established_ids if pid not in {str(k) for k in FLAGGED}]
random.seed(42)
control_ids = random.sample(established_ids, min(15, len(established_ids)))

print(f"\n{'=' * 100}\nCONTROL GROUP — {len(control_ids)} RANDOM ESTABLISHED STARTERS (15+ starts, NOT flagged)\n{'=' * 100}")
control_stats = []
for pid in control_ids:
    log = val[val["personId"] == pid].sort_values("game_date")
    bf = log["realized_batters_faced"]
    mean_bf, std_bf = bf.mean(), bf.std()
    cv = std_bf / mean_bf if mean_bf else np.nan
    control_stats.append({"personId": pid, "n_starts": len(log), "mean_bf": mean_bf, "std_bf": std_bf, "cv": cv})

flagged_df = pd.DataFrame(flagged_stats)
control_df = pd.DataFrame(control_stats)

print("\nFlagged group:")
print(flagged_df.round(3).to_string(index=False))
print("\nControl group:")
print(control_df.round(3).to_string(index=False))

print(f"\n{'=' * 100}\nCOMPARISON — coefficient of variation (std/mean) of realized_batters_faced per start\n{'=' * 100}")
print(f"Flagged group  (n={len(flagged_df)}): mean CV {flagged_df['cv'].mean():.3f}, mean std {flagged_df['std_bf'].mean():.3f}")
print(f"Control group  (n={len(control_df)}): mean CV {control_df['cv'].mean():.3f}, mean std {control_df['std_bf'].mean():.3f}")

md_lines = [
    "# Game-log check — volatility hypothesis follow-up on error_analysis_v2.md",
    "",
    "Full 2024 game logs for pitchers flagged from error_analysis_v2.md's worked",
    "examples (appearing in both over- and under-prediction lists, or repeat",
    "over-predictions with a known real-world workload-management storyline),",
    "vs. a random control group of established (15+ start) 2024 starters not flagged.",
    "",
    "## Flagged group",
    "",
    flagged_df.round(3).to_markdown(index=False),
    "",
    "## Control group",
    "",
    control_df.round(3).to_markdown(index=False),
    "",
    "## Comparison",
    "",
    f"Flagged group  (n={len(flagged_df)}): mean CV {flagged_df['cv'].mean():.3f}, mean std {flagged_df['std_bf'].mean():.3f}",
    f"Control group  (n={len(control_df)}): mean CV {control_df['cv'].mean():.3f}, mean std {control_df['std_bf'].mean():.3f}",
    "",
]
for pid, name in FLAGGED.items():
    log = val[val["personId"] == str(pid)].sort_values("game_date")
    if log.empty:
        continue
    md_lines += [f"### {name} ({pid})", "", log[LOG_COLS].round(2).to_markdown(index=False), ""]

(STAGE / "game_log_check.md").write_text("\n".join(md_lines) + "\n")
print(f"\nSaved {STAGE / 'game_log_check.md'}")
