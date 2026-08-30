"""
k_predictor: for the worst-residual 2024 starts surfaced by
run_xgboost_uncertainty.py / run_naive_batter_uncertainty.py, pull the
opposing team's starting lineup and each batter's full-2024-season batting
average + strikeout rate -- checking whether the "unforeseeable dominant
start" misses are partly explained by an unusually strikeout-prone (weak)
opposing lineup, rather than being pure surprise.

Run from src/models/k_predictor/ with:
    python experiments/count_distribution_check/investigate_worst_residuals.py
Requires AWS credentials with read access to s3://mlbdk (us-east-2). Single
season (2024), no pbp needed -- batter_boxscore already has ab/h/k/plate_appearances.
"""
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

import models.hit_predictor.processing.pipeline as hp_pipeline

BUCKET, REGION = "mlbdk", "us-east-2"
SEASON = 2024
boto_session = boto3.Session(region_name=REGION)

GAMES = [
    {"player_name": "Luis Gil", "personId": "661563", "gamepk": "745744"},
    {"player_name": "DJ Herz", "personId": "687792", "gamepk": "744846"},
    {"player_name": "Pablo López", "personId": "641154", "gamepk": "745648"},
    {"player_name": "Spencer Arrighetti", "personId": "681293", "gamepk": "746922"},
    {"player_name": "Ryan Pepiot", "personId": "686752", "gamepk": "746574"},
    {"player_name": "Blake Snell", "personId": "605483", "gamepk": "745307"},
    {"player_name": "Tyler Glasnow", "personId": "607192", "gamepk": "745924"},
]

print("Loading 2024 schedule, batter boxscore, pitcher boxscore...")
schedule = wr.s3.read_parquet(path=f"s3://{BUCKET}/processed_data/games/schedule/{SEASON}/", boto3_session=boto_session)
batter_boxscore = wr.s3.read_parquet(path=f"s3://{BUCKET}/processed_data/prepared/batter_boxscore/{SEASON}/", boto3_session=boto_session)
pitcher_boxscore = wr.s3.read_parquet(path=f"s3://{BUCKET}/processed_data/prepared/pitcher_boxscore/{SEASON}/", boto3_session=boto_session)

schedule = hp_pipeline.process_schedule(schedule)
pitcher_boxscore = hp_pipeline.process_pitcher_boxscore(pitcher_boxscore)
pitcher_boxscore["gamepk"] = pitcher_boxscore["gamepk"].astype(str)
pitcher_boxscore["personId"] = pitcher_boxscore["personId"].astype(str)
pitcher_boxscore["team_id"] = pitcher_boxscore["team_id"].astype(str)
batter_boxscore["gamepk"] = batter_boxscore["gamepk"].astype(str)
batter_boxscore["personId"] = batter_boxscore["personId"].astype(str)
batter_boxscore["team_id"] = batter_boxscore["team_id"].astype(str)

batting_order = hp_pipeline._create_batting_order(batter_boxscore)

# League-wide 2024 context: PA-weighted AVG/OBP/SLG/K rate. OBP is approximated as
# (H+BB)/(AB+BB) -- batter_boxscore has no HBP/SF columns, so this omits both;
# a real (if usually small) undercount for high-HBP hitters specifically.
league_ab, league_h, league_bb, league_k, league_pa, league_tb = (
    batter_boxscore["ab"].sum(), batter_boxscore["h"].sum(), batter_boxscore["bb"].sum(),
    batter_boxscore["k"].sum(), batter_boxscore["plate_appearances"].sum(), batter_boxscore["total_bases"].sum(),
)
league_avg = league_h / league_ab
league_obp = (league_h + league_bb) / (league_ab + league_bb)
league_slg = league_tb / league_ab
league_ops = league_obp + league_slg
league_k_rate = league_k / league_pa
print(f"\n2024 league context: AVG={league_avg:.3f}  OBP≈{league_obp:.3f}  SLG={league_slg:.3f}  "
      f"OPS≈{league_ops:.3f}  K rate={league_k_rate:.3f}\n")
print("(OBP/OPS are approximated as (H+BB)/(AB+BB) -- no HBP/SF in this table, so real OBP runs a bit higher)\n")

# Full-2024-season per-batter aggregates (descriptive only, not point-in-time-safe --
# fine for a one-off investigative check, not a model feature).
season_batter = (
    batter_boxscore.groupby("personId", as_index=False)
    .agg(ab=("ab", "sum"), h=("h", "sum"), bb=("bb", "sum"), k=("k", "sum"),
         pa=("plate_appearances", "sum"), tb=("total_bases", "sum"),
         player_name=("player_name", "first"))
)
season_batter["season_avg"] = season_batter["h"] / season_batter["ab"].replace(0, pd.NA)
season_batter["season_obp"] = (season_batter["h"] + season_batter["bb"]) / (season_batter["ab"] + season_batter["bb"]).replace(0, pd.NA)
season_batter["season_slg"] = season_batter["tb"] / season_batter["ab"].replace(0, pd.NA)
season_batter["season_ops"] = season_batter["season_obp"] + season_batter["season_slg"]
season_batter["season_k_rate"] = season_batter["k"] / season_batter["pa"].replace(0, pd.NA)

schedule_lookup = schedule.set_index("gamepk")[["home_id", "away_id", "home_name", "away_name", "game_date"]]

for g in GAMES:
    gamepk, person_id = g["gamepk"], g["personId"]
    print("=" * 90)
    row = pitcher_boxscore[(pitcher_boxscore["personId"] == person_id) & (pitcher_boxscore["gamepk"] == gamepk)]
    if row.empty:
        print(f"{g['player_name']} — gamepk {gamepk}: NOT FOUND in pitcher_boxscore, skipping")
        continue
    pitcher_team_id = str(row.iloc[0]["team_id"])
    sched_row = schedule_lookup.loc[gamepk]
    game_date = sched_row["game_date"]
    opp_team_id = sched_row["away_id"] if pitcher_team_id == sched_row["home_id"] else sched_row["home_id"]
    opp_team_name = sched_row["away_name"] if pitcher_team_id == sched_row["home_id"] else sched_row["home_name"]
    own_team_name = sched_row["home_name"] if pitcher_team_id == sched_row["home_id"] else sched_row["away_name"]

    print(f"{g['player_name']} ({own_team_name}) vs. {opp_team_name} — {game_date.date()} (gamepk {gamepk})")

    game_batters = batter_boxscore[batter_boxscore["gamepk"] == gamepk][
        ["personId", "team_id"]
    ].drop_duplicates().rename(columns={"personId": "batter_id"})

    lineup = batting_order[batting_order["gamepk"] == gamepk].merge(game_batters, on="batter_id", how="left")
    lineup = lineup[lineup["team_id"] == opp_team_id].sort_values("batting_order")
    lineup = lineup.merge(season_batter, left_on="batter_id", right_on="personId", how="left")

    if lineup.empty:
        print("  (no starting lineup found for opposing team in this game)")
        continue

    print(f"  {'Slot':<5}{'Batter':<24}{'AVG':>7}{'OBP≈':>7}{'SLG':>7}{'OPS≈':>7}{'K rate':>9}")
    for _, r in lineup.iterrows():
        def fmt(col):
            return f"{r[col]:.3f}" if pd.notnull(r[col]) else "n/a"
        print(f"  {int(r['batting_order']):<5}{str(r['player_name']):<24}"
              f"{fmt('season_avg'):>7}{fmt('season_obp'):>7}{fmt('season_slg'):>7}{fmt('season_ops'):>7}{fmt('season_k_rate'):>9}")

    lineup_valid = lineup.dropna(subset=["season_ops", "season_k_rate"])
    if len(lineup_valid):
        best_by_avg = lineup_valid.loc[lineup_valid["season_avg"].idxmax()]
        best_by_ops = lineup_valid.loc[lineup_valid["season_ops"].idxmax()]
        print(f"\n  Lineup avg K rate: {lineup_valid['season_k_rate'].mean():.3f}  (league: {league_k_rate:.3f})  |  "
              f"Lineup avg OPS≈: {lineup_valid['season_ops'].mean():.3f}  (league: {league_ops:.3f})")
        print(f"  Best hitter by AVG:  {best_by_avg['player_name']} "
              f"(AVG {best_by_avg['season_avg']:.3f}, OPS≈ {best_by_avg['season_ops']:.3f}, K rate {best_by_avg['season_k_rate']:.3f})")
        if best_by_avg["player_name"] == best_by_ops["player_name"]:
            print(f"  Best hitter by OPS≈: same player")
        else:
            print(f"  Best hitter by OPS≈: {best_by_ops['player_name']} "
                  f"(AVG {best_by_ops['season_avg']:.3f}, OPS≈ {best_by_ops['season_ops']:.3f}, K rate {best_by_ops['season_k_rate']:.3f})"
                  f"  <-- DIFFERENT from the AVG pick")
    print()
