import logging
import duckdb
import pandas as pd
import numpy as np
from data_readers import read_batter_boxscore, read_pitcher_boxscore, read_playbyplay
from utils import mlb_ip_to_true, identify_starters_from_pbp

logger = logging.getLogger(__name__)


def compute_batter_season_summary(year: str) -> pd.DataFrame:
    batter = read_batter_boxscore(year)

    con = duckdb.connect()
    con.register("batter_boxscore", batter)

    result = con.execute("""
        SELECT
            personId,
            player_name,
            game_season,
            COUNT(*)                                                            AS season_games,

            -- counting totals
            SUM(plate_appearances)                                              AS season_pa,
            SUM(ab)                                                             AS season_ab,
            SUM(h)                                                              AS season_h,
            SUM(singles)                                                        AS season_singles,
            SUM(doubles)                                                        AS season_doubles,
            SUM(triples)                                                        AS season_triples,
            SUM(hr)                                                             AS season_hr,
            SUM(total_bases)                                                    AS season_tb,
            SUM(rbi)                                                            AS season_rbi,
            SUM(bb)                                                             AS season_bb,
            SUM(k)                                                              AS season_k,
            SUM(sb)                                                             AS season_sb,
            SUM(lob)                                                            AS season_lob,

            -- rate stats
            SUM(h)::FLOAT      / NULLIF(SUM(plate_appearances), 0)             AS hit_rate,
            SUM(k)::FLOAT      / NULLIF(SUM(plate_appearances), 0)             AS k_rate,
            SUM(hr)::FLOAT     / NULLIF(SUM(plate_appearances), 0)             AS hr_rate,
            SUM(bb)::FLOAT     / NULLIF(SUM(plate_appearances), 0)             AS bb_rate,
            SUM(total_bases)::FLOAT / NULLIF(SUM(plate_appearances), 0)        AS tb_rate,
            SUM(rbi)::FLOAT    / NULLIF(SUM(plate_appearances), 0)             AS rbi_rate,

            -- OBP: (H + BB) / (AB + BB) — simplified, HBP not available
            (SUM(h) + SUM(bb))::FLOAT / NULLIF(SUM(ab) + SUM(bb), 0)          AS obp,

            -- SLG: total_bases / AB
            SUM(total_bases)::FLOAT / NULLIF(SUM(ab), 0)                       AS slg,

            -- ISO: SLG - AVG
            (SUM(total_bases)::FLOAT / NULLIF(SUM(ab), 0))
                - (SUM(h)::FLOAT / NULLIF(SUM(ab), 0))                         AS iso,

            -- wOBA (FanGraphs weights, HBP=0 not available)
            (0.69 * SUM(bb) + 0.89 * SUM(singles) + 1.27 * SUM(doubles)
                + 1.62 * SUM(triples) + 2.10 * SUM(hr))::FLOAT
                / NULLIF(SUM(ab) + SUM(bb), 0)                                 AS woba,

            -- Beta distribution prior parameters for Bayesian update in app
            -- hits:  alpha = hits,  beta = PA - hits
            SUM(h)::FLOAT                                                       AS beta_alpha_hits,
            (SUM(plate_appearances) - SUM(h))::FLOAT                           AS beta_beta_hits,

            -- strikeouts: alpha = K, beta = PA - K
            SUM(k)::FLOAT                                                       AS beta_alpha_k,
            (SUM(plate_appearances) - SUM(k))::FLOAT                           AS beta_beta_k,

            -- home runs: alpha = HR, beta = PA - HR
            SUM(hr)::FLOAT                                                      AS beta_alpha_hr,
            (SUM(plate_appearances) - SUM(hr))::FLOAT                          AS beta_beta_hr,

            -- total bases: alpha = TB, beta = PA - TB
            SUM(total_bases)::FLOAT                                             AS beta_alpha_tb,
            (SUM(plate_appearances) - SUM(total_bases))::FLOAT                 AS beta_beta_tb

        FROM batter_boxscore
        WHERE game_type = 'R'
          AND game_season != 2020
        GROUP BY personId, player_name, game_season
        ORDER BY personId, game_season
    """).df()

    logger.info(f"[Batter] output shape: {result.shape}")
    logger.info(f"[Batter] null hit_rate:  {result['hit_rate'].isna().sum()}")
    logger.info(f"[Batter] null k_rate:    {result['k_rate'].isna().sum()}")
    logger.info(f"[Batter] null obp:       {result['obp'].isna().sum()}")
    logger.info(f"[Batter] null slg:       {result['slg'].isna().sum()}")
    logger.info(f"[Batter] null woba:      {result['woba'].isna().sum()}")

    return result


def compute_sp_season_summary(year: str) -> pd.DataFrame:
    pitcher = read_pitcher_boxscore(year)
    pbp     = read_playbyplay(year)

    starters = identify_starters_from_pbp(pbp)

    sp = pitcher.merge(
        starters,
        left_on=["gamepk", "team_id", "personId"],
        right_on=["gamepk", "team_id", "starter_personId"],
        how="inner"
    ).copy()

    sp["ip_true"] = mlb_ip_to_true(sp["ip"])

    logger.info(f"[SP] starters shape: {sp.shape}")
    logger.info(f"[SP] null ip_true: {sp['ip_true'].isna().sum()}")

    con = duckdb.connect()
    con.register("sp_starters", sp)

    result = con.execute("""
        SELECT
            personId,
            starter_name                                                    AS player_name,
            game_season,
            COUNT(*)                                                        AS season_starts,

            -- counting totals
            SUM(ip_true)                                                    AS season_ip,
            SUM(h)::FLOAT                                                   AS season_h_allowed,
            SUM(er)::FLOAT                                                  AS season_er,
            SUM(bb)::FLOAT                                                  AS season_bb_allowed,
            SUM(k)::FLOAT                                                   AS season_k,
            SUM(hr)::FLOAT                                                  AS season_hr_allowed,

            -- per-9 rate stats (industry standard, keep for interpretability)
            SUM(er)::FLOAT  / NULLIF(SUM(ip_true), 0) * 9                  AS era,
            (SUM(h) + SUM(bb))::FLOAT / NULLIF(SUM(ip_true), 0)            AS whip,
            SUM(k)::FLOAT   / NULLIF(SUM(ip_true), 0) * 9                  AS k9,
            SUM(bb)::FLOAT  / NULLIF(SUM(ip_true), 0) * 9                  AS bb9,
            SUM(hr)::FLOAT  / NULLIF(SUM(ip_true), 0) * 9                  AS hr9,

            -- per-inning rate stats (direct multiplier for game projections)
            SUM(k)::FLOAT   / NULLIF(SUM(ip_true), 0)                      AS k_per_ip,
            SUM(bb)::FLOAT  / NULLIF(SUM(ip_true), 0)                      AS bb_per_ip,
            SUM(hr)::FLOAT  / NULLIF(SUM(ip_true), 0)                      AS hr_per_ip,
            SUM(h)::FLOAT   / NULLIF(SUM(ip_true), 0)                      AS h_per_ip,
            SUM(er)::FLOAT  / NULLIF(SUM(ip_true), 0)                      AS er_per_ip,

            -- avg innings per start (expected IP for projection)
            AVG(ip_true)                                                    AS avg_ip_per_start,
            STDDEV(ip_true)                                                 AS std_ip_per_start

        FROM sp_starters
        WHERE game_season != 2020
        GROUP BY personId, starter_name, game_season
        ORDER BY personId, game_season
    """).df()

    logger.info(f"[SP] output shape: {result.shape}")
    logger.info(f"[SP] null era:       {result['era'].isna().sum()}")
    logger.info(f"[SP] null whip:      {result['whip'].isna().sum()}")
    logger.info(f"[SP] null k_per_ip:  {result['k_per_ip'].isna().sum()}")

    return result


def compute_team_season_summary(year: str) -> pd.DataFrame:
    batter  = read_batter_boxscore(year)
    pitcher = read_pitcher_boxscore(year)

    pitcher["ip_true"] = mlb_ip_to_true(pitcher["ip"])

    con = duckdb.connect()
    con.register("batter_boxscore", batter)
    con.register("pitcher_with_ip", pitcher)

    offense = con.execute("""
        SELECT
            team_id,
            game_season,
            COUNT(DISTINCT gamepk)                                          AS season_games,
            SUM(plate_appearances)                                          AS season_pa,
            SUM(ab)                                                         AS season_ab,
            SUM(h)                                                          AS season_h,
            SUM(singles)                                                    AS season_singles,
            SUM(doubles)                                                    AS season_doubles,
            SUM(triples)                                                    AS season_triples,
            SUM(hr)                                                         AS season_hr,
            SUM(total_bases)                                                AS season_tb,
            SUM(r)                                                          AS season_r,
            SUM(rbi)                                                        AS season_rbi,
            SUM(bb)                                                         AS season_bb,
            SUM(k)                                                          AS season_k,
            SUM(sb)                                                         AS season_sb,
            SUM(lob)                                                        AS season_lob,

            -- rate stats
            (SUM(h) + SUM(bb))::FLOAT / NULLIF(SUM(ab) + SUM(bb), 0)      AS obp,
            SUM(total_bases)::FLOAT   / NULLIF(SUM(ab), 0)                 AS slg,
            (SUM(total_bases)::FLOAT  / NULLIF(SUM(ab), 0))
                - (SUM(h)::FLOAT      / NULLIF(SUM(ab), 0))                AS iso,
            (0.69*SUM(bb) + 0.89*SUM(singles) + 1.27*SUM(doubles)
                + 1.62*SUM(triples) + 2.10*SUM(hr))::FLOAT
                / NULLIF(SUM(ab) + SUM(bb), 0)                             AS woba,
            SUM(h)::FLOAT  / NULLIF(SUM(plate_appearances), 0)             AS hit_rate,
            SUM(k)::FLOAT  / NULLIF(SUM(plate_appearances), 0)             AS k_rate,
            SUM(bb)::FLOAT / NULLIF(SUM(plate_appearances), 0)             AS bb_rate,
            SUM(r)::FLOAT  / NULLIF(COUNT(DISTINCT gamepk), 0)             AS r_per_game

        FROM batter_boxscore
        WHERE game_type = 'R' AND game_season != 2020
        GROUP BY team_id, game_season
    """).df()

    pitching = con.execute("""
        SELECT
            team_id,
            game_season,
            SUM(ip_true)                                                    AS season_ip_pitched,
            SUM(h)::FLOAT                                                   AS season_h_allowed,
            SUM(er)::FLOAT                                                  AS season_er_allowed,
            SUM(bb)::FLOAT                                                  AS season_bb_allowed,
            SUM(k)::FLOAT                                                   AS season_k_pitched,
            SUM(hr)::FLOAT                                                  AS season_hr_allowed,

            -- per-9
            SUM(er)::FLOAT  / NULLIF(SUM(ip_true), 0) * 9                  AS era_allowed,
            (SUM(h) + SUM(bb))::FLOAT / NULLIF(SUM(ip_true), 0)            AS whip_allowed,
            SUM(k)::FLOAT   / NULLIF(SUM(ip_true), 0) * 9                  AS k9_pitched,
            SUM(bb)::FLOAT  / NULLIF(SUM(ip_true), 0) * 9                  AS bb9_allowed,
            SUM(hr)::FLOAT  / NULLIF(SUM(ip_true), 0) * 9                  AS hr9_allowed,

            -- per-inning
            SUM(k)::FLOAT   / NULLIF(SUM(ip_true), 0)                      AS k_per_ip_pitched,
            SUM(bb)::FLOAT  / NULLIF(SUM(ip_true), 0)                      AS bb_per_ip_allowed,
            SUM(hr)::FLOAT  / NULLIF(SUM(ip_true), 0)                      AS hr_per_ip_allowed,
            SUM(h)::FLOAT   / NULLIF(SUM(ip_true), 0)                      AS h_per_ip_allowed

        FROM pitcher_with_ip
        WHERE game_type = 'R' AND game_season != 2020
        GROUP BY team_id, game_season
    """).df()

    result = offense.merge(pitching, on=["team_id", "game_season"], how="inner")
    result = result.sort_values(["team_id", "game_season"])

    logger.info(f"[Team] output shape: {result.shape}")
    logger.info(f"[Team] null obp:         {result['obp'].isna().sum()}")
    logger.info(f"[Team] null era_allowed: {result['era_allowed'].isna().sum()}")
    logger.info(f"[Team] null woba:        {result['woba'].isna().sum()}")

    return result

def write_season_summary(df: pd.DataFrame, name: str, year: str) -> None:
    """Write season summary dataframe to S3 as parquet partitioned by game_season."""
    df = df.copy()
    df["game_season"] = df["game_season"].astype(int)  # ensure plain int for partition
    path = f"{S3_BASE}/{name}/game_season={year}"
    fs = s3fs.S3FileSystem()
    if fs.exists(path):
        fs.rm(path, recursive=True)
    table = pa.Table.from_pandas(df)  # keep game_season — pyarrow needs it to partition
    pq.write_to_dataset(
        table,
        root_path=f"{S3_BASE}/{name}",
        partition_cols=["game_season"],
        filesystem=fs,
    )
    logger.info(f"Written {len(df):,} rows to {S3_BASE}/{name}/game_season={year}")

if __name__ == "__main__":
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    import s3fs

    S3_BASE = "s3://mlbdk/processed_data/season_summaries"

    seasons = [str(y) for y in [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]]
    if len(sys.argv) > 1:
        seasons = [sys.argv[1]]  # single season override: python compute_season_summaries.py 2025

    for year in seasons:
        print(f"\n{'='*50}\nProcessing season {year}\n{'='*50}")

        batter = compute_batter_season_summary(year)
        sp     = compute_sp_season_summary(year)
        team   = compute_team_season_summary(year)

        write_season_summary(batter, "batter_season_summary", year)
        write_season_summary(sp,     "sp_season_summary",     year)
        write_season_summary(team,   "team_season_summary",   year)

        print(f"Batter summary:  {batter.shape}")
        print(f"SP summary:      {sp.shape}")
        print(f"Team summary:    {team.shape}")
