import duckdb
from data_readers import read_batter_boxscore, read_pitcher_boxscore
import pandas as pd

pd.set_option('display.max_columns', None)

def compute_team_batter_base_features(year:str):
    batter_boxscore = read_batter_boxscore(year)

    con = duckdb.connect()
    con.register("batter_boxscore", batter_boxscore)
    team_batter_stats = con.execute("""
        SELECT
            team_id,
            gamepk,
            game_date,
            game_season,
    
            -- counting
            SUM(plate_appearances)::FLOAT   AS total_plate_appearances,
            SUM(ab)::FLOAT                  AS total_ab,
            SUM(h)::FLOAT                   AS total_h,
            SUM(r)::FLOAT                   AS total_r,
    
            -- hit breakdown
            SUM(singles)::FLOAT             AS total_singles,
            SUM(doubles)::FLOAT             AS total_doubles,
            SUM(triples)::FLOAT             AS total_triples,
            SUM(hr)::FLOAT                  AS total_hr,
    
            -- bases & production
            SUM(total_bases)::FLOAT         AS total_bases,
            SUM(total_bases_from_h)::FLOAT  AS total_bases_from_h,
            SUM(rbi)::FLOAT                 AS total_rbi,
            SUM(lob)::FLOAT                 AS total_lob,
            SUM(sb)::FLOAT                  AS total_sb,
    
            -- discipline (batter side)
            SUM(bb)::FLOAT                  AS total_bb_bat,
            SUM(k)::FLOAT                   AS total_k_bat,
    
            -- fantasy
            SUM(fantasy_points)::FLOAT      AS total_batter_fantasy_points
    
        FROM batter_boxscore
        GROUP BY
            team_id,
            gamepk,
            game_date,
            game_season
        ORDER BY
            team_id,
            game_date
    """).df()
    
    team_base_features =  con.execute("""
                                            
        SELECT
            team_id,
            gamepk,
            game_date,
            game_season,
            total_plate_appearances,
            total_ab,
            total_h,
            total_r,
            total_singles,
            total_doubles,
            total_triples,
            total_hr,
            total_bases,
            total_bases_from_h,
            total_rbi,
            total_lob,
            total_sb,
            total_bb_bat,
            total_k_bat,
            total_batter_fantasy_points,
                                            
            --── plate appearances ──────────────────────────────────────────
            SUM(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_plate_appearances_sum,
            AVG(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_plate_appearances_avg,
            STDDEV(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_plate_appearances_std,
            MEDIAN(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_plate_appearances_median,
    
            SUM(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_plate_appearances_sum,
            AVG(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_plate_appearances_avg,
            STDDEV(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_plate_appearances_std,
            MEDIAN(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_plate_appearances_median,
    
            SUM(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_plate_appearances_sum,
            AVG(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_plate_appearances_avg,
            STDDEV(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_plate_appearances_std,
            MEDIAN(total_plate_appearances)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_plate_appearances_median,
    
            -- ── at bats ────────────────────────────────────────────────────
            SUM(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_ab_sum,
            AVG(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_ab_avg,
            STDDEV(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_ab_std,
            MEDIAN(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_ab_median,
    
            SUM(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_ab_sum,
            AVG(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_ab_avg,
            STDDEV(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_ab_std,
            MEDIAN(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_ab_median,
    
            SUM(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_ab_sum,
            AVG(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_ab_avg,
            STDDEV(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_ab_std,
            MEDIAN(total_ab)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_ab_median,
    
            -- ── hits ───────────────────────────────────────────────────────
            SUM(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_h_sum,
            AVG(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_h_avg,
            STDDEV(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_h_std,
            MEDIAN(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_h_median,
    
            SUM(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_h_sum,
            AVG(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_h_avg,
            STDDEV(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_h_std,
            MEDIAN(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_h_median,
    
            SUM(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_h_sum,
            AVG(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_h_avg,
            STDDEV(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_h_std,
            MEDIAN(total_h)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_h_median,
    
            -- ── runs ───────────────────────────────────────────────────────
            SUM(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_r_sum,
            AVG(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_r_avg,
            STDDEV(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_r_std,
            MEDIAN(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_r_median,
    
            SUM(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_r_sum,
            AVG(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_r_avg,
            STDDEV(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_r_std,
            MEDIAN(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_r_median,
    
            SUM(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_r_sum,
            AVG(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_r_avg,
            STDDEV(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_r_std,
            MEDIAN(total_r)
                OVER (PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_r_median
    
        FROM team_batter_stats
        ORDER BY
            team_id,
            game_date
    """).df()
    
    return team_base_features