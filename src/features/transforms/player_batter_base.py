import duckdb
from data_readers import read_batter_boxscore
import pandas as pd

pd.set_option('display.max_columns', None)


def compute_player_batter_base_features(year: str):
    batter_boxscore = read_batter_boxscore(year)

    con = duckdb.connect()
    con.register("batter_boxscore", batter_boxscore)

    # ── Step 1: player-game level (one row per personId per game) ──────────
    player_batter_stats = con.execute("""
        SELECT
            personId,
            team_id,
            gamepk,
            game_date,
            game_season,
            player_name,
            batting_order,

            -- counting (already player-level, but cast for consistency)
            plate_appearances::FLOAT    AS plate_appearances,
            ab::FLOAT                   AS ab,
            h::FLOAT                    AS h,
            r::FLOAT                    AS r,
            singles::FLOAT              AS singles,
            doubles::FLOAT              AS doubles,
            triples::FLOAT              AS triples,
            hr::FLOAT                   AS hr,
            total_bases::FLOAT          AS total_bases,
            total_bases_from_h::FLOAT   AS total_bases_from_h,
            rbi::FLOAT                  AS rbi,
            lob::FLOAT                  AS lob,
            sb::FLOAT                   AS sb,
            bb::FLOAT                   AS bb,
            k::FLOAT                    AS k,
            fantasy_points::FLOAT       AS fantasy_points


        FROM batter_boxscore
        ORDER BY personId, game_date
    """).df()

    con.register("player_batter_stats", player_batter_stats)

    # ── Step 2: rolling windows (mirrors team_base_features exactly) ───────
    player_batter_features = con.execute("""
        SELECT
            personId,
            team_id,
            gamepk,
            game_date,
            game_season,
            player_name,
            batting_order,

            -- raw counts (keep for downstream rate calculations)
            plate_appearances,
            ab,
            h,
            r,
            singles,
            doubles,
            triples,
            hr,
            total_bases,
            total_bases_from_h,
            rbi,
            lob,
            sb,
            bb,
            k,
            fantasy_points,

            --── hits ──────────────────────────────────────────────────────
            SUM(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_h_sum,
            AVG(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_h_avg,
            STDDEV(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_h_std,
            MEDIAN(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_h_median,

            SUM(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_h_sum,
            AVG(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_h_avg,
            STDDEV(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_h_std,
            MEDIAN(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_h_median,

            SUM(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_h_sum,
            AVG(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_h_avg,
            STDDEV(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_h_std,
            MEDIAN(h)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_h_median,

            --── strikeouts ────────────────────────────────────────────────
            SUM(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_k_sum,
            AVG(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_k_avg,
            STDDEV(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_k_std,
            MEDIAN(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_k_median,

            SUM(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_k_sum,
            AVG(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_k_avg,
            STDDEV(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_k_std,
            MEDIAN(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_k_median,

            SUM(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_k_sum,
            AVG(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_k_avg,
            STDDEV(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_k_std,
            MEDIAN(k)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_k_median,

            --── total bases ───────────────────────────────────────────────
            SUM(total_bases)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_tb_sum,
            AVG(total_bases)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_tb_avg,
            STDDEV(total_bases)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_tb_std,

            SUM(total_bases)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_tb_sum,
            AVG(total_bases)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_tb_avg,
            STDDEV(total_bases)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_tb_std,

            SUM(total_bases)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_tb_sum,
            AVG(total_bases)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_tb_avg,

            --── home runs ─────────────────────────────────────────────────
            SUM(hr)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_hr_sum,
            AVG(hr)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_hr_avg,

            SUM(hr)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_hr_sum,
            AVG(hr)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_hr_avg,

            --── rbi ───────────────────────────────────────────────────────
            SUM(rbi)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_rbi_sum,
            AVG(rbi)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_rbi_avg,

            SUM(rbi)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_rbi_sum,
            AVG(rbi)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_rbi_avg,

            --── walks ─────────────────────────────────────────────────────
            SUM(bb)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_bb_sum,
            AVG(bb)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_bb_avg,

            SUM(bb)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_bb_sum,
            AVG(bb)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_bb_avg,


            -- game count — tells you how much to trust the season rates
            COUNT(*)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_games_played,
                                         
            --── plate appearances ─────────────────────────────────────────────
            SUM(plate_appearances)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_pa_sum,
            AVG(plate_appearances)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)
                AS rolling_3game_pa_avg,

            SUM(plate_appearances)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_pa_sum,
            AVG(plate_appearances)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
                AS rolling_7game_pa_avg,

            SUM(plate_appearances)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_pa_sum,
            AVG(plate_appearances)
                OVER (PARTITION BY personId ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS season_pa_avg

        FROM player_batter_stats
        ORDER BY personId, game_date
    """).df()

    return player_batter_features