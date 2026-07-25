import duckdb
import pandas as pd
import numpy as np
from data_readers import read_pitcher_boxscore, read_playbyplay
from utils import mlb_ip_to_true, identify_starters_from_pbp

def compute_bullpen_features(year: str) -> pd.DataFrame:

    pitcher  = read_pitcher_boxscore(year)
    pbp      = read_playbyplay(year)

    # drop COVID season
    pitcher  = pitcher[pitcher["game_season"] != 2020]

    starters = identify_starters_from_pbp(pbp)

    # --- Step 1: filter to relievers only (anti-join) ---
    starters_keys = starters.rename(columns={"starter_personId": "personId"})[
        ["gamepk", "team_id", "personId"]
    ].assign(_is_starter=True)

    relievers = pitcher.merge(
        starters_keys,
        on=["gamepk", "team_id", "personId"],
        how="left"
    )
    relievers = relievers[relievers["_is_starter"].isna()].drop(columns=["_is_starter"]).copy()

    relievers["ip_true"] = mlb_ip_to_true(relievers["ip"])
    relievers["game_date"] = pd.to_datetime(relievers["game_date"])

    print(f"[BP] relievers shape: {relievers.shape}")

    # --- Step 2: aggregate to team level per game ---
    con = duckdb.connect()
    con.register("relievers", relievers)

    bp_game = con.execute("""
        SELECT
            team_id,
            gamepk,
            game_date,
            game_season,
            SUM(ip_true)              AS bp_ip_sum,
            SUM(h)::FLOAT             AS bp_h_sum,
            SUM(er)::FLOAT            AS bp_er_sum,
            SUM(bb)::FLOAT            AS bp_bb_sum,
            SUM(k)::FLOAT             AS bp_k_sum,
            SUM(hr)::FLOAT            AS bp_hr_sum,
            COUNT(DISTINCT personId)  AS bp_pitchers
        FROM relievers
        GROUP BY team_id, gamepk, game_date, game_season
        ORDER BY team_id, game_date
    """).df()

    # --- Step 3: compute team-level rate stats ---
    safe_ip = bp_game["bp_ip_sum"].replace(0, np.nan)

    bp_game["bp_era"]  = ((bp_game["bp_er_sum"] / safe_ip * 9)).clip(upper=27)
    bp_game["bp_whip"] = ((bp_game["bp_h_sum"] + bp_game["bp_bb_sum"]) / safe_ip).clip(upper=9)
    bp_game["bp_k9"]   = (bp_game["bp_k_sum"]  / safe_ip * 9).clip(upper=27)
    bp_game["bp_bb9"]  = (bp_game["bp_bb_sum"] / safe_ip * 9).clip(upper=27)
    bp_game["bp_hr9"]  = (bp_game["bp_hr_sum"] / safe_ip * 9).clip(upper=9)

    bp_game["bp_k_per_ip"]  = (bp_game["bp_k_sum"]  / safe_ip).clip(upper=3)
    bp_game["bp_bb_per_ip"] = (bp_game["bp_bb_sum"] / safe_ip).clip(upper=3)
    bp_game["bp_hr_per_ip"] = (bp_game["bp_hr_sum"] / safe_ip).clip(upper=1)
    bp_game["bp_h_per_ip"]  = (bp_game["bp_h_sum"]  / safe_ip).clip(upper=4)
    bp_game["bp_er_per_ip"] = (bp_game["bp_er_sum"] / safe_ip).clip(upper=3)

    print(f"[BP] game aggregates shape: {bp_game.shape}")
    print(f"[BP] null bp_era: {bp_game['bp_era'].isna().sum()}")

    # --- Step 4: rolling windows ---
    con.register("bp_game", bp_game)

    bp_rolling = con.execute("""
        SELECT
            team_id,
            gamepk,
            game_date,
            game_season,

            -- raw aggregated counts
            bp_ip_sum,
            bp_h_sum,
            bp_er_sum,
            bp_bb_sum,
            bp_k_sum,
            bp_hr_sum,
            bp_pitchers,

            -- raw rate stats
            bp_era,
            bp_whip,
            bp_k9,
            bp_bb9,
            bp_hr9,
            bp_k_per_ip,
            bp_bb_per_ip,
            bp_hr_per_ip,
            bp_h_per_ip,
            bp_er_per_ip,

            -- ── last game stats ─────────────────────────────────────────
            AVG(bp_era)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_era,
            AVG(bp_whip)     OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_whip,
            AVG(bp_k9)       OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_k9,
            AVG(bp_bb9)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_bb9,
            AVG(bp_hr9)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_hr9,
            AVG(bp_k_per_ip)  OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_k_per_ip,
            AVG(bp_bb_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_bb_per_ip,
            AVG(bp_hr_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_hr_per_ip,
            AVG(bp_h_per_ip)  OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_h_per_ip,
            AVG(bp_er_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_er_per_ip,
            AVG(bp_ip_sum)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_ip,
            AVG(bp_pitchers) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bp_pitchers,

            -- ── 7 game window ────────────────────────────────────────────
            AVG(bp_era)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_era_avg,
            STDDEV(bp_era)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_era_std,
            AVG(bp_whip)     OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_whip_avg,
            STDDEV(bp_whip)  OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_whip_std,
            AVG(bp_k9)       OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_k9_avg,
            STDDEV(bp_k9)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_k9_std,
            AVG(bp_bb9)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_bb9_avg,
            STDDEV(bp_bb9)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_bb9_std,
            AVG(bp_hr9)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_hr9_avg,
            STDDEV(bp_hr9)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_hr9_std,
            AVG(bp_k_per_ip)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_k_per_ip_avg,
            STDDEV(bp_k_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_k_per_ip_std,
            AVG(bp_bb_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_bb_per_ip_avg,
            STDDEV(bp_bb_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_bb_per_ip_std,
            AVG(bp_hr_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_hr_per_ip_avg,
            STDDEV(bp_hr_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_hr_per_ip_std,
            AVG(bp_h_per_ip)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_h_per_ip_avg,
            STDDEV(bp_h_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_h_per_ip_std,
            AVG(bp_er_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_er_per_ip_avg,
            STDDEV(bp_er_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_er_per_ip_std,
            AVG(bp_ip_sum)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_ip_avg,
            STDDEV(bp_ip_sum) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_ip_std,
            AVG(bp_pitchers) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_pitchers_avg,
            STDDEV(bp_pitchers) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_7game_bp_pitchers_std,

            -- ── 14 game window ───────────────────────────────────────────
            AVG(bp_era)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_era_avg,
            STDDEV(bp_era)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_era_std,
            AVG(bp_whip)     OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_whip_avg,
            STDDEV(bp_whip)  OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_whip_std,
            AVG(bp_k9)       OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_k9_avg,
            STDDEV(bp_k9)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_k9_std,
            AVG(bp_bb9)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_bb9_avg,
            STDDEV(bp_bb9)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_bb9_std,
            AVG(bp_hr9)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_hr9_avg,
            STDDEV(bp_hr9)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_hr9_std,
            AVG(bp_k_per_ip)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_k_per_ip_avg,
            STDDEV(bp_k_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_k_per_ip_std,
            AVG(bp_bb_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_bb_per_ip_avg,
            STDDEV(bp_bb_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_bb_per_ip_std,
            AVG(bp_hr_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_hr_per_ip_avg,
            STDDEV(bp_hr_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_hr_per_ip_std,
            AVG(bp_h_per_ip)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_h_per_ip_avg,
            STDDEV(bp_h_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_h_per_ip_std,
            AVG(bp_er_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_er_per_ip_avg,
            STDDEV(bp_er_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_er_per_ip_std,
            AVG(bp_ip_sum)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_ip_avg,
            STDDEV(bp_ip_sum) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_ip_std,
            AVG(bp_pitchers) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_pitchers_avg,
            STDDEV(bp_pitchers) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_14game_bp_pitchers_std,

            -- ── season window ────────────────────────────────────────────
            AVG(bp_era)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_era_avg,
            STDDEV(bp_era)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_era_std,
            AVG(bp_whip)     OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_whip_avg,
            STDDEV(bp_whip)  OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_whip_std,
            AVG(bp_k9)       OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_k9_avg,
            STDDEV(bp_k9)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_k9_std,
            AVG(bp_bb9)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_bb9_avg,
            STDDEV(bp_bb9)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_bb9_std,
            AVG(bp_hr9)      OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_hr9_avg,
            STDDEV(bp_hr9)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_hr9_std,
            AVG(bp_k_per_ip)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_k_per_ip_avg,
            STDDEV(bp_k_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_k_per_ip_std,
            AVG(bp_bb_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_bb_per_ip_avg,
            STDDEV(bp_bb_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_bb_per_ip_std,
            AVG(bp_hr_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_hr_per_ip_avg,
            STDDEV(bp_hr_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_hr_per_ip_std,
            AVG(bp_h_per_ip)    OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_h_per_ip_avg,
            STDDEV(bp_h_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_h_per_ip_std,
            AVG(bp_er_per_ip)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_er_per_ip_avg,
            STDDEV(bp_er_per_ip) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_er_per_ip_std,
            AVG(bp_ip_sum)   OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_ip_avg,
            STDDEV(bp_ip_sum) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_ip_std,
            AVG(bp_pitchers) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_pitchers_avg,
            STDDEV(bp_pitchers) OVER (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bp_pitchers_std

        FROM bp_game
        ORDER BY team_id, game_date
    """).df()

    print(f"[BP] rolling output shape: {bp_rolling.shape}")
    return bp_rolling