import duckdb
import pandas as pd
import numpy as np
from data_readers import read_pitcher_boxscore, read_playbyplay
from utils import mlb_ip_to_true, identify_starters_from_pbp

def compute_sp_base_features(year: str) -> pd.DataFrame:

    pitcher  = read_pitcher_boxscore(year)
    pbp      = read_playbyplay(year)

    # drop COVID season
    pitcher  = pitcher[pitcher["game_season"] != 2020]
    pbp      = pbp[pbp["game_season"] != 2020] if "game_season" in pbp.columns else pbp

    starters = identify_starters_from_pbp(pbp)

    # --- Step 1: filter to starters only ---
    sp = pitcher.merge(
        starters,
        left_on=["gamepk", "team_id", "personId"],
        right_on=["gamepk", "team_id", "starter_personId"],
        how="inner"
    ).copy()

    # --- Step 2: compute per-game rate stats ---
    sp["ip_true"] = mlb_ip_to_true(sp["ip"])

    safe_ip = sp["ip_true"].replace(0, np.nan)

    sp["era_gm"]  = ((sp["er"] / safe_ip * 9)).clip(upper=27)
    sp["whip_gm"] = ((sp["h"] + sp["bb"]) / safe_ip).clip(upper=9)
    sp["k9_gm"]   = (sp["k"]  / safe_ip * 9).clip(upper=27)
    sp["bb9_gm"]  = (sp["bb"] / safe_ip * 9).clip(upper=27)
    sp["hr9_gm"]  = (sp["hr"] / safe_ip * 9).clip(upper=9)

    sp["k_per_ip"]  = (sp["k"]  / safe_ip).clip(upper=3)
    sp["bb_per_ip"] = (sp["bb"] / safe_ip).clip(upper=3)
    sp["hr_per_ip"] = (sp["hr"] / safe_ip).clip(upper=1)
    sp["h_per_ip"]  = (sp["h"]  / safe_ip).clip(upper=4)
    sp["er_per_ip"] = (sp["er"] / safe_ip).clip(upper=3)

    sp["game_date"] = pd.to_datetime(sp["game_date"])
    sp = sp.sort_values(["personId", "game_date"])

    print(f"[SP] starters shape: {sp.shape}")
    print(f"[SP] null era_gm:  {sp['era_gm'].isna().sum()}")
    print(f"[SP] null whip_gm: {sp['whip_gm'].isna().sum()}")

    # --- Step 3: rolling windows ---
    con = duckdb.connect()
    con.register("sp_stats", sp)

    sp_rolling = con.execute("""
        SELECT
            personId,
            team_id,
            gamepk,
            game_date,
            game_season,
            starter_name   AS player_name,

            -- raw game stats
            ip_true,
            era_gm,
            whip_gm,
            k9_gm,
            bb9_gm,
            hr9_gm,
            k_per_ip,
            bb_per_ip,
            hr_per_ip,
            h_per_ip,
            er_per_ip,

            -- ── season games started ────────────────────────────────────
            COUNT(*) OVER (
                PARTITION BY personId ORDER BY game_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS season_games_started,

            -- ── last game stats ─────────────────────────────────────────
            AVG(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_era,
            AVG(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_whip,
            AVG(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_k9,
            AVG(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bb9,
            AVG(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_hr9,
            AVG(k_per_ip)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_k_per_ip,
            AVG(bb_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_bb_per_ip,
            AVG(hr_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_hr_per_ip,
            AVG(h_per_ip)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_h_per_ip,
            AVG(er_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_er_per_ip,
            AVG(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS last_game_ip,

            -- ── 3 start window ──────────────────────────────────────────
            AVG(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_era_avg,
            STDDEV(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_era_std,
            AVG(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_whip_avg,
            STDDEV(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_whip_std,
            AVG(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_k9_avg,
            STDDEV(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_k9_std,
            AVG(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_bb9_avg,
            STDDEV(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_bb9_std,
            AVG(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_hr9_avg,
            STDDEV(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_hr9_std,
            AVG(k_per_ip)    OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_k_per_ip_avg,
            STDDEV(k_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_k_per_ip_std,
            AVG(bb_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_bb_per_ip_avg,
            STDDEV(bb_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_bb_per_ip_std,
            AVG(hr_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_hr_per_ip_avg,
            STDDEV(hr_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_hr_per_ip_std,
            AVG(h_per_ip)    OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_h_per_ip_avg,
            STDDEV(h_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_h_per_ip_std,
            AVG(er_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_er_per_ip_avg,
            STDDEV(er_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_er_per_ip_std,
            AVG(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_ip_avg,
            STDDEV(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_3start_ip_std,

            -- ── 5 start window ──────────────────────────────────────────
            AVG(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_era_avg,
            STDDEV(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_era_std,
            AVG(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_whip_avg,
            STDDEV(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_whip_std,
            AVG(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_k9_avg,
            STDDEV(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_k9_std,
            AVG(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_bb9_avg,
            STDDEV(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_bb9_std,
            AVG(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_hr9_avg,
            STDDEV(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_hr9_std,
            AVG(k_per_ip)    OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_k_per_ip_avg,
            STDDEV(k_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_k_per_ip_std,
            AVG(bb_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_bb_per_ip_avg,
            STDDEV(bb_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_bb_per_ip_std,
            AVG(hr_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_hr_per_ip_avg,
            STDDEV(hr_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_hr_per_ip_std,
            AVG(h_per_ip)    OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_h_per_ip_avg,
            STDDEV(h_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_h_per_ip_std,
            AVG(er_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_er_per_ip_avg,
            STDDEV(er_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_er_per_ip_std,
            AVG(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_ip_avg,
            STDDEV(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_5start_ip_std,

            -- ── 10 start window ─────────────────────────────────────────
            AVG(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_era_avg,
            STDDEV(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_era_std,
            AVG(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_whip_avg,
            STDDEV(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_whip_std,
            AVG(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_k9_avg,
            STDDEV(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_k9_std,
            AVG(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_bb9_avg,
            STDDEV(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_bb9_std,
            AVG(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_hr9_avg,
            STDDEV(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_hr9_std,
            AVG(k_per_ip)    OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_k_per_ip_avg,
            STDDEV(k_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_k_per_ip_std,
            AVG(bb_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_bb_per_ip_avg,
            STDDEV(bb_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_bb_per_ip_std,
            AVG(hr_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_hr_per_ip_avg,
            STDDEV(hr_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_hr_per_ip_std,
            AVG(h_per_ip)    OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_h_per_ip_avg,
            STDDEV(h_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_h_per_ip_std,
            AVG(er_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_er_per_ip_avg,
            STDDEV(er_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_er_per_ip_std,
            AVG(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_ip_avg,
            STDDEV(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS rolling_10start_ip_std,

            -- ── season window ────────────────────────────────────────────
            AVG(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_era_avg,
            STDDEV(era_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_era_std,
            AVG(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_whip_avg,
            STDDEV(whip_gm) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_whip_std,
            AVG(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_k9_avg,
            STDDEV(k9_gm)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_k9_std,
            AVG(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bb9_avg,
            STDDEV(bb9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bb9_std,
            AVG(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_hr9_avg,
            STDDEV(hr9_gm)  OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_hr9_std,
            AVG(k_per_ip)    OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_k_per_ip_avg,
            STDDEV(k_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_k_per_ip_std,
            AVG(bb_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bb_per_ip_avg,
            STDDEV(bb_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_bb_per_ip_std,
            AVG(hr_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_hr_per_ip_avg,
            STDDEV(hr_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_hr_per_ip_std,
            AVG(h_per_ip)    OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_h_per_ip_avg,
            STDDEV(h_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_h_per_ip_std,
            AVG(er_per_ip)   OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_er_per_ip_avg,
            STDDEV(er_per_ip) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_er_per_ip_std,
            AVG(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_ip_avg,
            STDDEV(ip_true) OVER (PARTITION BY personId ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS season_ip_std

                             
        FROM sp_stats
        ORDER BY personId, game_date
    """).df()

    print(f"[SP] rolling output shape: {sp_rolling.shape}")
    return sp_rolling

