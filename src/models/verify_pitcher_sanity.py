#!/usr/bin/env python3
"""
verify_pitcher_sanity.py
=========================
Sanity checks on sp_rolling and pa_feats for the top pitcher by starts (2023).

Usage:
    python verify_pitcher_sanity.py
"""
import pandas as pd

YEAR = 2023

from build_pa_pitcher_features import (
    _read_pbp,
    _read_pitcher_box,
    _read_batter_box,
    _read_schedule,
    _read_player_info,
    _identify_starters,
    _extract_sp_pa,
    _pbp_pitcher_game_aggs,
    _pbp_pitcher_inn45_aggs,
    _pbp_batter_game_aggs,
    _build_pitcher_game,
    _build_batter_game,
    _pitcher_rolling,
    _batter_rolling,
    _add_career_avg_fallback,
    _add_last_season_avgs,
    _add_log5,
    SP_STAT_COLS, BAT_STAT_COLS, PARK_FACTORS, EXCLUDE_YEARS,
)
import numpy as np

# ── Load 2023 data ────────────────────────────────────────────────────────────
print(f"Loading {YEAR} data …")
pbp         = _read_pbp(YEAR)
pitcher_box = _read_pitcher_box(YEAR)
batter_box  = _read_batter_box(YEAR)
schedule    = _read_schedule(YEAR)
player_info = _read_player_info()

pitcher_box["game_date"] = pd.to_datetime(pitcher_box["game_date"])
batter_box["game_date"]  = pd.to_datetime(batter_box["game_date"])

starters     = _identify_starters(pbp)
pbp_sp_aggs  = _pbp_pitcher_game_aggs(pbp)
inn45_aggs   = _pbp_pitcher_inn45_aggs(pbp)
pbp_bat_aggs = _pbp_batter_game_aggs(pbp)

pitcher_game = _build_pitcher_game(pitcher_box, pbp_sp_aggs, starters)
batter_game  = _build_batter_game(batter_box, pbp_bat_aggs)

sp_rolling  = _pitcher_rolling(pitcher_game, inn45_aggs=inn45_aggs)
bat_rolling = _batter_rolling(batter_game)

sp_rolling  = _add_career_avg_fallback(
    sp_rolling, pitcher_game, SP_STAT_COLS,
    ytd_pa_col="spytd_pa_count", ytd_prefix="spytd_", out_prefix="sp_",
)
bat_rolling = _add_career_avg_fallback(
    bat_rolling, batter_game, BAT_STAT_COLS,
    ytd_pa_col="batytd_pa_count", ytd_prefix="batytd_", out_prefix="bat_",
)
sp_rolling  = _add_last_season_avgs(sp_rolling, pitcher_game, SP_STAT_COLS, out_prefix="sp_")
bat_rolling = _add_last_season_avgs(bat_rolling, batter_game, BAT_STAT_COLS, out_prefix="bat_")

# ── Build sp_pa and join to make pa_feats ─────────────────────────────────────
from build_pa_pitcher_features import (
    _add_target, _add_tto, _add_pitch_count, _add_base_out_state, _add_handedness,
)

sp_pa = _extract_sp_pa(pbp, starters)
game_meta = pitcher_box[["gamepk", "game_date", "game_season"]].drop_duplicates("gamepk")
sp_pa = sp_pa.merge(game_meta, on="gamepk", how="left")
sp_pa["game_date"] = pd.to_datetime(sp_pa["game_date"])

venue_map = (
    schedule[["gamepk", "venue_id"]]
    .drop_duplicates("gamepk")
    .assign(park_factor=lambda d: d["venue_id"].astype(str).map(PARK_FACTORS).fillna(100.0))
)

sp_pa = (
    sp_pa
    .pipe(_add_target)
    .pipe(_add_tto)
    .pipe(_add_pitch_count, pbp=pbp)
    .pipe(_add_base_out_state)
    .pipe(_add_handedness, player_info=player_info)
    .merge(venue_map, on="gamepk", how="left")
)

sp_raw  = set(SP_STAT_COLS + ["k_bb_ratio", "pa_count", "ip",
                               "k_rate", "bb_rate", "hr_rate"])
bat_raw = set(BAT_STAT_COLS + ["plate_appearances",
                                "k_rate", "bb_rate", "hr_rate", "iso", "tb_rate",
                                "chase_rate", "z_contact_pct", "o_contact_pct"])

sp_join_cols  = [c for c in sp_rolling.columns  if c not in sp_raw  or c in ("gamepk", "personId")]
bat_join_cols = [c for c in bat_rolling.columns if c not in bat_raw or c in ("gamepk", "personId")]

pa_feats = (
    sp_pa
    .merge(
        sp_rolling[sp_join_cols].rename(columns={"personId": "pitcher_id"}),
        on=["gamepk", "pitcher_id"],
        how="left",
    )
    .merge(
        bat_rolling[bat_join_cols].rename(columns={"personId": "batter_id"}),
        on=["gamepk", "batter_id"],
        how="left",
        suffixes=("", "_bat"),
    )
)
pa_feats = _add_log5(pa_feats, pitcher_game)
pa_feats["post_shift_ban"] = (pa_feats["game_date"] >= "2023-03-30").astype("int8")

print(f"pa_feats shape: {pa_feats.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Top 5 pitchers by start count in sp_rolling
# ══════════════════════════════════════════════════════════════════════════════
start_counts = (
    sp_rolling.groupby("personId")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="n_starts")
)

print("\n=== Top 5 pitchers by starts in sp_rolling ===")
print(start_counts.head(5).to_string(index=False))

print(f"\nsp_rolling personId dtype: {sp_rolling['personId'].dtype}")
print(f"pa_feats pitcher_id dtype:  {pa_feats['pitcher_id'].dtype}")

# Normalise to string — pitcher IDs are strings throughout the pipeline
chosen_id = str(start_counts.iloc[0]["personId"])
n_starts  = start_counts.iloc[0]["n_starts"]
print(f"\nChosen pitcher personId: {chosen_id}  (n_starts = {n_starts})")

# Cast sp_rolling personId to string for safe filter
sp_rolling = sp_rolling.assign(personId=sp_rolling["personId"].astype(str))

pitcher_df = sp_rolling[sp_rolling["personId"] == chosen_id].sort_values("game_date").reset_index(drop=True)
print(f"pitcher_df rows: {len(pitcher_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Chronological view: last start features
#    Check: row 0 is null, row 1+ populated, row N sp_last_start_k_rate == row N-1 k_rate
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Check 1: Chronological last-start features ===")
cols_ls = ["game_date", "k_rate", "sp_last_start_k_rate", "sp_last_start_ip", "sp_last_start_pitches"]
print(pitcher_df[cols_ls].to_string())

# Lag alignment: sp_last_start_k_rate[i] should == k_rate[i-1]
lag_match = (
    pitcher_df["sp_last_start_k_rate"]
    .round(6)
    .eq(pitcher_df["k_rate"].shift(1).round(6))
    .dropna()
)
print(f"\nLag alignment (sp_last_start_k_rate[i] == k_rate[i-1]): "
      f"{lag_match.sum()} / {len(lag_match)} match")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Rolling smoothness: sp_last_start_ip < sp3_avg_ip < sp10_avg_ip (std order)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Check 2: Rolling window smoothness ===")
cols_ip = ["game_date", "sp_last_start_ip", "sp3_avg_ip", "sp10_avg_ip"]
print(pitcher_df[cols_ip].to_string())

stds = pitcher_df[["sp_last_start_ip", "sp3_avg_ip", "sp10_avg_ip"]].std()
print(f"\nStd (should decrease from last_start → sp3 → sp10):")
print(stds.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 4. sp_rolling game-level cluster check (features constant within gamepk)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Check 3: sp_rolling game-level cluster (sp_last_start_pitches nunique per gamepk) ===")
cluster = (
    sp_rolling[sp_rolling["personId"] == chosen_id]
    .groupby("gamepk")["sp_last_start_pitches"]
    .nunique()
    .value_counts()
)
print(cluster.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. sp3_n_starts and sp10_n_starts value counts
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Check 4: Window fill — sp3_n_starts and sp10_n_starts value counts ===")
print("sp3_n_starts:")
print(pitcher_df["sp3_n_starts"].value_counts().sort_index().to_string())
print("\nsp10_n_starts:")
print(pitcher_df["sp10_n_starts"].value_counts().sort_index().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 6. pa_feats game-level leakage detector
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Check 5: pa_feats game-level leakage (all nunique should be 1) ===")
game_check = (
    pa_feats[pa_feats["pitcher_id"] == chosen_id]
    .groupby("gamepk")
    .agg(
        n_pas                    =("reached_base",          "count"),
        unique_last_start_pitches=("sp_last_start_pitches", "nunique"),
        unique_sp3_avg_ip        =("sp3_avg_ip",            "nunique"),
        unique_fatigue_score     =("fatigue_score",         "nunique"),
        min_last_start_k         =("sp_last_start_k_rate",  "min"),
        max_last_start_k         =("sp_last_start_k_rate",  "max"),
    )
    .reset_index()
)
print(game_check.head(10).to_string())

leakage_cols = ["unique_last_start_pitches", "unique_sp3_avg_ip", "unique_fatigue_score"]
for col in leakage_cols:
    bad = (game_check[col] > 1).sum()
    print(f"{col}: {bad} games with nunique > 1  {'⚠ LEAKAGE' if bad else '✓ clean'}")
