#!/usr/bin/env python3
"""
build_pa_pitcher_features.py
============================
Self-contained script: S3 → feature engineering → S3.

Target  : bases earned by batter on the PA (0=out, 1=1B/BB/HBP, 2=2B, 3=3B, 4=HR)
Grain   : one row per plate appearance, starting pitchers only
Years   : 2015–2024 excluding 2020–2021
Output  : s3://mlbdk/processed_data/features/plate_appearances/{year}.parquet

Rolling window rules
  - All windows strictly game_date < current_game_date (zero leakage)
  - Performance windows are games-based (ROWS BETWEEN N PRECEDING AND 1 PRECEDING)
  - Days-based only for: days_rest, cumulative_ip_last_7d

Temporal split (no random shuffle)
  Train : 2015–2022  |  Val : 2023  |  Test : 2024

Usage
  python build_pa_pitcher_features.py --years 2022 2023 2024
"""
import argparse
import logging
import sys

import awswrangler as wr
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BUCKET = "mlbdk"

# ── Event → bases earned by batter ───────────────────────────────────────────
# Keys match play_result values in the prepared PBP data (Title Case, spaces).
BASES_MAP = {
    # Hits
    "Home Run": 4, "Triple": 3, "Double": 2, "Single": 1,
    # Other on-base events
    "Walk": 1, "Intent Walk": 1, "Hit By Pitch": 1,
    "Field Error": 1, "Catcher Interference": 1,
    # Outs
    "Strikeout": 0, "Strikeout Double Play": 0,
    "Groundout": 0, "Flyout": 0, "Lineout": 0, "Pop Out": 0,
    "Forceout": 0, "Grounded Into DP": 0, "Double Play": 0,
    "Triple Play": 0, "Fielders Choice": 0, "Fielders Choice Out": 0,
    "Sac Fly": 0, "Sac Bunt": 0, "Sac Fly Double Play": 0,
    "Bunt Groundout": 0, "Bunt Lineout": 0, "Bunt Pop Out": 0,
    "Runner Out": 0, "Field Out": 0,
}

# MLB 13-zone: 1-9 = in-zone, 11-14 = out-of-zone (chase territory)
IN_ZONE  = frozenset(range(1, 10))
OUT_ZONE = frozenset(range(11, 15))

# Pitch-call codes
SWING_CODES   = frozenset(["S", "W", "T", "F", "X", "D", "E", "L", "M"])
CONTACT_CODES = frozenset(["F", "X", "T", "D", "E"])

# Static park factors (100 = league avg); key = venue_id as str
PARK_FACTORS: dict[str, float] = {
    "1": 103, "3": 96, "4": 102, "5": 100, "7": 98, "10": 97,
    "12": 103, "14": 101, "15": 97, "17": 102, "19": 98, "20": 100,
    "21": 96, "22": 102, "28": 97, "31": 100, "32": 97, "34": 100,
    "36": 98, "2392": 100, "2394": 99, "2395": 97, "2508": 102,
}

EXCLUDE_YEARS  = {2020, 2021}
TARGET_YEARS   = [y for y in range(2017, 2025) if y not in EXCLUDE_YEARS]

# Stat columns used for rolling + career-avg computation
SP_STAT_COLS = [
    "k_rate", "bb_rate", "hr_rate",
    "gb_pct", "csw_pct", "whiff_rate", "barrel_pct",
    "hard_hit_pct", "first_pitch_strike_pct", "p_per_pa",
]
BAT_STAT_COLS = [
    "k_rate", "bb_rate", "hr_rate",
    "iso", "tb_rate", "chase_rate", "z_contact_pct", "o_contact_pct",
]

# Shrinkage K constants for platoon splits (PA needed to half-weight the split)
PLATOON_K = {
    "k_rate":    50,
    "bb_rate":   120,
    "obp":       150,
    "iso":       200,
    "tb_rate":   150,
    "whiff_rate": 75,
}


# ═════════════════════════════════════════════════════════════════════════════
# S3 loaders
# ═════════════════════════════════════════════════════════════════════════════

def _read_pbp(year: int) -> pd.DataFrame:
    return wr.s3.read_parquet(f"s3://{BUCKET}/processed_data/prepared/playbyplay/{year}/")

def _read_pitcher_box(year: int) -> pd.DataFrame:
    return wr.s3.read_parquet(f"s3://{BUCKET}/processed_data/prepared/pitcher_boxscore/{year}/")

def _read_batter_box(year: int) -> pd.DataFrame:
    return wr.s3.read_parquet(f"s3://{BUCKET}/processed_data/prepared/batter_boxscore/{year}/")

def _read_schedule(year: int) -> pd.DataFrame:
    return wr.s3.read_parquet(f"s3://{BUCKET}/processed_data/games/schedule/{year}/")

def _read_player_info() -> pd.DataFrame:
    return wr.s3.read_parquet(f"s3://{BUCKET}/reference/player_info/player_info.parquet")


test = _read_pbp(2025)
test.columns.tolist()

test.play_result.value_counts()

test = _read_batter_box(2025)

test
# ═════════════════════════════════════════════════════════════════════════════
# Starters + SP plate appearances
# ═════════════════════════════════════════════════════════════════════════════

def _identify_starters(pbp: pd.DataFrame) -> pd.DataFrame:
    """First pitcher to appear per (gamepk, pitcher_team_id)."""
    return (
        pbp
        .sort_values(["gamepk", "pitcher_team_id", "inning", "event_index"])
        .groupby(["gamepk", "pitcher_team_id"], sort=False)
        .first()
        .reset_index()
        [["gamepk", "pitcher_team_id", "pitcher_id", "pitcher_name"]]
        .rename(columns={"pitcher_team_id": "team_id", "pitcher_id": "starter_id"})
    )


def _extract_sp_pa(pbp: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse pitch-level PBP → one row per PA (last event of each play),
    then keep only PAs faced by the identified game starter.
    """
    pa = (
        pbp
        .sort_values(["gamepk", "play_id", "event_index"])
        .groupby(["gamepk", "play_id"], sort=False)
        .last()
        .reset_index()
        .pipe(lambda d: d[d["play_result"].notna()])
        .copy()
    )
    starter_index = starters.set_index(["gamepk", "starter_id"]).index
    mask = pd.MultiIndex.from_arrays([pa["gamepk"], pa["pitcher_id"]]).isin(starter_index)
    return pa[mask].copy()


# ═════════════════════════════════════════════════════════════════════════════
# Game-context features (PA level)
# ═════════════════════════════════════════════════════════════════════════════

def _add_target(pa: pd.DataFrame) -> pd.DataFrame:
    bases = pa["play_result"].map(BASES_MAP).fillna(0)
    return pa.assign(
        target_bases=bases.astype("int8"),
        reached_base=(bases > 0).astype("int8"),
    )


def _add_tto(pa: pd.DataFrame) -> pd.DataFrame:
    """Times-through-order (1/2/3+) for each PA."""
    pa = pa.sort_values(["gamepk", "pitcher_id", "event_index"])
    return pa.assign(
        times_through_order=(
            pa.groupby(["gamepk", "pitcher_id", "batter_id"])
            .cumcount()
            .add(1)
            .clip(upper=3)
        )
    )


def _add_pitch_count(pa: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Cumulative pitches thrown by pitcher BEFORE each PA starts."""
    counts = (
        pbp[pbp["is_pitch"]]
        .groupby(["gamepk", "pitcher_id", "play_id"], sort=False)
        .size()
        .reset_index(name="n_pitches")
        .sort_values(["gamepk", "pitcher_id", "play_id"])
        .assign(
            pitch_count_in_game=lambda d: (
                d.groupby(["gamepk", "pitcher_id"])["n_pitches"]
                .transform(lambda s: s.cumsum().shift(fill_value=0))
                .astype(int)
            )
        )
    )
    return pa.merge(
        counts[["gamepk", "pitcher_id", "play_id", "pitch_count_in_game"]],
        on=["gamepk", "pitcher_id", "play_id"],
        how="left",
    )


def _add_base_out_state(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Base-out state (0–23) at the START of each PA.
    Encoding: outs * 8 + base_bitmask  (bit0=1B, bit1=2B, bit2=3B).
    Simplified runner-advancement model: good enough as a predictive feature.
    """
    pa = pa.sort_values(["gamepk", "half_inning", "inning", "event_index"]).copy()

    def _track(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("event_index").reset_index(drop=True)
        on_1b = on_2b = on_3b = False
        outs = 0
        base_out_states, outs_col = [], []

        for _, row in grp.iterrows():
            bitmask = int(on_1b) | (int(on_2b) << 1) | (int(on_3b) << 2)
            base_out_states.append(outs * 8 + bitmask)
            outs_col.append(outs)

            et = str(row.get("event_type") or "")
            if et == "home_run":
                on_1b = on_2b = on_3b = False
            elif et == "triple":
                on_1b = on_2b = False; on_3b = True
            elif et == "double":
                on_1b = False; on_2b = True; on_3b = False
            elif et == "single":
                on_3b = False; on_2b = on_1b; on_1b = True
            elif et in ("walk", "intent_walk", "hit_by_pitch", "catcher_interf"):
                if on_1b and on_2b: on_3b = True
                elif on_1b:         on_2b = True
                on_1b = True
            elif et in ("strikeout", "strikeout_double_play", "field_out",
                        "bunt_groundout", "bunt_lineout", "bunt_pop_out", "other_out"):
                outs = min(outs + 1, 3)
            elif et in ("grounded_into_double_play", "double_play", "runner_double_play"):
                outs = min(outs + 2, 3); on_1b = False
            elif et == "triple_play":
                outs = 3; on_1b = on_2b = on_3b = False
            elif et in ("fielders_choice", "fielders_choice_out", "force_out"):
                outs = min(outs + 1, 3); on_1b = True
            elif et in ("sac_fly", "sac_fly_double_play"):
                outs = min(outs + 1, 3); on_3b = False
            elif et == "sac_bunt":
                outs = min(outs + 1, 3)
                on_3b = on_2b; on_2b = on_1b; on_1b = False
            elif et == "field_error":
                on_1b = True

        grp["base_out_state"] = base_out_states
        grp["outs_when_up"]   = outs_col
        return grp

    return (
        pa
        .groupby(["gamepk", "half_inning", "inning"], sort=False, group_keys=False)
        .apply(_track)
    )


def _add_handedness(pa: pd.DataFrame, player_info: pd.DataFrame) -> pd.DataFrame:
    """
    Join batter/pitcher handedness from player_info and compute platoon matchup features.

    Resolution rules:
      - pitchHand == "S"  → treat as "R"
      - batSide  == "S"  → bats opposite resolved pitcher hand (R→L, L→R)

    New columns (int8):
      platoon_advantage  1 if L vs R or R vs L after resolution
      same_hand          1 if batter and pitcher share the same hand after resolution
    """
    assert "personId" in player_info.columns, \
        "player_info missing personId — normalize at load time in build_pa_features"

    # Cast join keys to the same dtype to prevent silent merge mismatches
    id_dtype = pa["batter_id"].dtype
    player_info = player_info.assign(personId=lambda d: d["personId"].astype(id_dtype))

    missing_batters = pa[~pa["batter_id"].isin(player_info["personId"])]["batter_id"].unique()
    missing_pitchers = pa[~pa["pitcher_id"].isin(player_info["personId"])]["pitcher_id"].unique()
    print(f"Missing batters: {len(missing_batters)}")
    print(f"Missing pitchers: {len(missing_pitchers)}")
    print(f"Sample missing batter IDs: {missing_batters[:10]}")
    print(f"Sample missing pitcher IDs: {missing_pitchers[:10]}")
    print(f"player_info total rows: {len(player_info)}")
    print(f"player_info personId dtype: {player_info['personId'].dtype}")
    print(f"sp_pa batter_id dtype: {pa['batter_id'].dtype}")

    sides = (
        player_info[["personId", "batSide", "pitchHand"]]
        .drop_duplicates("personId")
        .astype({"batSide": object, "pitchHand": object})
    )

    return (
        pa
        .merge(
            sides[["personId", "batSide"]].rename(
                columns={"personId": "batter_id", "batSide": "bat_side_raw"}
            ),
            on="batter_id",
            how="left",
        )
        .merge(
            sides[["personId", "pitchHand"]].rename(
                columns={"personId": "pitcher_id", "pitchHand": "pitch_hand_raw"}
            ),
            on="pitcher_id",
            how="left",
        )
        .assign(
            pitch_hand_resolved=lambda d: np.where(
                d["pitch_hand_raw"] == "S", "R", d["pitch_hand_raw"]
            ),
            bat_side_resolved=lambda d: np.where(
                d["bat_side_raw"] == "S",
                np.where(d["pitch_hand_resolved"] == "R", "L", "R"),
                d["bat_side_raw"],
            ),
            platoon_advantage=lambda d: (
                ((d["bat_side_resolved"] == "L") & (d["pitch_hand_resolved"] == "R"))
                | ((d["bat_side_resolved"] == "R") & (d["pitch_hand_resolved"] == "L"))
            ).astype("int8"),
            same_hand=lambda d: (
                d["bat_side_resolved"] == d["pitch_hand_resolved"]
            ).astype("int8"),
        )
        .drop(columns=["bat_side_raw", "pitch_hand_raw", "bat_side_resolved", "pitch_hand_resolved"])
    )


# ═════════════════════════════════════════════════════════════════════════════
# PBP per-game aggregates (inputs to rolling features)
# ═════════════════════════════════════════════════════════════════════════════

def _pbp_pitcher_game_aggs(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Pitch-quality metrics per (pitcher_id, gamepk) from pitch-level PBP.
    Returns: gb_pct, csw_pct, whiff_rate, barrel_pct, hard_hit_pct,
             first_pitch_strike_pct, p_per_pa, pa_count.
    """
    pitches = pbp[pbp["is_pitch"]].copy().assign(
        is_called_strike=lambda d: d["pitch_call_code"].eq("C"),
        is_swing_miss   =lambda d: d["pitch_call_code"].isin(["S", "W", "T"]),
        is_swing        =lambda d: d["pitch_call_code"].isin(SWING_CODES),
        is_contact      =lambda d: d["pitch_call_code"].isin(CONTACT_CODES),
        is_groundball   =lambda d: d["trajectory"].fillna("").str.lower().eq("ground_ball"),
        is_barrel       =lambda d: (
            d["launch_speed"].ge(98) & d["launch_angle"].between(26, 30)
        ).fillna(False),
        is_hard_hit     =lambda d: d["launch_speed"].ge(95).fillna(False),
        is_first_pitch  =lambda d: d["pitch_number"].eq(1),
    ).assign(
        is_fp_strike=lambda d: d["is_first_pitch"] & (
            d["is_called_strike"] | d["is_swing_miss"] | d["is_contact"]
        )
    )

    pa_counts = (
        pbp[pbp["play_result"].notna()]
        .groupby(["gamepk", "pitcher_id"])["play_id"]
        .nunique()
        .rename("pa_count")
        .reset_index()
    )

    return (
        pitches
        .groupby(["gamepk", "pitcher_id"], sort=False)
        .agg(
            total_pitches =("is_pitch",        "sum"),
            called_strikes=("is_called_strike", "sum"),
            swing_misses  =("is_swing_miss",    "sum"),
            swings        =("is_swing",         "sum"),
            groundballs   =("is_groundball",    "sum"),
            in_play       =("is_in_play",       "sum"),
            barrels       =("is_barrel",        "sum"),
            hard_hits     =("is_hard_hit",      "sum"),
            fp_strikes    =("is_fp_strike",     "sum"),
            fp_total      =("is_first_pitch",   "sum"),
        )
        .reset_index()
        .merge(pa_counts, on=["gamepk", "pitcher_id"], how="left")
        .assign(
            gb_pct                =lambda d: d["groundballs"]   / d["in_play"].clip(lower=1),
            csw_pct               =lambda d: (d["called_strikes"] + d["swing_misses"]) / d["total_pitches"].clip(lower=1),
            whiff_rate            =lambda d: d["swing_misses"]  / d["swings"].clip(lower=1),
            barrel_pct            =lambda d: d["barrels"]       / d["in_play"].clip(lower=1),
            hard_hit_pct          =lambda d: d["hard_hits"]     / d["in_play"].clip(lower=1),
            first_pitch_strike_pct=lambda d: d["fp_strikes"]    / d["fp_total"].clip(lower=1),
            p_per_pa              =lambda d: d["total_pitches"] / d["pa_count"].clip(lower=1),
        )
        [["gamepk", "pitcher_id", "pa_count", "total_pitches",
          "gb_pct", "csw_pct", "whiff_rate", "barrel_pct",
          "hard_hit_pct", "first_pitch_strike_pct", "p_per_pa"]]
    )


def _pbp_batter_game_aggs(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Plate-discipline metrics per (batter_id, gamepk).
    Returns: chase_rate, z_contact_pct, o_contact_pct.
    """
    pitches = pbp[pbp["is_pitch"]].copy().assign(
        in_zone  =lambda d: d["zone"].isin(IN_ZONE).fillna(False),
        out_zone =lambda d: d["zone"].isin(OUT_ZONE).fillna(False),
        is_swing =lambda d: d["pitch_call_code"].isin(SWING_CODES),
        is_contact=lambda d: d["pitch_call_code"].isin(CONTACT_CODES),
    ).assign(
        chase    =lambda d: d["out_zone"] & d["is_swing"],
        z_swing  =lambda d: d["in_zone"]  & d["is_swing"],
        z_contact=lambda d: d["in_zone"]  & d["is_swing"] & d["is_contact"],
        o_swing  =lambda d: d["out_zone"] & d["is_swing"],
        o_contact=lambda d: d["out_zone"] & d["is_swing"] & d["is_contact"],
    )

    return (
        pitches
        .groupby(["gamepk", "batter_id"], sort=False)
        .agg(
            out_zone_n  =("out_zone",  "sum"),
            chase_n     =("chase",     "sum"),
            z_swing_n   =("z_swing",   "sum"),
            z_contact_n =("z_contact", "sum"),
            o_swing_n   =("o_swing",   "sum"),
            o_contact_n =("o_contact", "sum"),
            pitches_seen=("is_pitch",  "sum"),
        )
        .reset_index()
        .assign(
            chase_rate   =lambda d: d["chase_n"]     / d["out_zone_n"].clip(lower=1),
            z_contact_pct=lambda d: d["z_contact_n"] / d["z_swing_n"].clip(lower=1),
            o_contact_pct=lambda d: d["o_contact_n"] / d["o_swing_n"].clip(lower=1),
        )
        [["gamepk", "batter_id",
          "chase_rate", "z_contact_pct", "o_contact_pct",
          "chase_n", "out_zone_n", "z_swing_n", "z_contact_n", "pitches_seen"]]
    )


# On-base events (hit + walk + HBP) used for OBP-allowed computation
OBP_EVENTS = frozenset([
    "Single", "Double", "Triple", "Home Run",
    "Walk", "Intent Walk", "Hit By Pitch",
    "Field Error", "Catcher Interference",
])


def _pbp_pitcher_inn45_aggs(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (pitcher_id, gamepk) aggregates for innings 4 and 5 only.
    Returns raw counts; per-game rates (inn45_p_per_pa_rate, inn45_obp_rate)
    are computed downstream so that DuckDB can window over them.
    Games where the pitcher did not pitch into inn 4-5 are absent (left-join
    in the caller produces NaN, which AVG windows naturally skip).
    """
    inn45 = pbp[pbp["inning"].isin([4, 5])].copy()

    pitch_counts = (
        inn45[inn45["is_pitch"]]
        .groupby(["gamepk", "pitcher_id"], sort=False)
        .size()
        .reset_index(name="inn45_pitches")
    )

    pa_events = (
        inn45[inn45["play_result"].notna()]
        .drop_duplicates(subset=["gamepk", "pitcher_id", "play_id"])
        .assign(is_onbase=lambda d: d["play_result"].isin(OBP_EVENTS))
        .groupby(["gamepk", "pitcher_id"], sort=False)
        .agg(
            inn45_pa     =("play_result", "count"),
            inn45_onbase =("is_onbase",   "sum"),
        )
        .reset_index()
    )

    return pitch_counts.merge(pa_events, on=["gamepk", "pitcher_id"], how="outer")


def _pbp_pitcher_process_aggs(
    pbp: pd.DataFrame,
    player_info: pd.DataFrame,
    starters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Statcast-based process quality metrics per (pitcher_id, gamepk).
    Filtered to starter pitches only via inner join on starters table.
    Returns: stuff quality (avg_velo, avg_spin_rate, avg_extension, avg_h_break, avg_v_break),
             contact quality — in-play only and all-contact (foul tier omitted: exit velocity
             not tracked on foul balls in this dataset),
             command metrics (zone_pct, chase_induced_pct, two_strike_whiff_pct).
    Assumes 2017+ Statcast data. contact_count < 2 → contact metrics nulled.
    """
    starter_keys = starters.set_index(["gamepk", "starter_id"]).index
    pitch_mask = pd.MultiIndex.from_arrays(
        [pbp["gamepk"], pbp["pitcher_id"]]
    ).isin(starter_keys)
    pbp = pbp[pitch_mask].copy()

    # count_strikes/count_balls are post-pitch; derive pre-pitch counts via lag within at-bat
    _pitches_raw = pbp[pbp["is_pitch"]].copy().sort_values(
        ["gamepk", "play_id", "event_index"]
    )
    _gb_play = _pitches_raw.groupby(["gamepk", "play_id"], sort=False)
    _pitches_raw["pre_count_strikes"] = _gb_play["count_strikes"].shift(1).fillna(0).astype(int)
    _pitches_raw["pre_count_balls"] = _gb_play["count_balls"].shift(1).fillna(0).astype(int)

    pitches = (
        _pitches_raw
        .assign(
            in_zone=lambda d: d["zone"].isin(IN_ZONE).fillna(False),
            out_zone=lambda d: d["zone"].isin(OUT_ZONE).fillna(False),
            is_swing=lambda d: d["pitch_call_code"].isin(SWING_CODES),
            is_hard_hit=lambda d: d["launch_speed"].ge(95).fillna(False),
            is_sweet_spot=lambda d: d["launch_angle"].between(8, 32).fillna(False),
            is_two_strike_swing=lambda d: (
                d["pre_count_strikes"].eq(2) & d["pitch_call_code"].isin(SWING_CODES)
            ),
            is_two_strike_miss=lambda d: (
                d["pre_count_strikes"].eq(2) & d["pitch_call_code"].isin(["S", "W", "T"])
            ),
            pitcher_ahead=lambda d: d["pre_count_strikes"].gt(d["pre_count_balls"]),
            pitcher_behind=lambda d: d["pre_count_balls"].gt(d["pre_count_strikes"]),
            is_edge_zone=lambda d: d["zone"].isin([1, 2, 3, 4, 6, 7, 8, 9]).fillna(False),
            is_called_strike=lambda d: d["pitch_call_code"].eq("C"),
        )
        .assign(
            chase_swing=lambda d: d["out_zone"] & d["is_swing"],
            inplay_hard=lambda d: d["is_in_play"] & d["is_hard_hit"],
            contact_hard=lambda d: d["is_in_play"] & d["is_hard_hit"],
            contact_sweet=lambda d: d["is_in_play"] & d["is_sweet_spot"],
            ahead_swing=lambda d: d["pitcher_ahead"] & d["is_swing"],
            ahead_miss=lambda d: d["pitcher_ahead"] & d["pitch_call_code"].isin(["S", "W", "T"]),
            behind_swing=lambda d: d["pitcher_behind"] & d["is_swing"],
            behind_miss=lambda d: d["pitcher_behind"] & d["pitch_call_code"].isin(["S", "W", "T"]),
            edge_called_strike=lambda d: d["is_edge_zone"] & d["is_called_strike"],
        )
    )

    # Per-game stuff + command aggregate (all pitches)
    aggs = (
        pitches
        .groupby(["gamepk", "pitcher_id"], sort=False)
        .agg(
            avg_velo=("start_speed", "mean"),
            avg_spin_rate=("spin_rate", "mean"),
            avg_extension=("release_pos_y", "mean"),
            avg_h_break=("break_horizontal", "mean"),
            avg_v_break=("break_vertical_induced", "mean"),
            in_zone_n=("in_zone", "sum"),
            total_pitches_n=("is_pitch", "count"),
            out_zone_n=("out_zone", "sum"),
            chase_swing_n=("chase_swing", "sum"),
            two_strike_swing_n=("is_two_strike_swing", "sum"),
            two_strike_miss_n=("is_two_strike_miss", "sum"),
            inplay_n=("is_in_play", "sum"),
            inplay_hard_n=("inplay_hard", "sum"),
            contact_n=("is_in_play", "sum"),
            contact_hard_n=("contact_hard", "sum"),
            contact_sweet_n=("contact_sweet", "sum"),
            ahead_swing_n=("ahead_swing", "sum"),
            ahead_miss_n=("ahead_miss", "sum"),
            behind_swing_n=("behind_swing", "sum"),
            behind_miss_n=("behind_miss", "sum"),
            edge_zone_n=("is_edge_zone", "sum"),
            edge_called_strike_n=("edge_called_strike", "sum"),
        )
        .reset_index()
    )

    # Conditional mean exit velocity — in-play only (contact == in-play here)
    inplay_velo = (
        pitches[pitches["is_in_play"]]
        .groupby(["gamepk", "pitcher_id"], sort=False)["launch_speed"]
        .mean()
        .rename("inplay_exit_velo")
        .reset_index()
    )

    return (
        aggs
        .merge(inplay_velo, on=["gamepk", "pitcher_id"], how="left")
        .assign(
            contact_exit_velo=lambda d: d["inplay_exit_velo"],
            zone_pct=lambda d: d["in_zone_n"] / d["total_pitches_n"].clip(lower=1),
            chase_induced_pct=lambda d: d["chase_swing_n"] / d["out_zone_n"].clip(lower=1),
            two_strike_whiff_pct=lambda d: (
                d["two_strike_miss_n"] / d["two_strike_swing_n"].clip(lower=1)
            ),
            inplay_hard_hit_pct=lambda d: d["inplay_hard_n"] / d["inplay_n"].clip(lower=1),
            contact_hard_hit_pct=lambda d: d["contact_hard_n"] / d["contact_n"].clip(lower=1),
            contact_sweet_spot_pct=lambda d: d["contact_sweet_n"] / d["contact_n"].clip(lower=1),
            contact_count=lambda d: d["contact_n"].astype(float),
            whiff_rate_ahead=lambda d: (
                d["ahead_miss_n"] / d["ahead_swing_n"].clip(lower=1)
            ),
            whiff_rate_behind=lambda d: (
                d["behind_miss_n"] / d["behind_swing_n"].clip(lower=1)
            ),
            called_strike_edge_pct=lambda d: (
                d["edge_called_strike_n"] / d["edge_zone_n"].clip(lower=1)
            ),
        )
        # Minimum threshold: < 2 contact events → null all contact quality metrics
        .assign(
            inplay_exit_velo=lambda d: d["inplay_exit_velo"].where(d["contact_count"] >= 2),
            inplay_hard_hit_pct=lambda d: d["inplay_hard_hit_pct"].where(d["contact_count"] >= 2),
            contact_exit_velo=lambda d: d["contact_exit_velo"].where(d["contact_count"] >= 2),
            contact_hard_hit_pct=lambda d: d["contact_hard_hit_pct"].where(d["contact_count"] >= 2),
            contact_sweet_spot_pct=lambda d: d["contact_sweet_spot_pct"].where(d["contact_count"] >= 2),
        )
        [["gamepk", "pitcher_id",
          "avg_velo", "avg_spin_rate", "avg_extension", "avg_h_break", "avg_v_break",
          "inplay_exit_velo", "inplay_hard_hit_pct",
          "contact_exit_velo", "contact_hard_hit_pct", "contact_sweet_spot_pct", "contact_count",
          "zone_pct", "chase_induced_pct", "two_strike_whiff_pct",
          "whiff_rate_ahead", "whiff_rate_behind", "called_strike_edge_pct"]]
    )


# ═════════════════════════════════════════════════════════════════════════════
# Per-game stat tables (boxscore + PBP combined)
# ═════════════════════════════════════════════════════════════════════════════

def _build_pitcher_game(
    pitcher_box: pd.DataFrame,
    pbp_aggs: pd.DataFrame,
    starters: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (personId, gamepk) for starters only.
    ip is already in true decimal per prepared schema.
    Rate stat column names use the SP_STAT_COLS convention (no _gm suffix).
    """
    sp_box = pitcher_box.merge(
        starters.rename(columns={"starter_id": "personId"})[["gamepk", "personId"]],
        on=["gamepk", "personId"],
        how="inner",
    )
    sp_box = sp_box.merge(
        pbp_aggs.rename(columns={"pitcher_id": "personId"}),
        on=["gamepk", "personId"],
        how="left",
    )
    safe_pa = sp_box["pa_count"].clip(lower=1)
    return sp_box.assign(
        k_rate    =lambda d: d["k"]  / safe_pa,
        bb_rate   =lambda d: d["bb"] / safe_pa,
        hr_rate   =lambda d: d["hr"] / safe_pa,
        k_bb_ratio=lambda d: d["k"]  / d["bb"].replace(0, np.nan),
        game_date =lambda d: pd.to_datetime(d["game_date"]),
    )


def _build_batter_game(
    batter_box: pd.DataFrame,
    pbp_aggs: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (personId, gamepk).
    All rates divided by plate_appearances (not ab), per spec.
    """
    safe_pa = batter_box["plate_appearances"].clip(lower=1)
    return (
        batter_box
        .assign(
            k_rate   =lambda d: d["k"]  / safe_pa,
            bb_rate  =lambda d: d["bb"] / safe_pa,
            hr_rate  =lambda d: d["hr"] / safe_pa,
            iso      =lambda d: (d["doubles"] + 2 * d["triples"] + 3 * d["hr"]) / safe_pa,
            tb_rate  =lambda d: d["total_bases"] / safe_pa,
            game_date=lambda d: pd.to_datetime(d["game_date"]),
        )
        .merge(
            pbp_aggs.rename(columns={"batter_id": "personId"}),
            on=["gamepk", "personId"],
            how="left",
        )
    )


# ═════════════════════════════════════════════════════════════════════════════
# Team-level offensive context features
# ═════════════════════════════════════════════════════════════════════════════

def _build_team_game(
    batter_box: pd.DataFrame,
    pbp_bat_aggs: pd.DataFrame,
    sp_pa: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (team_id, gamepk, game_date, game_season).
    Aggregates all batters on the team for that game.
    Rates computed from team-level sums — not averages of individual rates.
    """
    # ── Detect team column in batter_box ─────────────────────────────────────
    _team_col = next(
        (c for c in batter_box.columns if c.lower() in ("team_id", "teamid", "team")),
        None,
    )
    if _team_col is None:
        raise ValueError(
            f"Cannot find team column in batter_box. Columns: {batter_box.columns.tolist()}"
        )
    if _team_col != "team_id":
        batter_box = batter_box.rename(columns={_team_col: "team_id"})

    # ── Batter boxscore → team level ─────────────────────────────────────────
    _has_hits = "hits" in batter_box.columns
    _has_hbp  = "hbp"  in batter_box.columns
    if not _has_hits or not _has_hbp:
        print(f"batter_box available columns: {batter_box.columns.tolist()}")
        print(f"hits present: {_has_hits}, hbp present: {_has_hbp}")

    _agg_dict: dict = dict(
        team_pa         =("plate_appearances", "sum"),
        team_k          =("k",                 "sum"),
        team_bb         =("bb",                "sum"),
        team_hr         =("hr",                "sum"),
        team_total_bases=("total_bases",       "sum"),
        team_doubles    =("doubles",           "sum"),
        team_triples    =("triples",           "sum"),
    )
    if _has_hits:
        _agg_dict["team_hits"] = ("hits", "sum")
    if _has_hbp:
        _agg_dict["team_hbp"]  = ("hbp",  "sum")

    team_box = (
        batter_box
        .groupby(["team_id", "gamepk", "game_date", "game_season"], sort=False)
        .agg(**_agg_dict)
        .reset_index()
        .assign(
            team_k_rate  =lambda d: d["team_k"]  / d["team_pa"].clip(lower=1),
            team_bb_rate =lambda d: d["team_bb"] / d["team_pa"].clip(lower=1),
            team_hr_rate =lambda d: d["team_hr"] / d["team_pa"].clip(lower=1),
            team_iso     =lambda d: (d["team_doubles"] + 2*d["team_triples"] + 3*d["team_hr"])
                                    / d["team_pa"].clip(lower=1),
            team_tb_rate =lambda d: d["team_total_bases"] / d["team_pa"].clip(lower=1),
        )
    )

    if _has_hits and _has_hbp:
        team_box = team_box.assign(
            team_obp=lambda d: (d["team_hits"] + d["team_bb"] + d["team_hbp"].fillna(0))
                               / d["team_pa"].clip(lower=1),
        )
    elif _has_hits:
        team_box = team_box.assign(
            team_obp=lambda d: (d["team_hits"] + d["team_bb"]) / d["team_pa"].clip(lower=1),
        )
    else:
        # Proxy: use tb_rate as best available stand-in
        team_box = team_box.assign(team_obp=lambda d: d["team_tb_rate"])

    # ── PBP batter aggs → team level ─────────────────────────────────────────
    team_pbp = (
        pbp_bat_aggs
        .merge(
            sp_pa[["gamepk", "batter_id", "batter_team_id"]].drop_duplicates(),
            on=["gamepk", "batter_id"],
            how="left",
        )
        .groupby(["batter_team_id", "gamepk"], sort=False)
        .agg(
            team_chase_n     =("chase_n",      "sum"),
            team_out_zone_n  =("out_zone_n",   "sum"),
            team_z_swing_n   =("z_swing_n",    "sum"),
            team_z_contact_n =("z_contact_n",  "sum"),
            team_pitches_seen=("pitches_seen", "sum"),
        )
        .reset_index()
        .rename(columns={"batter_team_id": "team_id"})
        .assign(
            team_chase_rate   =lambda d: d["team_chase_n"]    / d["team_out_zone_n"].clip(lower=1),
            team_z_contact_pct=lambda d: d["team_z_contact_n"]/ d["team_z_swing_n"].clip(lower=1),
        )
    )

    # ── Merge boxscore + pbp ──────────────────────────────────────────────────
    return (
        team_box
        .merge(team_pbp, on=["team_id", "gamepk"], how="left")
        .assign(
            team_p_per_pa=lambda d: d["team_pitches_seen"] / d["team_pa"].clip(lower=1),
        )
    )


def _team_rolling(team_game: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly-lagged rolling team offensive features.
    Windows: last game (w1), 10 games (w10), season YTD (wytd).
    Partition by team_id, order by game_date.
    """
    team_game = team_game.sort_values(["team_id", "game_date"])
    con = duckdb.connect()
    con.register("team_game", team_game)

    return con.execute("""
        SELECT
            team_id, gamepk, game_date, game_season,

            -- last game (1 PRECEDING TO 1 PRECEDING)
            AVG(team_k_rate)        OVER w1 AS team_last_game_k_rate,
            AVG(team_bb_rate)       OVER w1 AS team_last_game_bb_rate,
            AVG(team_hr_rate)       OVER w1 AS team_last_game_hr_rate,
            AVG(team_obp)           OVER w1 AS team_last_game_obp,
            AVG(team_iso)           OVER w1 AS team_last_game_iso,
            AVG(team_tb_rate)       OVER w1 AS team_last_game_tb_rate,
            AVG(team_chase_rate)    OVER w1 AS team_last_game_chase_rate,
            AVG(team_z_contact_pct) OVER w1 AS team_last_game_z_contact_pct,
            AVG(team_p_per_pa)      OVER w1 AS team_last_game_p_per_pa,

            -- 10-game window
            AVG(team_k_rate)        OVER w10 AS team_10g_k_rate,
            AVG(team_bb_rate)       OVER w10 AS team_10g_bb_rate,
            AVG(team_hr_rate)       OVER w10 AS team_10g_hr_rate,
            AVG(team_obp)           OVER w10 AS team_10g_obp,
            AVG(team_iso)           OVER w10 AS team_10g_iso,
            AVG(team_tb_rate)       OVER w10 AS team_10g_tb_rate,
            AVG(team_chase_rate)    OVER w10 AS team_10g_chase_rate,
            AVG(team_z_contact_pct) OVER w10 AS team_10g_z_contact_pct,
            AVG(team_p_per_pa)      OVER w10 AS team_10g_p_per_pa,
            COUNT(*)                OVER w10 AS team_10g_n_games,

            -- season YTD
            AVG(team_k_rate)        OVER wytd AS team_ytd_k_rate,
            AVG(team_bb_rate)       OVER wytd AS team_ytd_bb_rate,
            AVG(team_hr_rate)       OVER wytd AS team_ytd_hr_rate,
            AVG(team_obp)           OVER wytd AS team_ytd_obp,
            AVG(team_iso)           OVER wytd AS team_ytd_iso,
            AVG(team_tb_rate)       OVER wytd AS team_ytd_tb_rate,
            AVG(team_chase_rate)    OVER wytd AS team_ytd_chase_rate,
            AVG(team_z_contact_pct) OVER wytd AS team_ytd_z_contact_pct,
            AVG(team_p_per_pa)      OVER wytd AS team_ytd_p_per_pa,
            SUM(team_pa)            OVER wytd AS team_ytd_pa_count,
            COUNT(*)                OVER wytd AS team_ytd_n_games

        FROM team_game
        WINDOW
            w1   AS (PARTITION BY team_id
                     ORDER BY game_date
                     ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING),
            w10  AS (PARTITION BY team_id
                     ORDER BY game_date
                     ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
            wytd AS (PARTITION BY team_id, game_season
                     ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ORDER BY team_id, game_date
    """).df()


# ═════════════════════════════════════════════════════════════════════════════
# Rolling features via DuckDB window functions
# ═════════════════════════════════════════════════════════════════════════════

def _pitcher_rolling(pitcher_game: pd.DataFrame, inn45_aggs: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Strictly-lagged rolling pitcher features.
    Windows: 3 starts, 10 starts, season YTD (games-based, no leakage).
    Days-based: days_rest computed in pandas; cumulative_ip_last_7d via rolling window.
    Also computes last-start features (w1), workload/endurance features, and
    late-inning (inn 4-5) efficiency features when inn45_aggs is provided.
    """
    pitcher_game = pitcher_game.sort_values(["personId", "game_date"]).assign(
        days_rest=lambda d: (
            d.groupby("personId")["game_date"]
            .transform(lambda s: s.diff().dt.days - 1)
        )
    )

    def _ip_last7(g: pd.DataFrame) -> pd.Series:
        return (
            g.set_index("game_date")["ip"]
            .sort_index()
            .shift(1)                         # exclude current game
            .rolling("7D", min_periods=0)
            .sum()
            .reset_index(drop=True)
        )

    pitcher_game = pitcher_game.assign(
        cumulative_ip_last_7d=(
            pitcher_game
            .groupby("personId", group_keys=False)
            .apply(_ip_last7)
            .values
        )
    )

    # ── Inn 4-5 per-game rates (NaN when pitcher didn't reach those innings) ──
    if inn45_aggs is not None:
        pitcher_game = (
            pitcher_game
            .merge(
                inn45_aggs.rename(columns={"pitcher_id": "personId"}),
                on=["gamepk", "personId"],
                how="left",
            )
            .assign(
                inn45_p_per_pa_rate=lambda d: np.where(
                    d["inn45_pa"].fillna(0) > 0,
                    d["inn45_pitches"] / d["inn45_pa"],
                    np.nan,
                ),
                inn45_obp_rate=lambda d: np.where(
                    d["inn45_pa"].fillna(0) > 0,
                    d["inn45_onbase"] / d["inn45_pa"],
                    np.nan,
                ),
            )
        )
    else:
        pitcher_game = pitcher_game.assign(
            inn45_p_per_pa_rate=np.nan,
            inn45_obp_rate=np.nan,
        )

    con = duckdb.connect()
    con.register("pg", pitcher_game)

    return con.execute("""
        SELECT
            personId, gamepk, game_date, game_season, team_id,
            days_rest, cumulative_ip_last_7d,

            -- last start (single-row lookback: 1 PRECEDING)
            AVG(k_rate)                 OVER w1 AS sp_last_start_k_rate,
            AVG(bb_rate)                OVER w1 AS sp_last_start_bb_rate,
            AVG(hr_rate)                OVER w1 AS sp_last_start_hr_rate,
            AVG(gb_pct)                 OVER w1 AS sp_last_start_gb_pct,
            AVG(csw_pct)                OVER w1 AS sp_last_start_csw_pct,
            AVG(whiff_rate)             OVER w1 AS sp_last_start_whiff_rate,
            AVG(barrel_pct)             OVER w1 AS sp_last_start_barrel_pct,
            AVG(hard_hit_pct)           OVER w1 AS sp_last_start_hard_hit_pct,
            AVG(first_pitch_strike_pct) OVER w1 AS sp_last_start_first_pitch_strike_pct,
            AVG(p_per_pa)               OVER w1 AS sp_last_start_p_per_pa,
            AVG(ip)                     OVER w1 AS sp_last_start_ip,
            AVG(total_pitches)          OVER w1 AS sp_last_start_pitches,

            -- 10-start window
            AVG(k_rate)             OVER w10 AS sp10_k_rate,
            AVG(bb_rate)            OVER w10 AS sp10_bb_rate,
            AVG(hr_rate)            OVER w10 AS sp10_hr_rate,
            AVG(k_bb_ratio)         OVER w10 AS sp10_k_bb,
            AVG(gb_pct)             OVER w10 AS sp10_gb_pct,
            AVG(csw_pct)            OVER w10 AS sp10_csw_pct,
            AVG(whiff_rate)         OVER w10 AS sp10_whiff_rate,
            AVG(barrel_pct)         OVER w10 AS sp10_barrel_pct,
            AVG(hard_hit_pct)       OVER w10 AS sp10_hard_hit_pct,
            AVG(first_pitch_strike_pct) OVER w10 AS sp10_fps_pct,
            AVG(p_per_pa)           OVER w10 AS sp10_p_per_pa,
            AVG(ip)                 OVER w10 AS sp10_avg_ip,
            AVG(total_pitches)      OVER w10 AS sp10_avg_pitches,
            SUM(total_pitches)      OVER w10 AS sp10_total_pitches,
            AVG(pa_count)           OVER w10 AS sp10_avg_batters_faced,
            SUM(pa_count)           OVER w10 AS sp10_pa_count,
            COUNT(*)                OVER w10 AS sp10_n_starts,
            AVG(inn45_p_per_pa_rate) OVER w10 AS sp10_p_per_pa_inn4_5,
            AVG(inn45_obp_rate)      OVER w10 AS sp10_obp_allowed_inn4_5,

            -- season YTD (resets each season)
            AVG(k_rate)             OVER wytd AS spytd_k_rate,
            AVG(bb_rate)            OVER wytd AS spytd_bb_rate,
            AVG(hr_rate)            OVER wytd AS spytd_hr_rate,
            AVG(k_bb_ratio)         OVER wytd AS spytd_k_bb,
            AVG(gb_pct)             OVER wytd AS spytd_gb_pct,
            AVG(csw_pct)            OVER wytd AS spytd_csw_pct,
            AVG(whiff_rate)         OVER wytd AS spytd_whiff_rate,
            AVG(barrel_pct)         OVER wytd AS spytd_barrel_pct,
            AVG(hard_hit_pct)       OVER wytd AS spytd_hard_hit_pct,
            AVG(first_pitch_strike_pct) OVER wytd AS spytd_fps_pct,
            AVG(p_per_pa)           OVER wytd AS spytd_p_per_pa,
            AVG(ip)                 OVER wytd AS spytd_avg_ip,
            AVG(total_pitches)      OVER wytd AS spytd_avg_pitches,
            AVG(pa_count)           OVER wytd AS spytd_avg_batters_faced,
            SUM(pa_count)           OVER wytd AS spytd_pa_count,
            COUNT(*)                OVER wytd AS spytd_n_starts,

            -- last start process quality
            AVG(avg_velo)               OVER w1 AS sp_last_start_avg_velo,
            AVG(avg_spin_rate)          OVER w1 AS sp_last_start_avg_spin_rate,
            AVG(avg_extension)          OVER w1 AS sp_last_start_avg_extension,
            AVG(avg_h_break)            OVER w1 AS sp_last_start_avg_h_break,
            AVG(avg_v_break)            OVER w1 AS sp_last_start_avg_v_break,
            AVG(contact_exit_velo)      OVER w1 AS sp_last_start_contact_exit_velo,
            AVG(inplay_exit_velo)       OVER w1 AS sp_last_start_inplay_exit_velo,
            AVG(contact_hard_hit_pct)   OVER w1 AS sp_last_start_hard_contact_pct,
            AVG(contact_sweet_spot_pct) OVER w1 AS sp_last_start_sweet_spot_pct,
            AVG(zone_pct)               OVER w1 AS sp_last_start_zone_pct,
            AVG(chase_induced_pct)      OVER w1 AS sp_last_start_chase_induced_pct,
            AVG(two_strike_whiff_pct)   OVER w1 AS sp_last_start_two_strike_whiff_pct,
            AVG(whiff_rate_ahead)       OVER w1 AS sp_last_start_whiff_rate_ahead,
            AVG(whiff_rate_behind)      OVER w1 AS sp_last_start_whiff_rate_behind,
            AVG(called_strike_edge_pct) OVER w1 AS sp_last_start_called_strike_edge_pct,

            -- 10-start process quality
            AVG(avg_velo)               OVER w10 AS sp10_avg_velo,
            AVG(avg_spin_rate)          OVER w10 AS sp10_avg_spin_rate,
            AVG(avg_extension)          OVER w10 AS sp10_avg_extension,
            AVG(contact_exit_velo)      OVER w10 AS sp10_contact_exit_velo,
            AVG(contact_hard_hit_pct)   OVER w10 AS sp10_hard_contact_pct,
            AVG(contact_sweet_spot_pct) OVER w10 AS sp10_sweet_spot_pct,
            AVG(zone_pct)               OVER w10 AS sp10_zone_pct,
            AVG(chase_induced_pct)      OVER w10 AS sp10_chase_induced_pct,
            AVG(whiff_rate_ahead)       OVER w10 AS sp10_whiff_rate_ahead,
            AVG(whiff_rate_behind)      OVER w10 AS sp10_whiff_rate_behind,
            AVG(called_strike_edge_pct) OVER w10 AS sp10_called_strike_edge_pct,

            -- YTD process quality
            AVG(avg_velo)               OVER wytd AS spytd_avg_velo,
            AVG(avg_spin_rate)          OVER wytd AS spytd_avg_spin_rate,
            AVG(avg_extension)          OVER wytd AS spytd_avg_extension,
            AVG(contact_exit_velo)      OVER wytd AS spytd_contact_exit_velo,
            AVG(contact_hard_hit_pct)   OVER wytd AS spytd_hard_contact_pct,
            AVG(contact_sweet_spot_pct) OVER wytd AS spytd_sweet_spot_pct,
            AVG(zone_pct)               OVER wytd AS spytd_zone_pct,
            AVG(chase_induced_pct)      OVER wytd AS spytd_chase_induced_pct,
            AVG(whiff_rate_ahead)       OVER wytd AS spytd_whiff_rate_ahead,
            AVG(whiff_rate_behind)      OVER wytd AS spytd_whiff_rate_behind,
            AVG(called_strike_edge_pct) OVER wytd AS spytd_called_strike_edge_pct,

            -- spacing / fatigue inputs
            DATEDIFF('day',
                LAG(game_date, 3)  OVER wlag,
                LAG(game_date, 1)  OVER wlag) / 2.0
                AS sp_avg_days_between_starts_3,
            DATEDIFF('day',
                LAG(game_date, 10) OVER wlag,
                LAG(game_date, 1)  OVER wlag) / 9.0
                AS sp_avg_days_between_starts_10,

            -- raw game stats passed through (needed for career-avg computation)
            k_rate, bb_rate, hr_rate, gb_pct, csw_pct, whiff_rate,
            barrel_pct, hard_hit_pct, first_pitch_strike_pct, p_per_pa,
            pa_count, ip

        FROM pg
        WINDOW
            w1   AS (PARTITION BY personId
                     ORDER BY game_date
                     ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING),
            w10  AS (PARTITION BY personId
                     ORDER BY game_date
                     ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
            wytd AS (PARTITION BY personId, game_season
                     ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
            wlag AS (PARTITION BY personId
                     ORDER BY game_date)
        ORDER BY personId, game_date
    """).df().assign(
        fatigue_score=lambda d: d["sp_last_start_pitches"] / d["days_rest"].clip(lower=1),
        sp_velo_trend=lambda d: d["sp_last_start_avg_velo"] - d["sp10_avg_velo"],
        sp_spin_trend=lambda d: d["sp_last_start_avg_spin_rate"] - d["sp10_avg_spin_rate"],
        sp_contact_velo_trend=lambda d: d["sp_last_start_contact_exit_velo"] - d["sp10_contact_exit_velo"],
    )


def _batter_rolling(batter_game: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly-lagged rolling batter features.
    Windows: 3, 15, 50 games, season YTD (games-based, no leakage).
    """
    batter_game = batter_game.sort_values(["personId", "game_date"]).assign(
        bat_days_since_last_game=lambda d: (
            d.groupby("personId")["game_date"]
            .transform(lambda s: s.diff().dt.days)
        )
    )

    con = duckdb.connect()
    con.register("bg", batter_game)

    return con.execute("""
        SELECT
            personId, gamepk, game_date, game_season,
            CAST(batting_order AS INTEGER) AS batting_order_int,
            bat_days_since_last_game,

            -- last game (1 PRECEDING TO 1 PRECEDING)
            AVG(k_rate)        OVER w1 AS bat_last_game_k_rate,
            AVG(bb_rate)       OVER w1 AS bat_last_game_bb_rate,
            AVG(hr_rate)       OVER w1 AS bat_last_game_hr_rate,
            AVG(iso)           OVER w1 AS bat_last_game_iso,
            AVG(tb_rate)       OVER w1 AS bat_last_game_tb_rate,
            AVG(chase_rate)    OVER w1 AS bat_last_game_chase_rate,
            AVG(z_contact_pct) OVER w1 AS bat_last_game_z_contact,
            AVG(o_contact_pct) OVER w1 AS bat_last_game_o_contact,

            -- 15-game window
            AVG(k_rate)      OVER w15 AS bat15_k_rate,
            AVG(bb_rate)     OVER w15 AS bat15_bb_rate,
            AVG(hr_rate)     OVER w15 AS bat15_hr_rate,
            AVG(iso)         OVER w15 AS bat15_iso,
            AVG(tb_rate)     OVER w15 AS bat15_tb_rate,
            AVG(chase_rate)  OVER w15 AS bat15_chase_rate,
            AVG(z_contact_pct) OVER w15 AS bat15_z_contact,
            AVG(o_contact_pct) OVER w15 AS bat15_o_contact,
            SUM(plate_appearances) OVER w15 AS bat15_pa_count,
            COUNT(*)         OVER w15 AS bat15_n_games,

            -- 50-game window
            AVG(k_rate)      OVER w50 AS bat50_k_rate,
            AVG(bb_rate)     OVER w50 AS bat50_bb_rate,
            AVG(hr_rate)     OVER w50 AS bat50_hr_rate,
            AVG(iso)         OVER w50 AS bat50_iso,
            AVG(tb_rate)     OVER w50 AS bat50_tb_rate,
            AVG(chase_rate)  OVER w50 AS bat50_chase_rate,
            AVG(z_contact_pct) OVER w50 AS bat50_z_contact,
            AVG(o_contact_pct) OVER w50 AS bat50_o_contact,
            SUM(plate_appearances) OVER w50 AS bat50_pa_count,
            COUNT(*)         OVER w50 AS bat50_n_games,

            -- season YTD (resets each season)
            AVG(k_rate)      OVER wytd AS batytd_k_rate,
            AVG(bb_rate)     OVER wytd AS batytd_bb_rate,
            AVG(hr_rate)     OVER wytd AS batytd_hr_rate,
            AVG(iso)         OVER wytd AS batytd_iso,
            AVG(tb_rate)     OVER wytd AS batytd_tb_rate,
            AVG(chase_rate)  OVER wytd AS batytd_chase_rate,
            AVG(z_contact_pct) OVER wytd AS batytd_z_contact,
            AVG(o_contact_pct) OVER wytd AS batytd_o_contact,
            SUM(plate_appearances) OVER wytd AS batytd_pa_count,
            COUNT(*)         OVER wytd AS batytd_n_games,

            -- raw game stats passed through (needed for career-avg computation)
            k_rate, bb_rate, hr_rate, iso, tb_rate,
            chase_rate, z_contact_pct, o_contact_pct, plate_appearances

        FROM bg
        WINDOW
            w1   AS (PARTITION BY personId
                     ORDER BY game_date
                     ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING),
            w15  AS (PARTITION BY personId
                     ORDER BY game_date
                     ROWS BETWEEN 15 PRECEDING AND 1 PRECEDING),
            w50  AS (PARTITION BY personId
                     ORDER BY game_date
                     ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING),
            wytd AS (PARTITION BY personId, game_season
                     ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ORDER BY personId, game_date
    """).df()


# ═════════════════════════════════════════════════════════════════════════════
# Weighted career-average fallback (early-season: YTD PA < 30)
# ═════════════════════════════════════════════════════════════════════════════

def _add_career_avg_fallback(
    rolling_df: pd.DataFrame,
    raw_game_df: pd.DataFrame,
    stat_cols: list[str],
    ytd_pa_col: str,
    ytd_prefix: str,
    out_prefix: str,
    pa_threshold: int = 30,
) -> pd.DataFrame:
    """
    Computes weighted season averages (3×curr + 2×prior + 1×two_yrs_ago) / 6
    and replaces the YTD rolling feature wherever ytd_pa < pa_threshold.

    stat_cols     : raw column names in raw_game_df (e.g. "k_rate")
    ytd_pa_col    : column in rolling_df holding YTD PA count (e.g. "spytd_pa_count")
    ytd_prefix    : prefix of YTD columns in rolling_df (e.g. "spytd_")
    out_prefix    : career avg column prefix (e.g. "sp_")
    """
    season_avgs = (
        raw_game_df
        .groupby(["personId", "game_season"])[stat_cols]
        .mean()
        .reset_index()
    )
    # s0 = current full season, s1 = prior, s2 = two years ago
    # We shift the join season key so s1 joins to "current season + 1", etc.
    s1 = season_avgs.assign(game_season=lambda d: d["game_season"] + 1).rename(
        columns={c: f"s1_{c}" for c in stat_cols}
    )
    s2 = season_avgs.assign(game_season=lambda d: d["game_season"] + 2).rename(
        columns={c: f"s2_{c}" for c in stat_cols}
    )
    career = (
        season_avgs.rename(columns={c: f"s0_{c}" for c in stat_cols})
        .merge(s1, on=["personId", "game_season"], how="left")
        .merge(s2, on=["personId", "game_season"], how="left")
    )

    career_cols = []
    for c in stat_cols:
        v0, v1, v2 = career[f"s0_{c}"], career[f"s1_{c}"], career[f"s2_{c}"]
        num = v0.fillna(0) * 3 + v1.fillna(0) * 2 + v2.fillna(0) * 1
        den = v0.notna().astype(int) * 3 + v1.notna().astype(int) * 2 + v2.notna().astype(int) * 1
        career[f"{out_prefix}3yr_wtd_{c}"] = np.where(den > 0, num / den, np.nan)
        career_cols.append(f"{out_prefix}3yr_wtd_{c}")

    rolling_df = rolling_df.merge(
        career[["personId", "game_season"] + career_cols],
        on=["personId", "game_season"],
        how="left",
    )

    # Replace YTD with career avg where YTD sample < threshold
    low_sample = rolling_df[ytd_pa_col].fillna(0) < pa_threshold
    for c in stat_cols:
        ytd_col    = f"{ytd_prefix}{c}"
        career_col = f"{out_prefix}3yr_wtd_{c}"
        if ytd_col in rolling_df.columns and career_col in rolling_df.columns:
            rolling_df[ytd_col] = np.where(
                low_sample, rolling_df[career_col], rolling_df[ytd_col]
            )

    return rolling_df


# ═════════════════════════════════════════════════════════════════════════════
# Last-season average features (standalone, not a fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _add_last_season_avgs(
    rolling_df: pd.DataFrame,
    raw_game_df: pd.DataFrame,
    stat_cols: list[str],
    out_prefix: str,
) -> pd.DataFrame:
    """
    Computes each player's simple per-season mean for stat_cols, then joins
    last season's averages onto rolling_df (game_season - 1 → current season).
    Columns are named {out_prefix}last_szn_{stat}.
    """
    last_szn = (
        raw_game_df
        .groupby(["personId", "game_season"])[stat_cols]
        .mean()
        .reset_index()
        .assign(game_season=lambda d: d["game_season"] + 1)
        .rename(columns={c: f"{out_prefix}last_szn_{c}" for c in stat_cols})
    )
    return rolling_df.merge(last_szn, on=["personId", "game_season"], how="left")


# ═════════════════════════════════════════════════════════════════════════════
# Log5 interaction features
# ═════════════════════════════════════════════════════════════════════════════

def _log5(b: pd.Series, p: pd.Series, lg: pd.Series) -> pd.Series:
    """
    Bill James Log5 expected rate.
    b = batter rate, p = pitcher allowed rate, lg = league average rate.
    """
    num   = b * p / lg
    denom = num + (1 - b) * (1 - p) / (1 - lg)
    return np.where(denom > 0, num / denom, np.nan)


def _add_log5(features: pd.DataFrame, pitcher_game: pd.DataFrame) -> pd.DataFrame:
    """
    Compute league rates from prior season data, then apply log5 for K, BB, HR.
    Uses spytd_* and batytd_* rates (already have career fallback applied).
    """
    # Full-season league averages per season
    lg = (
        pitcher_game
        .groupby("game_season")
        .agg(lg_k_rate=("k_rate", "mean"),
             lg_bb_rate=("bb_rate", "mean"),
             lg_hr_rate=("hr_rate", "mean"))
        .reset_index()
        # Shift +1 so each target season gets the PRIOR year's league rate
        .assign(game_season=lambda d: d["game_season"] + 1)
    )

    features = features.merge(lg, on="game_season", how="left")

    # Fill league defaults if no prior-year data exists (e.g. first year in dataset)
    lg_k  = features["lg_k_rate"].fillna(0.220)
    lg_bb = features["lg_bb_rate"].fillna(0.085)
    lg_hr = features["lg_hr_rate"].fillna(0.034)

    return features.assign(
        log5_expected_k =lambda d: _log5(d["batytd_k_rate"],  d["spytd_k_rate"],  lg_k),
        log5_expected_bb=lambda d: _log5(d["batytd_bb_rate"], d["spytd_bb_rate"], lg_bb),
        log5_expected_hr=lambda d: _log5(d["batytd_hr_rate"], d["spytd_hr_rate"], lg_hr),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Team OAA (optional defensive context)
# ═════════════════════════════════════════════════════════════════════════════

def _load_team_oaa(years: list[int]) -> pd.DataFrame:
    """
    Season-level team OAA from pybaseball (static per team-season).
    Returns empty DataFrame if pybaseball is unavailable.
    """
    try:
        from pybaseball import statcast_outs_above_average
        frames = []
        for yr in years:
            oaa = statcast_outs_above_average(yr, pos_list="all")
            if oaa is not None and len(oaa):
                frames.append(oaa.assign(game_season=yr))
        if not frames:
            raise ValueError("no OAA rows returned")
        return (
            pd.concat(frames, ignore_index=True)
            .groupby(["team_id", "game_season"])["outs_above_average"]
            .sum()
            .reset_index()
            .rename(columns={"outs_above_average": "team_oaa_season"})
        )
    except Exception as exc:
        logger.warning(f"Team OAA unavailable ({exc}); team_oaa_season will be NaN")
        return pd.DataFrame(columns=["team_id", "game_season", "team_oaa_season"])


# ═════════════════════════════════════════════════════════════════════════════
# Platoon split game aggregates (Bin 2 — pitchers, Bin 3 — batters)
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_handedness(df: pd.DataFrame, player_info: pd.DataFrame, id_dtype) -> pd.DataFrame:
    """
    Join batSide + pitchHand from player_info and return resolved columns.
    Switch pitchers → R; switch batters → opposite of resolved pitcher hand.
    """
    assert "personId" in player_info.columns, \
        "player_info missing personId — normalize at load time in build_pa_features"

    sides = (
        player_info[["personId", "batSide", "pitchHand"]]
        .drop_duplicates("personId")
        .assign(personId=lambda d: d["personId"].astype(id_dtype))
        .astype({"batSide": object, "pitchHand": object})
    )
    return (
        df
        .merge(
            sides[["personId", "batSide"]].rename(
                columns={"personId": "batter_id", "batSide": "bat_side_raw"}
            ),
            on="batter_id", how="left",
        )
        .merge(
            sides[["personId", "pitchHand"]].rename(
                columns={"personId": "pitcher_id", "pitchHand": "pitch_hand_raw"}
            ),
            on="pitcher_id", how="left",
        )
        .assign(
            pitch_hand_resolved=lambda d: np.where(
                d["pitch_hand_raw"] == "S", "R", d["pitch_hand_raw"]
            ),
        )
        .assign(
            bat_side_resolved=lambda d: np.where(
                d["bat_side_raw"] == "S",
                np.where(d["pitch_hand_resolved"] == "R", "L", "R"),
                d["bat_side_raw"],
            ),
        )
    )


def _pbp_pitcher_game_aggs_by_hand(
    pbp: pd.DataFrame,
    player_info: pd.DataFrame,
    bat_side: str,
) -> pd.DataFrame:
    """
    Per (pitcher_id, gamepk) aggregates filtering to batters of a given hand.
    Switch hitters resolved to opposite pitcher hand before filtering.

    Returns: k_rate_vs_{bat_side}, bb_rate_vs_{bat_side}, obp_allowed_vs_{bat_side},
             whiff_rate_vs_{bat_side}, pa_count_vs_{bat_side}.
    """
    s = bat_side
    id_dtype = pbp["batter_id"].dtype
    strikeout_events = frozenset(["Strikeout", "Strikeout Double Play"])
    walk_events = frozenset(["Walk", "Intent Walk"])

    # PA-level events
    pa_resolved = _resolve_handedness(
        pbp[pbp["play_result"].notna()]
        .drop_duplicates(subset=["gamepk", "pitcher_id", "play_id"])
        .copy(),
        player_info, id_dtype,
    )
    pa_filtered = pa_resolved[pa_resolved["bat_side_resolved"] == s]

    pa_aggs = (
        pa_filtered
        .assign(
            is_k=lambda d: d["play_result"].isin(strikeout_events),
            is_bb=lambda d: d["play_result"].isin(walk_events),
            is_onbase=lambda d: d["play_result"].isin(OBP_EVENTS),
        )
        .groupby(["gamepk", "pitcher_id"], sort=False)
        .agg(
            pa_count=("play_result", "count"),
            k_count=("is_k", "sum"),
            bb_count=("is_bb", "sum"),
            obp_count=("is_onbase", "sum"),
        )
        .reset_index()
    )

    # Pitch-level events (whiff)
    pitch_resolved = _resolve_handedness(
        pbp[pbp["is_pitch"]].copy(),
        player_info, id_dtype,
    )
    pitch_filtered = pitch_resolved[pitch_resolved["bat_side_resolved"] == s]

    pitch_aggs = (
        pitch_filtered
        .assign(
            is_swing_miss=lambda d: d["pitch_call_code"].isin(["S", "W", "T"]),
            is_swing=lambda d: d["pitch_call_code"].isin(SWING_CODES),
        )
        .groupby(["gamepk", "pitcher_id"], sort=False)
        .agg(
            swing_misses=("is_swing_miss", "sum"),
            swings=("is_swing", "sum"),
        )
        .reset_index()
    )

    return (
        pa_aggs
        .merge(pitch_aggs, on=["gamepk", "pitcher_id"], how="left")
        .assign(
            **{
                f"k_rate_vs_{s}":      lambda d: d["k_count"]      / d["pa_count"].clip(lower=1),
                f"bb_rate_vs_{s}":     lambda d: d["bb_count"]      / d["pa_count"].clip(lower=1),
                f"obp_allowed_vs_{s}": lambda d: d["obp_count"]     / d["pa_count"].clip(lower=1),
                f"whiff_rate_vs_{s}":  lambda d: d["swing_misses"]  / d["swings"].clip(lower=1),
                f"pa_count_vs_{s}":    lambda d: d["pa_count"].astype(float),
            }
        )
        [["gamepk", "pitcher_id",
          f"k_rate_vs_{s}", f"bb_rate_vs_{s}", f"obp_allowed_vs_{s}",
          f"whiff_rate_vs_{s}", f"pa_count_vs_{s}"]]
    )


def _pitcher_platoon_rolling(pitcher_game: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly-lagged rolling platoon split features for pitchers.
    Windows: last start, 10 starts, season YTD.
    Also computes spytd_obp_allowed (aggregate for shrinkage).

    Expects pitcher_game to contain: k_rate_vs_L, k_rate_vs_R, bb_rate_vs_L,
    bb_rate_vs_R, obp_allowed_vs_L, obp_allowed_vs_R, whiff_rate_vs_L,
    whiff_rate_vs_R, pa_count_vs_L, pa_count_vs_R, obp_allowed.
    """
    pg = pitcher_game.sort_values(["personId", "game_date"])
    con = duckdb.connect()
    con.register("pg", pg)

    return con.execute("""
        SELECT
            personId, gamepk,

            -- last start platoon splits
            AVG(k_rate_vs_L)        OVER w1 AS sp_last_start_k_rate_vs_L,
            AVG(k_rate_vs_R)        OVER w1 AS sp_last_start_k_rate_vs_R,
            AVG(bb_rate_vs_L)       OVER w1 AS sp_last_start_bb_rate_vs_L,
            AVG(bb_rate_vs_R)       OVER w1 AS sp_last_start_bb_rate_vs_R,
            AVG(obp_allowed_vs_L)   OVER w1 AS sp_last_start_obp_allowed_vs_L,
            AVG(obp_allowed_vs_R)   OVER w1 AS sp_last_start_obp_allowed_vs_R,
            AVG(whiff_rate_vs_L)    OVER w1 AS sp_last_start_whiff_rate_vs_L,
            AVG(whiff_rate_vs_R)    OVER w1 AS sp_last_start_whiff_rate_vs_R,

            -- 10-start platoon splits
            AVG(k_rate_vs_L)        OVER w10 AS sp10_k_rate_vs_L,
            AVG(k_rate_vs_R)        OVER w10 AS sp10_k_rate_vs_R,
            AVG(bb_rate_vs_L)       OVER w10 AS sp10_bb_rate_vs_L,
            AVG(bb_rate_vs_R)       OVER w10 AS sp10_bb_rate_vs_R,
            AVG(obp_allowed_vs_L)   OVER w10 AS sp10_obp_allowed_vs_L,
            AVG(obp_allowed_vs_R)   OVER w10 AS sp10_obp_allowed_vs_R,
            AVG(whiff_rate_vs_L)    OVER w10 AS sp10_whiff_rate_vs_L,
            AVG(whiff_rate_vs_R)    OVER w10 AS sp10_whiff_rate_vs_R,

            -- YTD platoon splits
            AVG(k_rate_vs_L)        OVER wytd AS spytd_k_rate_vs_L,
            AVG(k_rate_vs_R)        OVER wytd AS spytd_k_rate_vs_R,
            AVG(bb_rate_vs_L)       OVER wytd AS spytd_bb_rate_vs_L,
            AVG(bb_rate_vs_R)       OVER wytd AS spytd_bb_rate_vs_R,
            AVG(obp_allowed_vs_L)   OVER wytd AS spytd_obp_allowed_vs_L,
            AVG(obp_allowed_vs_R)   OVER wytd AS spytd_obp_allowed_vs_R,
            AVG(whiff_rate_vs_L)    OVER wytd AS spytd_whiff_rate_vs_L,
            AVG(whiff_rate_vs_R)    OVER wytd AS spytd_whiff_rate_vs_R,
            SUM(pa_count_vs_L)      OVER wytd AS spytd_pa_count_vs_L,
            SUM(pa_count_vs_R)      OVER wytd AS spytd_pa_count_vs_R,

            -- YTD aggregate OBP allowed (all batters) — used as shrinkage prior
            AVG(obp_allowed)        OVER wytd AS spytd_obp_allowed

        FROM pg
        WINDOW
            w1   AS (PARTITION BY personId
                     ORDER BY game_date
                     ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING),
            w10  AS (PARTITION BY personId
                     ORDER BY game_date
                     ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
            wytd AS (PARTITION BY personId, game_season
                     ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ORDER BY personId, game_date
    """).df()


def _apply_pitcher_platoon_shrinkage(df: pd.DataFrame, stat: str, k: int) -> pd.DataFrame:
    """
    Bayesian shrinkage for one pitcher platoon stat.
    Blends spytd_{stat}_vs_{hand} toward spytd_{stat} (aggregate prior).
    w = pa_count_vs_{hand} / (pa_count_vs_{hand} + k).

    Null fallback: if both split and aggregate are null (debut, no history),
    fill with league average columns lg_k_rate / lg_bb_rate / etc. already
    joined in _add_log5.
    # TODO: apply per-row debut detection once debut dates are joined in.
    """
    for hand in ["L", "R"]:
        split_col = f"spytd_{stat}_vs_{hand}"
        agg_col   = f"spytd_{stat}"
        pa_col    = f"spytd_pa_count_vs_{hand}"
        out_col   = f"spytd_{stat}_vs_{hand}_shrunk"

        pa = df[pa_col].fillna(0)
        w  = pa / (pa + k)
        df[out_col] = w * df[split_col] + (1 - w) * df[agg_col]
    return df


def _pbp_batter_game_aggs_by_hand(
    pbp: pd.DataFrame,
    player_info: pd.DataFrame,
    pitch_hand: str,
) -> pd.DataFrame:
    """
    Per (batter_id, gamepk) aggregates filtering to pitchers of a given hand.
    Switch-handed pitchers resolved to R before filtering.

    Returns: obp_vs_{pitch_hand}, k_rate_vs_{pitch_hand}, iso_vs_{pitch_hand},
             tb_rate_vs_{pitch_hand}, pa_count_vs_{pitch_hand}.
    """
    s = pitch_hand
    id_dtype = pbp["batter_id"].dtype
    strikeout_events = frozenset(["Strikeout", "Strikeout Double Play"])

    pa_resolved = _resolve_handedness(
        pbp[pbp["play_result"].notna()]
        .drop_duplicates(subset=["gamepk", "batter_id", "play_id"])
        .copy(),
        player_info, id_dtype,
    )
    pa_filtered = pa_resolved[pa_resolved["pitch_hand_resolved"] == s]

    return (
        pa_filtered
        .assign(
            is_k=lambda d: d["play_result"].isin(strikeout_events),
            is_onbase=lambda d: d["play_result"].isin(OBP_EVENTS),
            bases=lambda d: d["play_result"].map(BASES_MAP).fillna(0),
            is_xbh=lambda d: d["play_result"].isin(["Double", "Triple", "Home Run"]),
            extra_bases=lambda d: (
                (d["play_result"] == "Double").astype(int)
                + (d["play_result"] == "Triple").astype(int) * 2
                + (d["play_result"] == "Home Run").astype(int) * 3
            ),
        )
        .groupby(["gamepk", "batter_id"], sort=False)
        .agg(
            pa_count=("play_result",    "count"),
            k_count=("is_k",            "sum"),
            obp_count=("is_onbase",     "sum"),
            total_bases=("bases",       "sum"),
            extra_bases=("extra_bases", "sum"),
        )
        .reset_index()
        .assign(
            **{
                f"k_rate_vs_{s}":  lambda d: d["k_count"]    / d["pa_count"].clip(lower=1),
                f"obp_vs_{s}":     lambda d: d["obp_count"]   / d["pa_count"].clip(lower=1),
                f"iso_vs_{s}":     lambda d: d["extra_bases"] / d["pa_count"].clip(lower=1),
                f"tb_rate_vs_{s}": lambda d: d["total_bases"] / d["pa_count"].clip(lower=1),
                f"pa_count_vs_{s}": lambda d: d["pa_count"].astype(float),
            }
        )
        [["gamepk", "batter_id",
          f"k_rate_vs_{s}", f"obp_vs_{s}", f"iso_vs_{s}",
          f"tb_rate_vs_{s}", f"pa_count_vs_{s}"]]
    )


def _batter_platoon_rolling(batter_game: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly-lagged rolling platoon split features for batters.
    Windows: YTD only (15 and 50 game windows skipped — sample too thin by hand).
    Also computes batytd_obp (aggregate for shrinkage).

    Expects batter_game to contain: obp_vs_L, obp_vs_R, k_rate_vs_L, k_rate_vs_R,
    iso_vs_L, iso_vs_R, tb_rate_vs_L, tb_rate_vs_R, pa_count_vs_L, pa_count_vs_R,
    obp_per_game.
    """
    bg = batter_game.sort_values(["personId", "game_date"])
    con = duckdb.connect()
    con.register("bg", bg)

    return con.execute("""
        SELECT
            personId, gamepk,

            -- YTD platoon splits
            AVG(obp_vs_L)       OVER wytd AS batytd_obp_vs_L,
            AVG(obp_vs_R)       OVER wytd AS batytd_obp_vs_R,
            AVG(k_rate_vs_L)    OVER wytd AS batytd_k_rate_vs_L,
            AVG(k_rate_vs_R)    OVER wytd AS batytd_k_rate_vs_R,
            AVG(iso_vs_L)       OVER wytd AS batytd_iso_vs_L,
            AVG(iso_vs_R)       OVER wytd AS batytd_iso_vs_R,
            AVG(tb_rate_vs_L)   OVER wytd AS batytd_tb_rate_vs_L,
            AVG(tb_rate_vs_R)   OVER wytd AS batytd_tb_rate_vs_R,
            SUM(pa_count_vs_L)  OVER wytd AS batytd_pa_count_vs_L,
            SUM(pa_count_vs_R)  OVER wytd AS batytd_pa_count_vs_R,

            -- YTD aggregate OBP (all pitchers) — used as shrinkage prior
            AVG(obp_per_game)   OVER wytd AS batytd_obp

        FROM bg
        WINDOW
            wytd AS (PARTITION BY personId, game_season
                     ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ORDER BY personId, game_date
    """).df()


def _apply_batter_platoon_shrinkage(df: pd.DataFrame, stat: str, k: int) -> pd.DataFrame:
    """
    Bayesian shrinkage for one batter platoon stat.
    Blends batytd_{stat}_vs_{hand} toward batytd_{stat} (aggregate prior).
    w = pa_count_vs_{hand} / (pa_count_vs_{hand} + k).

    # TODO: apply debut null fallback using lg averages once debut dates joined.
    """
    for hand in ["L", "R"]:
        split_col = f"batytd_{stat}_vs_{hand}"
        agg_col   = f"batytd_{stat}"
        pa_col    = f"batytd_pa_count_vs_{hand}"
        out_col   = f"batytd_{stat}_vs_{hand}_shrunk"

        pa = df[pa_col].fillna(0)
        w  = pa / (pa + k)
        df[out_col] = w * df[split_col] + (1 - w) * df[agg_col]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def build_pa_features(target_years: list[int]) -> None:
    """
    End-to-end pipeline for a list of target years.
    Loads only target_years; rows with null rolling features are dropped before writing.
    Writes one parquet per target year to S3.
    """
    all_years = sorted(target_years)
    logger.info(f"Load years: {all_years}")

    # ── Load ─────────────────────────────────────────────────────────────────
    logger.info("Loading PBP …")
    pbp = pd.concat([_read_pbp(y) for y in all_years], ignore_index=True)

    # ── Pre-step: Statcast coverage check ────────────────────────────────────
    statcast_cols = ['launch_speed', 'launch_angle', 'start_speed',
                     'spin_rate', 'release_pos_y', 'break_horizontal',
                     'break_vertical_induced']
    print("=== Statcast coverage check ===")
    print(f"Total PBP rows: {len(pbp):,}")
    print(f"Is pitch rows: {pbp['is_pitch'].sum():,}")
    print("\nNull rates on Statcast columns (pitch rows only):")
    print((pbp[pbp['is_pitch']][statcast_cols].isna().mean() * 100).round(1).to_string())
    print("\nSample of populated rows:")
    print(pbp[pbp['is_pitch'] & pbp['launch_speed'].notna()][statcast_cols].head(5).to_string())

    logger.info("Loading pitcher boxscore …")
    pitcher_box = pd.concat([_read_pitcher_box(y) for y in all_years], ignore_index=True)
    pitcher_box["game_date"] = pd.to_datetime(pitcher_box["game_date"])

    logger.info("Loading batter boxscore …")
    batter_box = pd.concat([_read_batter_box(y) for y in all_years], ignore_index=True)
    batter_box["game_date"] = pd.to_datetime(batter_box["game_date"])

    logger.info("Loading schedule …")
    schedule = pd.concat([_read_schedule(y) for y in all_years], ignore_index=True)

    logger.info("Loading player info …")
    player_info = _read_player_info()

    # Normalize personId column name once at load — never touch again downstream
    _id_col = next(
        (c for c in player_info.columns
         if c.lower() in ("personid", "id", "mlbam_id", "player_id", "person_id")),
        None
    )
    assert _id_col is not None, f"player_info missing ID column — found: {player_info.columns.tolist()}"
    if _id_col != "personId":
        player_info = player_info.rename(columns={_id_col: "personId"})
    logger.info(f"player_info personId normalized from '{_id_col}', rows: {len(player_info)}")

    # Align personId dtype to batter_id from the already-loaded PBP
    _batter_dtype = pbp["batter_id"].dtype
    player_info = player_info.assign(personId=lambda d: d["personId"].astype(_batter_dtype))
    logger.info(f"player_info personId cast to {_batter_dtype}")

    venue_map = (
        schedule[["gamepk", "venue_id"]]
        .drop_duplicates("gamepk")
        .assign(park_factor=lambda d: d["venue_id"].astype(str).map(PARK_FACTORS).fillna(100.0))
    )

    # ── Starters & SP plate appearances ──────────────────────────────────────
    logger.info("Identifying starters …")
    starters = _identify_starters(pbp)

    logger.info("Extracting SP plate appearances …")
    sp_pa = _extract_sp_pa(pbp, starters)

    # Join game_date + game_season from pitcher_box (not in PBP schema)
    game_meta = pitcher_box[["gamepk", "game_date", "game_season"]].drop_duplicates("gamepk")
    sp_pa = sp_pa.merge(game_meta, on="gamepk", how="left")
    sp_pa["game_date"] = pd.to_datetime(sp_pa["game_date"])

    # ── Pre-step: verify join keys ────────────────────────────────────────────
    print("=== Join key verification ===")
    print("batter_team_id in sp_pa:", "batter_team_id" in sp_pa.columns)
    print("pitcher_team_id in sp_pa:", "pitcher_team_id" in sp_pa.columns)
    print("batter_team_id dtype:", sp_pa["batter_team_id"].dtype)
    print("pitcher_team_id dtype:", sp_pa["pitcher_team_id"].dtype)
    print("batter_box columns:", batter_box.columns.tolist())
    print("batter_team_id sample:", sp_pa["batter_team_id"].value_counts().head(5).to_string())

    # ── Data contract checks ──────────────────────────────────────────────────
    logger.info("Running data contract checks …")
    assert "personId" in player_info.columns, "player_info missing personId"
    assert "batter_id" in sp_pa.columns, "sp_pa missing batter_id"
    assert "pitcher_id" in sp_pa.columns, "sp_pa missing pitcher_id"
    assert player_info["personId"].dtype == sp_pa["batter_id"].dtype, (
        f"dtype mismatch: player_info personId={player_info['personId'].dtype} "
        f"vs sp_pa batter_id={sp_pa['batter_id'].dtype}"
    )
    print("── Data contract checks ──")
    print(f"player_info columns: {player_info.columns.tolist()}")
    print(f"player_info personId dtype: {player_info['personId'].dtype}")
    print(f"player_info rows: {len(player_info)}")
    print(f"sp_pa batter_id dtype: {sp_pa['batter_id'].dtype}")
    print(f"sp_pa pitcher_id dtype: {sp_pa['pitcher_id'].dtype}")
    print("All data contract checks passed ✓")

    # ── Game-context features ─────────────────────────────────────────────────
    logger.info("Computing game-context features …")
    sp_pa = (
        sp_pa
        .pipe(_add_target)
        .pipe(_add_tto)
        .pipe(_add_pitch_count, pbp=pbp)
        .pipe(_add_base_out_state)
        .pipe(_add_handedness, player_info=player_info)
        .merge(venue_map, on="gamepk", how="left")
    )

    # ── PBP game aggregates ───────────────────────────────────────────────────
    logger.info("Aggregating PBP by pitcher-game …")
    pbp_sp_aggs  = _pbp_pitcher_game_aggs(pbp)

    logger.info("Aggregating PBP by pitcher-game (innings 4-5) …")
    inn45_aggs = _pbp_pitcher_inn45_aggs(pbp)

    print("\n── inn45_aggs diagnostic ──")
    print(inn45_aggs[['gamepk', 'pitcher_id', 'inn45_pitches', 'inn45_pa']].describe())

    logger.info("Aggregating PBP by batter-game …")
    pbp_bat_aggs = _pbp_batter_game_aggs(pbp)

    # ── Build game-level stat tables ──────────────────────────────────────────
    logger.info("Building pitcher game table …")
    pitcher_game = _build_pitcher_game(pitcher_box, pbp_sp_aggs, starters)

    print("\n── pitcher_game diagnostic ──")
    print(pitcher_game[['personId', 'gamepk', 'total_pitches', 'pa_count']].describe())
    print(pitcher_game[['total_pitches', 'pa_count']].head(10))

    logger.info("Building batter game table …")
    batter_game  = _build_batter_game(batter_box, pbp_bat_aggs)

    # ── Team offensive context features ──────────────────────────────────────
    logger.info("Computing team rolling features …")
    team_game = _build_team_game(batter_box, pbp_bat_aggs, sp_pa)
    team_rolling = _team_rolling(team_game)

    print("\n=== Task 1 verification ===")
    print(f"team_game rows: {len(team_game):,}")
    print("Expected: ~30 teams × ~162 games = ~4,860 rows for 2023")
    print(team_game[["team_k_rate", "team_bb_rate", "team_obp",
                      "team_iso", "team_p_per_pa", "team_chase_rate"]].describe())
    print(f"\nteam_k_rate mean: {team_game['team_k_rate'].mean():.3f} (expect ~0.22)")
    print(f"team_bb_rate mean: {team_game['team_bb_rate'].mean():.3f} (expect ~0.085)")
    print(f"team_obp mean: {team_game['team_obp'].mean():.3f} (expect ~0.315-0.325)")
    print(f"team_p_per_pa mean: {team_game['team_p_per_pa'].mean():.3f} (expect ~3.8-4.0)")
    most_patient = team_game.groupby("team_id")["team_bb_rate"].mean().idxmax()
    print(f"\nMost patient team_id: {most_patient}")
    print(team_game[team_game["team_id"] == most_patient][
        ["gamepk", "team_k_rate", "team_bb_rate", "team_obp", "team_p_per_pa"]
    ].head(5).to_string())

    print("\n=== Task 2 verification ===")
    print(f"team_rolling rows: {len(team_rolling):,}")
    last_game_cols = [c for c in team_rolling.columns if "last_game" in c]
    ten_g_cols     = [c for c in team_rolling.columns if "10g" in c]
    ytd_cols       = [c for c in team_rolling.columns if "ytd" in c]
    print(f"last_game cols: {len(last_game_cols)}")
    print(f"10g cols: {len(ten_g_cols)}")
    print(f"ytd cols: {len(ytd_cols)}")
    print("\nNull rates:")
    print((team_rolling[last_game_cols + ten_g_cols[:3]].isna().mean() * 100).round(1).to_string())
    _test_team = team_rolling[team_rolling["team_id"] == most_patient].sort_values("game_date")
    print(f"\nMost patient team rolling (first 15 games):")
    print(_test_team[[
        "game_date",
        "team_last_game_bb_rate", "team_10g_bb_rate", "team_ytd_bb_rate",
        "team_last_game_p_per_pa", "team_10g_p_per_pa", "team_ytd_p_per_pa",
        "team_10g_n_games",
    ]].head(15).to_string())

    # ── Pitcher platoon game aggregates ───────────────────────────────────────
    logger.info("Computing pitcher platoon game aggregates …")
    pbp_sp_aggs_L = _pbp_pitcher_game_aggs_by_hand(pbp, player_info, "L")
    pbp_sp_aggs_R = _pbp_pitcher_game_aggs_by_hand(pbp, player_info, "R")

    # Combined per-game OBP allowed (all batters) — used as shrinkage aggregate
    obp_combined = (
        pbp_sp_aggs_L.rename(columns={"pa_count_vs_L": "_pa_L", "obp_allowed_vs_L": "_obp_L"})
        [["gamepk", "pitcher_id", "_pa_L", "_obp_L"]]
        .merge(
            pbp_sp_aggs_R.rename(columns={"pa_count_vs_R": "_pa_R", "obp_allowed_vs_R": "_obp_R"})
            [["gamepk", "pitcher_id", "_pa_R", "_obp_R"]],
            on=["gamepk", "pitcher_id"], how="outer",
        )
        .assign(
            obp_allowed=lambda d: (
                (d["_obp_L"].fillna(0) * d["_pa_L"].fillna(0)
                 + d["_obp_R"].fillna(0) * d["_pa_R"].fillna(0))
                / (d["_pa_L"].fillna(0) + d["_pa_R"].fillna(0)).clip(lower=1)
            )
        )
        [["gamepk", "pitcher_id", "obp_allowed"]]
    )

    pitcher_game = (
        pitcher_game
        .merge(pbp_sp_aggs_L.rename(columns={"pitcher_id": "personId"}),
               on=["gamepk", "personId"], how="left")
        .merge(pbp_sp_aggs_R.rename(columns={"pitcher_id": "personId"}),
               on=["gamepk", "personId"], how="left")
        .merge(obp_combined.rename(columns={"pitcher_id": "personId"}),
               on=["gamepk", "personId"], how="left")
    )

    # ── Process quality game aggregates ──────────────────────────────────────
    logger.info("Aggregating PBP process quality by pitcher-game …")
    process_aggs = _pbp_pitcher_process_aggs(pbp, player_info, starters)

    print("\n=== Task 1 verification ===")
    print(process_aggs.describe())
    print(f"\nRows in process_aggs: {len(process_aggs):,}")
    print(f"Null rates:")
    print((process_aggs.isna().mean() * 100).round(1).to_string())
    print(f"\nMean contact_exit_velo: {process_aggs['contact_exit_velo'].mean():.1f}")
    print(f"Mean inplay_exit_velo:  {process_aggs['inplay_exit_velo'].mean():.1f}")
    print(f"Mean avg_velo: {process_aggs['avg_velo'].mean():.1f}")
    print(f"Mean zone_pct: {process_aggs['zone_pct'].mean():.3f}")

    # Single pitcher game check — Gerrit Cole (543037)
    cole_games = process_aggs[process_aggs["pitcher_id"] == "543037"]
    print("\n=== Cole process aggs sample (5 games) ===")
    print(cole_games[[
        "gamepk", "avg_velo", "avg_spin_rate",
        "contact_exit_velo", "inplay_exit_velo",
        "contact_hard_hit_pct", "zone_pct", "chase_induced_pct",
        "two_strike_whiff_pct", "contact_count"
    ]].head(5).to_string())

    pitcher_game = pitcher_game.merge(
        process_aggs.rename(columns={"pitcher_id": "personId"}),
        on=["gamepk", "personId"],
        how="left",
    )

    # ── Batter platoon game aggregates ────────────────────────────────────────
    logger.info("Computing batter platoon game aggregates …")
    pbp_bat_aggs_L = _pbp_batter_game_aggs_by_hand(pbp, player_info, "L")
    pbp_bat_aggs_R = _pbp_batter_game_aggs_by_hand(pbp, player_info, "R")

    # Combined per-game batter OBP (all pitchers) — used as shrinkage aggregate
    bat_obp_combined = (
        pbp_bat_aggs_L.rename(columns={"pa_count_vs_L": "_pa_L", "obp_vs_L": "_obp_L"})
        [["gamepk", "batter_id", "_pa_L", "_obp_L"]]
        .merge(
            pbp_bat_aggs_R.rename(columns={"pa_count_vs_R": "_pa_R", "obp_vs_R": "_obp_R"})
            [["gamepk", "batter_id", "_pa_R", "_obp_R"]],
            on=["gamepk", "batter_id"], how="outer",
        )
        .assign(
            obp_per_game=lambda d: (
                (d["_obp_L"].fillna(0) * d["_pa_L"].fillna(0)
                 + d["_obp_R"].fillna(0) * d["_pa_R"].fillna(0))
                / (d["_pa_L"].fillna(0) + d["_pa_R"].fillna(0)).clip(lower=1)
            )
        )
        [["gamepk", "batter_id", "obp_per_game"]]
    )

    batter_game = (
        batter_game
        .merge(pbp_bat_aggs_L.rename(columns={"batter_id": "personId"}),
               on=["gamepk", "personId"], how="left")
        .merge(pbp_bat_aggs_R.rename(columns={"batter_id": "personId"}),
               on=["gamepk", "personId"], how="left")
        .merge(bat_obp_combined.rename(columns={"batter_id": "personId"}),
               on=["gamepk", "personId"], how="left")
    )

    # ── Rolling features ──────────────────────────────────────────────────────
    logger.info("Computing pitcher rolling features …")
    sp_rolling = _pitcher_rolling(pitcher_game, inn45_aggs=inn45_aggs)

    logger.info("Computing pitcher platoon rolling features …")
    sp_platoon = _pitcher_platoon_rolling(pitcher_game)
    sp_rolling = sp_rolling.merge(sp_platoon, on=["personId", "gamepk"], how="left")

    print("\n=== Task 2 verification ===")
    last_start_cols = [c for c in sp_rolling.columns if 'sp_last_start' in c and
                       any(x in c for x in ['velo', 'spin', 'contact', 'zone', 'chase'])]
    sp10_cols = [c for c in sp_rolling.columns if c.startswith('sp10_') and
                 any(x in c for x in ['velo', 'spin', 'contact', 'zone', 'chase'])]
    spytd_cols = [c for c in sp_rolling.columns if c.startswith('spytd_') and
                  any(x in c for x in ['velo', 'spin', 'contact', 'zone', 'chase'])]
    print(f"Last start process cols: {len(last_start_cols)} — {last_start_cols}")
    print(f"sp10 process cols: {len(sp10_cols)} — {sp10_cols}")
    print(f"spytd process cols: {len(spytd_cols)} — {spytd_cols}")
    print("\nLast start distributions:")
    print(sp_rolling[last_start_cols].describe())
    cole = sp_rolling[sp_rolling['personId'] == "543037"].sort_values('game_date')
    print("\nCole process quality (last 5 starts):")
    print(cole[['game_date',
                'sp_last_start_avg_velo', 'sp10_avg_velo', 'sp_velo_trend',
                'sp_last_start_contact_exit_velo', 'sp10_contact_exit_velo',
                'sp_last_start_zone_pct', 'sp_last_start_chase_induced_pct'
                ]].tail(5).to_string())
    print("\nTrend feature distributions:")
    print(sp_rolling[['sp_velo_trend', 'sp_spin_trend',
                       'sp_contact_velo_trend']].describe())

    foul_cols = [c for c in sp_rolling.columns if 'foul' in c]
    print(f"Foul columns remaining: {foul_cols}")

    logger.info("Computing batter rolling features …")
    bat_rolling = _batter_rolling(batter_game)

    top_batter = bat_rolling['personId'].value_counts().index[0]
    test_bat = bat_rolling[bat_rolling['personId'] == top_batter].sort_values('game_date')
    print("=== Top batter last game features ===")
    print(test_bat[[
        'game_date',
        'bat_last_game_k_rate',
        'bat_last_game_iso',
        'bat_last_game_tb_rate',
        'bat_days_since_last_game',
    ]].head(15).to_string())

    logger.info("Computing batter platoon rolling features …")
    bat_platoon = _batter_platoon_rolling(batter_game)
    bat_rolling = bat_rolling.merge(bat_platoon, on=["personId", "gamepk"], how="left")

    # ── Career-average fallback (YTD PA < 30) ─────────────────────────────────
    logger.info("Applying career-average fallback …")
    sp_rolling = _add_career_avg_fallback(
        sp_rolling, pitcher_game, SP_STAT_COLS,
        ytd_pa_col="spytd_pa_count", ytd_prefix="spytd_", out_prefix="sp_",
    )
    bat_rolling = _add_career_avg_fallback(
        bat_rolling, batter_game, BAT_STAT_COLS,
        ytd_pa_col="batytd_pa_count", ytd_prefix="batytd_", out_prefix="bat_",
    )

    # ── Pitcher platoon shrinkage (YTD only) ──────────────────────────────────
    logger.info("Applying pitcher platoon shrinkage …")
    for _stat, _k in [
        ("k_rate",      PLATOON_K["k_rate"]),
        ("bb_rate",     PLATOON_K["bb_rate"]),
        ("obp_allowed", PLATOON_K["obp"]),
        ("whiff_rate",  PLATOON_K["whiff_rate"]),
    ]:
        sp_rolling = _apply_pitcher_platoon_shrinkage(sp_rolling, _stat, _k)

    # ── Part 1 validation print ───────────────────────────────────────────────
    test = sp_rolling[sp_rolling["personId"] == 543037].sort_values("game_date")
    print("=== Cole platoon splits ===")
    print(test[[
        "game_date",
        "spytd_k_rate_vs_L", "spytd_k_rate_vs_R",
        "spytd_k_rate_vs_L_shrunk", "spytd_k_rate_vs_R_shrunk",
        "spytd_pa_count_vs_L", "spytd_pa_count_vs_R",
    ]].tail(10).to_string())

    # ── Batter platoon shrinkage (YTD only) ───────────────────────────────────
    logger.info("Applying batter platoon shrinkage …")
    for _stat, _k in [
        ("obp",     PLATOON_K["obp"]),
        ("k_rate",  PLATOON_K["k_rate"]),
        ("iso",     PLATOON_K["iso"]),
        ("tb_rate", PLATOON_K["tb_rate"]),
    ]:
        bat_rolling = _apply_batter_platoon_shrinkage(bat_rolling, _stat, _k)

    # ── Part 2 validation print ───────────────────────────────────────────────
    print(bat_rolling.filter(like="_vs_L").describe())
    print(bat_rolling.filter(like="_vs_R").describe())
    print(f"pa_count_vs_L mean: {bat_rolling['batytd_pa_count_vs_L'].mean():.1f}")
    print(f"pa_count_vs_R mean: {bat_rolling['batytd_pa_count_vs_R'].mean():.1f}")

    _lhb_ids = (
        player_info[player_info["batSide"] == "L"]["personId"]
        .astype(bat_rolling["personId"].dtype)
    )
    lhb = bat_rolling[bat_rolling["personId"].isin(_lhb_ids)]
    print(f"\nLHB OBP vs R: {lhb['batytd_obp_vs_R_shrunk'].mean():.3f}")
    print(f"LHB OBP vs L: {lhb['batytd_obp_vs_L_shrunk'].mean():.3f}")
    print("LHB should show higher OBP vs R than vs L")

    # ── Last-season average features ──────────────────────────────────────────
    logger.info("Computing last-season average features …")
    sp_rolling = _add_last_season_avgs(sp_rolling, pitcher_game, SP_STAT_COLS, out_prefix="sp_")
    bat_rolling = _add_last_season_avgs(bat_rolling, batter_game, BAT_STAT_COLS, out_prefix="bat_")

    # ── Join rolling features onto PA rows ────────────────────────────────────
    logger.info("Joining features to PA rows …")

    # Raw per-game stat columns to exclude from join (would be current-game leakage)
    sp_raw  = set(SP_STAT_COLS + ["k_bb_ratio", "pa_count", "ip",
                                   "k_rate", "bb_rate", "hr_rate"])
    bat_raw = set(BAT_STAT_COLS + ["plate_appearances",
                                    "k_rate", "bb_rate", "hr_rate", "iso", "tb_rate",
                                    "chase_rate", "z_contact_pct", "o_contact_pct"])

    sp_join_cols = [c for c in sp_rolling.columns
                    if c not in sp_raw or c in ("gamepk", "personId")]
    bat_join_cols = [c for c in bat_rolling.columns
                     if c not in bat_raw or c in ("gamepk", "personId")]

    features = (
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

    # ── Team offensive context join ───────────────────────────────────────────
    logger.info("Joining team rolling features …")
    team_feat_cols = [
        c for c in team_rolling.columns
        if any(c.startswith(p) for p in ("team_last_game_", "team_10g_", "team_ytd_"))
    ]

    # Opposing team — batter_team_id faces this pitcher
    features = features.merge(
        team_rolling[["team_id", "gamepk"] + team_feat_cols].rename(columns={
            "team_id": "batter_team_id",
            **{c: f"opp_{c}" for c in team_feat_cols},
        }),
        on=["gamepk", "batter_team_id"],
        how="left",
    )

    # Own team — pitcher_team_id is the pitcher's own team
    features = features.merge(
        team_rolling[["team_id", "gamepk"] + team_feat_cols].rename(columns={
            "team_id": "pitcher_team_id",
            **{c: f"own_{c}" for c in team_feat_cols},
        }),
        on=["gamepk", "pitcher_team_id"],
        how="left",
    )

    print("\n=== Task 3 verification ===")
    opp_feats = [c for c in features.columns if c.startswith("opp_team_")]
    own_feats = [c for c in features.columns if c.startswith("own_team_")]
    print(f"opp_team features: {len(opp_feats)}")
    print(f"own_team features: {len(own_feats)}")
    same = (features["opp_team_ytd_obp"] == features["own_team_ytd_obp"]).sum()
    print(f"\nRows where opp == own OBP: {same} (expect ~0)")
    print("\nNull rates on team features:")
    print(
        features[opp_feats + own_feats].isna().mean()
        .sort_values(ascending=False).head(10).to_string()
    )
    print("\nopp_team distributions:")
    print(features[[
        "opp_team_last_game_obp", "opp_team_10g_obp", "opp_team_ytd_obp",
        "opp_team_last_game_p_per_pa", "opp_team_10g_p_per_pa", "opp_team_ytd_p_per_pa",
        "opp_team_ytd_bb_rate", "opp_team_ytd_k_rate",
    ]].describe())
    _game_check = features.groupby("gamepk").agg(
        unique_opp_obp=("opp_team_ytd_obp", "nunique"),
        unique_own_obp=("own_team_ytd_obp", "nunique"),
    ).reset_index()
    print("\nGame level consistency:")
    print(_game_check[["unique_opp_obp", "unique_own_obp"]].value_counts())

    # ── Log5 ─────────────────────────────────────────────────────────────────
    logger.info("Computing log5 interaction features …")
    features = _add_log5(features, pitcher_game)

    # ── Post-shift ban indicator ──────────────────────────────────────────────
    features["post_shift_ban"] = (features["game_date"] >= "2023-03-30").astype("int8")

    # ── Defensive context ─────────────────────────────────────────────────────
    team_oaa = _load_team_oaa(target_years)
    features = features.merge(
        team_oaa.rename(columns={"team_id": "pitcher_team_id"}),
        on=["pitcher_team_id", "game_season"],
        how="left",
    ) if len(team_oaa) else features.assign(team_oaa_season=np.nan)

    # ── Drop rows with null rolling features ──────────────────────────────────
    rolling_cols = [c for c in features.columns if c.startswith((
        "sp10_", "spytd_", "bat_last_game_", "bat15_", "bat50_", "batytd_"
    ))]

    # ── Dropna diagnostic ─────────────────────────────────────────────────────
    features_pre_drop = features.copy()
    before = len(features_pre_drop)

    null_pct = (
        features_pre_drop[rolling_cols].isnull().mean() * 100
    ).round(1).sort_values(ascending=False)

    dropped_mask = features_pre_drop[rolling_cols].isnull().any(axis=1)
    dropped = features_pre_drop[dropped_mask].copy()
    dropped["game_month"] = pd.to_datetime(dropped["game_date"]).dt.month

    print(f"\n── Dropna diagnostic ──")
    print(f"Total rows before dropna: {before:,}")
    print(f"Rows with any null rolling feature: {dropped_mask.sum():,} ({dropped_mask.mean()*100:.1f}%)")
    print(f"\nTop 15 columns by null rate:")
    print(null_pct[null_pct > 0].head(15).to_string())
    print(f"\nDropped rows by season:")
    print(dropped["game_season"].value_counts().sort_index().to_string())
    print(f"\nDropped rows by month:")
    print(dropped["game_month"].value_counts().sort_index().to_string())
    print(f"\nDropped rows by pitcher (top 10):")
    print(dropped["pitcher_id"].value_counts().head(10).to_string())

    features = features.dropna(subset=rolling_cols)
    logger.info(f"Dropped {before - len(features):,} rows with null rolling features; {len(features):,} remaining")

    return features, sp_rolling, bat_rolling, player_info

pa_feats, sp_rolling, bat_rolling, player_info = build_pa_features(list(range(2017,2025)))

cole_fix1 = sp_rolling[sp_rolling['personId'] == "543037"].sort_values('game_date')
print("=== Fix 1 — Cole velo after starter filter ===")
print(cole_fix1[['game_date', 'sp_last_start_avg_velo', 'sp10_avg_velo']].tail(8).to_string())

print("\n=== Fix 2 — Raw PBP count_strikes convention ===")
_cole_gamepk = sp_rolling[sp_rolling['personId'] == "543037"]['gamepk'].iloc[0]
_pbp_diag = _read_pbp(2023)
cole_pitches = _pbp_diag[
    (_pbp_diag['gamepk'] == _cole_gamepk) &
    (_pbp_diag['pitcher_id'] == "543037") &
    (_pbp_diag['is_pitch'])
][['play_id', 'pitch_number', 'count_balls', 'count_strikes', 'pitch_call_code']].head(25)
print(cole_pitches.to_string())

print("\n=== Velo trend diagnostic ===")
print(f"sp10_avg_velo null rate: {sp_rolling['sp10_avg_velo'].isna().mean():.3f}")
print(f"sp_velo_trend null rate: {sp_rolling['sp_velo_trend'].isna().mean():.3f}")

non_null = sp_rolling[
    sp_rolling['sp_velo_trend'].notna() &
    sp_rolling['sp10_avg_velo'].notna()
]
print(f"\nNon-null rows: {len(non_null):,}")
print("\nCorrelation on non-null rows only:")
print(non_null[['sp_velo_trend', 'sp_last_start_avg_velo',
                'sp10_avg_velo']].corr().to_string())

print("\nSample of 10 rows — verify subtraction is correct:")
print(non_null[['game_date', 'sp_last_start_avg_velo',
                'sp10_avg_velo', 'sp_velo_trend']].head(10).to_string())

cole = sp_rolling[sp_rolling['personId'] == "543037"].sort_values('game_date')

print("=== Cole process quality — last 8 starts ===")
print(cole[[
    'game_date',
    'sp_last_start_avg_velo', 'sp10_avg_velo', 'sp_velo_trend',
    'sp_last_start_avg_spin_rate', 'sp10_avg_spin_rate', 'sp_spin_trend',
    'sp_last_start_contact_exit_velo', 'sp10_contact_exit_velo',
    'sp_last_start_zone_pct', 'sp_last_start_chase_induced_pct',
    'sp_last_start_two_strike_whiff_pct',
]].tail(8).to_string())

new_control_cols = [c for c in sp_rolling.columns if
                    any(x in c for x in ['whiff_rate_ahead', 'whiff_rate_behind', 'called_strike_edge'])]
print(f"\nNew control cols: {new_control_cols}")
print(sp_rolling[new_control_cols].describe())
cole_ctrl = sp_rolling[sp_rolling['personId'] == "543037"].sort_values('game_date')
print("\nCole control quality (last 5 starts):")
print(cole_ctrl[['game_date'] + new_control_cols].tail(5).to_string())

print("\n── p_per_pa feature diagnostic ──")
print(pa_feats[['sp_last_start_p_per_pa', 'sp10_p_per_pa_inn4_5']].describe())

print("\n── Handedness matchup features ──")
print(pa_feats[["platoon_advantage", "same_hand"]].value_counts())
print(pa_feats[["platoon_advantage", "same_hand"]].mean())
print(pa_feats[["platoon_advantage", "same_hand"]].isna().sum())
print("bat_side_resolved in columns:", "bat_side_resolved" in pa_feats.columns)
print("pitch_hand_resolved in columns:", "pitch_hand_resolved" in pa_feats.columns)

print("\n── 3yr_wtd / last_szn / ytd column comparison ──")
print(pa_feats[['sp_3yr_wtd_k_rate', 'sp_last_szn_k_rate', 'spytd_k_rate']].describe())
print("\nColumns still containing 'career':", pa_feats.filter(like='career').columns.tolist())

print("\n── sp_last_start features ──")
print(pa_feats.filter(like='sp_last_start').describe())

print("\n── Last start vs rolling K-rate correlation ──")
print(pa_feats[['sp_last_start_k_rate', 'sp10_k_rate']].corr())

print("\n── Fatigue score ──")
print(pa_feats[['fatigue_score', 'days_rest', 'sp_last_start_pitches']].describe())

print("\n── Late inning (inn 4-5) features ──")
print(pa_feats.filter(like='inn4_5').describe())

print("\n── sp3_ columns (should be empty) ──")
print(pa_feats.filter(like='sp3_').columns.tolist())


# gotta think through what's actually going to give me lift

# pitcher process 
# batter ability we should look at
# team ability 
# interaction terms for sure


# probably just want to switch to bullpen and see what it looks like there
# 
#  


# ── Model training and evaluation ─────────────────────────────────────────
_FEATURE_PREFIXES = (
    "sp10_", "spytd_", "sp_3yr_wtd_", "sp_last_szn_", "sp_last_start_",
    "bat_last_game_", "bat15_", "bat50_", "batytd_", "bat_3yr_wtd_", "bat_last_szn_",
    "log5_",
    "opp_team_last_game_", "opp_team_10g_", "opp_team_ytd_",
    "own_team_last_game_", "own_team_10g_", "own_team_ytd_",
)
_CONTEXT_FEATURES = [
    "times_through_order", "pitch_count_in_game", "base_out_state",
    "outs_when_up", "batting_order_int", "park_factor",
    "team_oaa_season", "post_shift_ban", "days_rest", "cumulative_ip_last_7d",
]

feat_cols = [
    c for c in pa_feats.columns
    if (c.startswith(_FEATURE_PREFIXES) or c in _CONTEXT_FEATURES)
]
logger.info(f"Using {len(feat_cols)} feature columns")

# ── Task 3 verification ───────────────────────────────────────────────────
process_feats = [c for c in feat_cols if any(x in c for x in
                 ['avg_velo', 'spin_rate', 'avg_extension', 'contact_exit_velo',
                  'hard_contact_pct', 'sweet_spot_pct', 'zone_pct',
                  'chase_induced_pct', 'two_strike_whiff', 'velo_trend',
                  'spin_trend', 'contact_velo_trend'])]
print(f"Process quality features in model: {len(process_feats)}")
print(process_feats)

print("\n=== Final validation ===")
print(f"Total features: {len(feat_cols)}")
print(f"pa_feats shape: {pa_feats.shape}")

print("\nNull rates on process quality features:")
print(pa_feats[process_feats].isna().mean().sort_values(ascending=False).head(10).to_string())

check = pa_feats.groupby(['gamepk', 'pitcher_id']).agg(
    unique_velo=('sp_last_start_avg_velo', 'nunique'),
    unique_contact=('sp_last_start_contact_exit_velo', 'nunique'),
).reset_index()
print("\nGame level consistency:")
print(check[['unique_velo', 'unique_contact']].value_counts())

print("\nVelo trend correlation check:")
print(pa_feats[['sp_velo_trend', 'sp_last_start_avg_velo',
                'sp10_avg_velo']].corr().to_string())

# Temporal split
TRAIN_YEARS = set(range(2015, 2023)) - {2021}
VAL_YEARS   = {2023}
TEST_YEARS  = {2024}

train_mask = pa_feats["game_season"].isin(TRAIN_YEARS)
val_mask   = pa_feats["game_season"].isin(VAL_YEARS)
test_mask  = pa_feats["game_season"].isin(TEST_YEARS)

X_train = pa_feats.loc[train_mask, feat_cols]
y_train = pa_feats.loc[train_mask, "reached_base"]
X_val   = pa_feats.loc[val_mask,   feat_cols]
y_val   = pa_feats.loc[val_mask,   "reached_base"]
X_test  = pa_feats.loc[test_mask,  feat_cols]
y_test  = pa_feats.loc[test_mask,  "reached_base"]


logger.info(f"Split sizes — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

test_probs = model.predict_proba(X_test)[:, 1]
val_probs  = model.predict_proba(X_val)[:, 1]

for split_name, y_s, probs_s in [
    ("Val  (2023)", y_val,  val_probs),
    ("Test (2024)", y_test, test_probs),
]:
    print(f"\n── {split_name} ──────────────────────────")
    print(f"  ROC-AUC : {roc_auc_score(y_s, probs_s):.4f}")
    print(f"  Log loss: {log_loss(y_s, probs_s):.4f}")

baseline_p     = float(y_train.mean())
baseline_probs = np.full(len(y_test), baseline_p)

print(f"\n── Baseline — train-set league avg P(reached) = {baseline_p:.4f} applied to Test (2024) ──")
print(f"  ROC-AUC : {roc_auc_score(y_test, baseline_probs):.4f}")
print(f"  Log loss: {log_loss(y_test, baseline_probs):.4f}")

frac_pos, mean_pred = calibration_curve(y_test, test_probs, n_bins=10)
print("\n── Calibration curve — Test 2024 (mean_predicted → fraction_positive) ──")
print(f"  {'mean_pred':>12}  {'frac_pos':>12}")
for mp, fp in zip(mean_pred, frac_pos):
    print(f"  {mp:12.4f}  {fp:12.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(mean_pred, frac_pos, "s-", label="XGBoost")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.axhline(baseline_p, color="gray", linestyle=":", label=f"Baseline ({baseline_p:.3f})")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration curve – Test 2024")
ax.legend()
plt.tight_layout()
plt.show()

importance = (
    pd.Series(model.feature_importances_, index=feat_cols)
    .sort_values(ascending=False)
    .head(20)
)
print("\n── Top 20 Feature Importances ──")
print(importance.to_string())

