#!/usr/bin/env python3
"""
debug_targets.py
================
Traces why target_bases / reached_base are all 0 for a single game.
Prints only to console. No S3 writes.

Usage:
  python debug_targets.py
"""
import awswrangler as wr
import pandas as pd
import numpy as np

BUCKET   = "mlbdk"
GAMEPK   = 413716
YEAR     = 2019   # adjust if gamepk is from a different year

BASES_MAP = {
    "home_run": 4, "triple": 3, "double": 2,
    "single": 1, "walk": 1, "intent_walk": 1,
    "hit_by_pitch": 1, "field_error": 1, "catcher_interf": 1,
    "strikeout": 0, "strikeout_double_play": 0,
    "field_out": 0, "grounded_into_double_play": 0, "double_play": 0,
    "triple_play": 0, "fielders_choice": 0, "fielders_choice_out": 0,
    "force_out": 0, "sac_fly": 0, "sac_bunt": 0,
    "sac_fly_double_play": 0, "bunt_groundout": 0, "bunt_lineout": 0,
    "bunt_pop_out": 0, "runner_double_play": 0, "other_out": 0,
}

SEP = "─" * 70

# ── Step 0: load raw PBP for the game ─────────────────────────────────────────
print(SEP)
print("STEP 0 — Raw PBP for gamepk", GAMEPK)
print(SEP)

pbp_all = wr.s3.read_parquet(f"s3://{BUCKET}/processed_data/prepared/playbyplay/{YEAR}/")
pbp = pbp_all[pbp_all["gamepk"] == GAMEPK].copy()
print(f"Total rows: {len(pbp)}")

# ── Step 1: column inventory ───────────────────────────────────────────────────
print(SEP)
print("STEP 1 — Column names")
print(SEP)
print(sorted(pbp.columns.tolist()))

# ── Step 2: unique values in candidate target columns ─────────────────────────
print(SEP)
print("STEP 2 — Unique values in 'event_type' (used by BASES_MAP)")
print(SEP)
if "event_type" in pbp.columns:
    print(pbp["event_type"].value_counts(dropna=False).to_string())
else:
    print("  !! 'event_type' column NOT FOUND in PBP")

print()
print("STEP 2b — Unique values in 'play_result' (used for PA-end filter)")
print(SEP)
if "play_result" in pbp.columns:
    print(pbp["play_result"].value_counts(dropna=False).to_string())
else:
    print("  !! 'play_result' column NOT FOUND in PBP")

# Check every column whose name might hold the PA result
print()
print("STEP 2c — Any other columns containing 'event' or 'result' in name:")
candidates = [c for c in pbp.columns if any(k in c.lower() for k in ("event", "result", "outcome", "type"))]
for col in candidates:
    uniq = pbp[col].dropna().unique()
    overlap = set(str(v).lower() for v in uniq) & set(BASES_MAP.keys())
    print(f"  {col:40s}  nunique={pbp[col].nunique():4d}  BASES_MAP_matches={len(overlap)}")

# ── Step 3: reproduce _extract_sp_pa ──────────────────────────────────────────
print(SEP)
print("STEP 3 — Reproduce _extract_sp_pa (last event per play_id)")
print(SEP)

pa_raw = (
    pbp
    .sort_values(["gamepk", "play_id", "event_index"])
    .groupby(["gamepk", "play_id"], sort=False)
    .last()
    .reset_index()
)
print(f"Rows after groupby.last(): {len(pa_raw)}")
print(f"play_result non-null: {pa_raw['play_result'].notna().sum()}")
pa_filtered = pa_raw[pa_raw["play_result"].notna()].copy()
print(f"Rows after play_result notna filter: {len(pa_filtered)}")

# ── Step 4: check what event_type looks like AFTER groupby.last() ─────────────
print(SEP)
print("STEP 4 — 'event_type' values in PA-level rows (after groupby.last + play_result filter)")
print(SEP)
if "event_type" in pa_filtered.columns:
    vc = pa_filtered["event_type"].value_counts(dropna=False)
    print(vc.to_string())
    mapped = pa_filtered["event_type"].map(BASES_MAP)
    print(f"\n  Mapped non-null: {mapped.notna().sum()} / {len(mapped)}")
    print(f"  All null/zero: {(mapped.fillna(0) == 0).all()}")
else:
    print("  !! 'event_type' NOT FOUND after collapse")

print()
print("STEP 4b — 'play_result' values in PA-level rows:")
if "play_result" in pa_filtered.columns:
    vc2 = pa_filtered["play_result"].value_counts(dropna=False)
    print(vc2.to_string())
    mapped2 = pa_filtered["play_result"].map(BASES_MAP)
    print(f"\n  Mapped non-null: {mapped2.notna().sum()} / {len(mapped2)}")
    print(f"  Reached-base count if using play_result: {(mapped2.fillna(0) > 0).sum()}")
else:
    print("  !! 'play_result' NOT FOUND after collapse")

# ── Step 5: check if BASES_MAP keys match actual values case-insensitively ─────
print(SEP)
print("STEP 5 — Case / whitespace check for event_type vs BASES_MAP keys")
print(SEP)
if "event_type" in pa_filtered.columns:
    uniq_et = pa_filtered["event_type"].dropna().unique()
    for v in sorted(uniq_et):
        exact  = v in BASES_MAP
        lower  = str(v).lower().strip() in BASES_MAP
        print(f"  {str(v):40s}  exact={exact}  lower_strip={lower}")

# ── Step 6: apply BASES_MAP to every candidate column ─────────────────────────
print(SEP)
print("STEP 6 — Try BASES_MAP on all candidate columns to find the right one")
print(SEP)
for col in candidates:
    mapped = pa_filtered[col].map(BASES_MAP)
    n_nonzero = (mapped.fillna(0) > 0).sum()
    n_mapped  = mapped.notna().sum()
    pct = n_nonzero / max(len(pa_filtered), 1) * 100
    print(f"  {col:40s}  mapped={n_mapped:4d}  reached_base={n_nonzero:4d}  pct={pct:.1f}%")

# ── Step 7: sample a few rows for visual inspection ───────────────────────────
print(SEP)
print("STEP 7 — Sample 10 PA rows (all candidate columns)")
print(SEP)
display_cols = ["play_id", "event_index"] + candidates
display_cols = [c for c in display_cols if c in pa_filtered.columns]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(pa_filtered[display_cols].head(10).to_string(index=False))

print(SEP)
print("DONE")
print(SEP)
