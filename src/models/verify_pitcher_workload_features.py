#!/usr/bin/env python3
"""
verify_pitcher_workload_features.py
====================================
Incremental verification of workload / fatigue features added to _pitcher_rolling.
Runs on 2023 only for speed.

Usage:
    python verify_pitcher_workload_features.py
"""
import pandas as pd

YEAR = 2023

from build_pa_pitcher_features import (  # noqa: E402  (same-dir import, run from src/models)
    _read_pbp,
    _read_pitcher_box,
    _identify_starters,
    _pbp_pitcher_game_aggs,
    _pbp_pitcher_inn45_aggs,
    _build_pitcher_game,
    _pitcher_rolling,
)

print(f"Loading {YEAR} data …")
pbp         = _read_pbp(YEAR)
pitcher_box = _read_pitcher_box(YEAR)
pitcher_box["game_date"] = pd.to_datetime(pitcher_box["game_date"])

starters    = _identify_starters(pbp)
pbp_aggs    = _pbp_pitcher_game_aggs(pbp)
inn45_aggs  = _pbp_pitcher_inn45_aggs(pbp)

pitcher_game = _build_pitcher_game(pitcher_box, pbp_aggs, starters)
sp_rolling   = _pitcher_rolling(pitcher_game, inn45_aggs=inn45_aggs)

print(f"sp_rolling shape: {sp_rolling.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — Last start features
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Last start features ===")
print(sp_rolling.filter(like='sp_last_start').describe())
print(f"Null rate: {sp_rolling.filter(like='sp_last_start').isna().mean().mean():.3f}")

test_pitcher = sp_rolling[sp_rolling['personId'] == 543037].sort_values('game_date')
print("\n=== Gerrit Cole — last start features ===")
print(test_pitcher[['game_date', 'sp_last_start_k_rate', 'sp_last_start_ip',
                     'sp_last_start_pitches']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — Workload features
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Workload features ===")
print(sp_rolling[['sp3_avg_ip', 'sp10_avg_ip', 'spytd_avg_ip']].describe())
print(sp_rolling[['sp3_avg_pitches', 'sp10_avg_pitches', 'sp3_total_pitches', 'sp10_total_pitches']].describe())

print("\n=== Gerrit Cole — workload features ===")
print(test_pitcher[['game_date', 'sp3_avg_ip', 'sp3_avg_pitches',
                     'sp3_total_pitches', 'days_rest',
                     'fatigue_score']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — Spacing and fatigue
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Fatigue features ===")
print(sp_rolling[['sp_avg_days_between_starts_3', 'sp_avg_days_between_starts_10', 'fatigue_score']].describe())
print(f"fatigue_score nulls: {sp_rolling['fatigue_score'].isna().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# Full verification
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Full verification ===")
new_cols = (
    sp_rolling.filter(like='sp_last_start').columns.tolist()
    + sp_rolling.filter(like='avg_ip').columns.tolist()
    + sp_rolling.filter(like='avg_pitches').columns.tolist()
    + sp_rolling.filter(like='total_pitches').columns.tolist()
    + ['fatigue_score', 'sp_avg_days_between_starts_3', 'sp_avg_days_between_starts_10']
)
print(f"New columns added: {len(new_cols)}")
print(f"Any unexpected nulls:\n{sp_rolling[new_cols].isna().mean().sort_values(ascending=False).head(5)}")

print("\n# Sanity check correlations — fatigue score should correlate with pitch count")
print(sp_rolling[['fatigue_score', 'sp_last_start_pitches', 'days_rest']].corr())

for col in ['sp3_k_rate', 'sp10_k_rate', 'spytd_k_rate', 'sp3_avg_ip', 'sp10_avg_ip']:
    assert col in sp_rolling.columns, f"MISSING: {col}"
print("\nAll existing sp3_*, sp10_*, spytd_* spot-checks passed.")
