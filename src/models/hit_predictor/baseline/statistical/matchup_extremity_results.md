# Matchup-Extremity Slice Check — batter_hit_predictor

**Date:** 2026-08-23  
**Question:** does the shrinkage baseline (and reality) actually predict near-zero hit probability for a weak batter facing a dominant pitcher, once pooled across every such PA in the val season (not read off one game)?  
**Val season:** 2024 — 173,067 PA rows, 34,193 dropped (missing last_season_ba or pitcher_last_season_pa_hit_rate)  

## Batter strength × pitcher dominance grid

| Batter | Pitcher | N | Obs hit rate | Mean pred (shrinkage) | Reliable (n≥200) |
|--------|---------|---|--------------|------------------------|-------------------|
| weak | dominant | 17,925 | 0.202 | 0.213 | True |
| weak | avg | 17,293 | 0.206 | 0.213 | True |
| weak | weak | 17,000 | 0.222 | 0.213 | True |
| avg | dominant | 13,722 | 0.203 | 0.235 | True |
| avg | avg | 13,454 | 0.219 | 0.234 | True |
| avg | weak | 13,534 | 0.234 | 0.235 | True |
| strong | dominant | 15,260 | 0.219 | 0.254 | True |
| strong | avg | 15,285 | 0.232 | 0.255 | True |
| strong | weak | 15,401 | 0.242 | 0.256 | True |

## Setup

- batter axis: last_season_ba (same feature the shrinkage baseline shrinks toward)
- pitcher axis: pitcher_last_season_pa_hit_rate (role-tagged: sp per-individual, bullpen pooled by team), joined via REALIZED pitcher_role/pitcher_id -- a post-hoc diagnostic, not a production feature, so no point-in-time leakage concern
- n_bins = 3, min_n = 200
