# MLB Stats & Feature Glossary

A reference for feature engineering in `hit_predictor`: what each metric measures, how it's calculated, why it exists, and — specific to this repo — whether it's computable today from `raw_data`/`processed_data` or requires new ingestion.

Organized in layers: raw tracking data first, then metrics derived from it, then value/rate stats built on top of those. Read roughly top-to-bottom if you're new to this domain.

**How to read the "In this repo" column:**
- ✅ **have it** — column already exists in `schema.py` (`PBP`) or `preprocessing.py` (`BATTER_BOXSCORE_COLUMNS` / `PITCHER_BOXSCORE_COLUMNS` / `PLAYER_INFO_COLUMNS` / `GAME_INFO_COLUMNS`)
- ⚠️ **partial** — computable with a caveat (missing a component column, needs approximation, or needs external context)
- ❌ **gap** — not in the current schema at all; would need new raw ingestion, not just a new feature function

---

## 1. Raw Statcast Tracking Data (the atoms)

Everything else is built from these. Statcast (MLB's camera/radar tracking system) captures these directly — nothing is "calculated" here, it's measured.

| Metric | What it measures | Why it matters as a feature | In this repo |
|---|---|---|---|
| **Exit Velocity (EV)** | Speed of the ball off the bat (mph) at contact | Single strongest predictor of batted-ball outcome quality — hard-hit balls become hits/XBH far more often regardless of luck | ✅ `launch_speed` |
| **Launch Angle (LA)** | Vertical angle the ball leaves the bat (0° = parallel to ground, negative = grounder) | Combined with EV, determines grounder/liner/fly-ball/pop-up. 105mph at −20° is a routine grounder — EV alone is an incomplete story | ✅ `launch_angle` |
| **Spin Rate** | RPM of the pitch as it leaves the hand | Higher spin four-seamers = more perceived "rise" (less drop), inducing whiffs up in the zone; higher spin breaking balls = sharper/later break | ✅ `spin_rate` |
| **Spin Axis / Direction** | Tilt/orientation of the spin (clock-face notation) | Determines *which direction* the ball moves — two pitches with identical spin rate can move completely differently by axis | ✅ `spin_direction` |
| **Pitch Velocity** | Speed of the pitch (mph), out of hand and at plate | Baseline stuff quality; also the input for velocity-differential-between-pitch-types deception features | ✅ `start_speed`, `end_speed` |
| **Extension** | Distance (ft) from the rubber to release point | Releasing closer to the plate gives hitters less reaction time — makes a given velocity "play up" | ✅ `extension` |
| **Sprint Speed** | Max run speed (ft/sec) on a competitive run | Proxy for raw speed independent of stolen-base totals/instincts — feeds defense/baserunning value models | ❌ not in `pbp` or box score — would need a separate Statcast sprint-speed pull |
| **Route Efficiency / OAA inputs** | Distance covered vs. optimal path, reaction time, burst | Raw inputs to Outs Above Average (below) | ❌ not tracked in current ingestion |
| **Release Point (x, y, z)** | 3D coordinates of ball release | Used to detect pitch tunneling (multiple pitch types released from the same point/trajectory to deceive hitters) | ✅ `release_pos_x/y/z` |
| **Plate Location (px, pz)** | Where the pitch crosses the front of the plate | Basis for zone-based features (edge%, chase rate, called-strike probability) | ✅ `plate_x`, `plate_z`, `zone` |
| **Bat Speed** | Swing speed measured ~6in from the bat head (the "sweet spot"), at load-bearing point in the swing | Describes the swing itself rather than the outcome — a newer (2023+) layer of hitter skill data | ❌ gap — Statcast bat-tracking, not in current schema |
| **Swing Length** | Total distance traveled by the bat head from swing start to contact | Shorter/more direct swings vs. longer, more power-oriented swings | ❌ gap — same bat-tracking source as above |
| **Attack Angle** | Vertical direction of the bat's sweet spot at impact | Complements pitch launch angle — how "matched" the swing plane is to the pitch | ❌ gap |

---

## 2. Derived Statcast Metrics (built from the raw data)

### Barrel Rate
- **What it measures:** rate of batted balls hit with the EV/LA combination that historically produces elite outcomes (~98mph+ EV at 26°–30° LA, with the qualifying angle range widening as EV increases).
- **Why calculated this way:** empirically defined — the EV/LA zone that historically produced a **.500+ AVG and 1.500+ SLG**. Not a fixed cutoff; it's a curve, because higher EV tolerates a wider angle range.
- **Feature use:** Barrel% (barrels / batted-ball events) is one of the most predictive, *sticky* (year-over-year stable) contact-quality metrics — more stable than batting average.
- **In this repo:** ⚠️ have `launch_speed` + `launch_angle`, so an approximation is computable now; the real MLB definition uses a full speed/angle curve, not a single threshold.

### Expected Batting Average (xBA)
- **What it measures:** the AVG a ball in play "should" produce given its EV, LA (and, in some versions, sprint speed on weakly-hit balls).
- **How calculated:** a model trained on historical outcomes of every batted ball with similar EV/LA — "given 1,000 balls hit like this, what fraction became hits historically?"
- **Why it exists:** isolates quality-of-contact from defense positioning, ballpark, and luck (a well-struck ball right at a fielder = out; a bloop = hit).
- **In this repo:** ❌ gap — this is a model output, not a raw column. Would need to check whether the raw MLB Stats API pull exposes Baseball Savant's `estimated_ba_using_speedangle` field; if not, out of scope until that model exists.

### Expected Slugging (xSLG) and Expected wOBA (xwOBA)
- **What they measure:** same idea as xBA, incorporating expected extra-base value (xSLG) or expected overall offensive value including walks/HBP (xwOBA).
- **Why:** xwOBA is generally the best single "true talent" hitting metric — strips out defense/luck/park while still weighting outcomes correctly.
- **Feature use:** excellent for identifying regression candidates (actual vs. expected divergence).
- **In this repo:** ❌ gap, same reason as xBA.

### Hard-Hit Rate
- **What it measures:** % of batted balls with EV ≥ 95mph.
- **Why this cutoff:** empirically the threshold where outcomes start meaningfully improving — a simpler, single-variable cousin of Barrel% that ignores launch angle.
- **In this repo:** ✅ implemented — `_create_batter_in_play_contact_stats` → `batter_season_contact_hard_hit_rate`. Uses the raw `launch_speed >= 95` threshold with nulls excluded from the denominator (not counted as "not hard hit"). More precise than the pitcher-side implementation (`_create_pitcher_contact_quality_stats`), which still uses the categorical `hardness == 'Hard'` field instead of the raw threshold.

### Sweet-Spot %
- **What it measures:** % of batted balls with launch angle 8°–32° (well-struck line drives/fly balls), regardless of EV.
- **Why:** isolates the *angle* component of contact quality separately from *speed* — hitting hard and hitting at a good angle are somewhat independent skills.
- **In this repo:** ✅ implemented for batters — `_create_batter_in_play_contact_stats` → `batter_season_contact_sweet_spot_rate` (inclusive 8–32° range, nulls excluded from denominator). Not yet implemented on the pitcher side.

### Whiff Rate / Chase Rate (O-Swing%) / Z-Swing% / Contact%
- **Whiff%** = swings-and-misses / total swings — bat-missing "stuff" for pitchers, swing-and-miss tendency for hitters. *Not currently implemented* — what's built instead is **SwStr%** (below), a related but distinct denominator.
- **SwStr% (swinging-strike rate)** = swings-and-misses / **all pitches seen** (not swings-only) — the standard FanGraphs definition. ✅ implemented — `_create_batter_plate_discipline_stats` → `batter_season_swinging_strike_rate`.
- **Chase Rate (O-Swing%)** = swings at pitches outside the zone / **pitches outside the zone**. ⚠️ implemented with a known denominator gap — `_create_batter_plate_discipline_stats` → `batter_season_o_swing_rate` computes `is_chase.mean()` over **all pitches**, not pitches-outside-zone only. Inherited unchanged from the pre-existing pitcher version (`_create_pitcher_stuff_command_stats` → `command_chase_rate`), which has the same gap. Two batters with identical true chase discipline can show different `o_swing_rate` values purely because one saw more out-of-zone pitches than the other. Flagged, not yet fixed — deferred pending a decision on whether to also change the already-shipped pitcher column.
- **Z-Swing%** = swings at pitches inside the zone / **pitches inside the zone**. Same denominator gap as O-Swing% above — `_create_batter_plate_discipline_stats` → `batter_season_z_swing_rate` also divides by all pitches, not zone-filtered ones.
- **Zone%** = pitches inside the zone / all pitches seen — this one *is* correctly defined over all pitches by convention (no gap). ✅ implemented — `batter_season_zone_rate`.
- **Contact%** = contact made (foul or in-play) / **swings only** — distinct from SwStr%'s all-pitches denominator. ✅ implemented — `batter_season_contact_rate`, filtered to `is_swing == True` before aggregating.
- **Why zone-filtering matters for O-Swing/Z-Swing specifically:** conflates two different skills if not filtered — aggression inside the zone (good) vs. chasing outside it (bad) — the whole point of these stats is to isolate discipline as independent of how often the pitcher even throws in the zone.
- **In this repo, underlying flags:** `is_swinging_strike`, `is_chase`, `is_zone_swing`, `is_swing` are computed once in `pipeline.py`'s `_initial_pbp_processing` and reused by both pitcher and batter build functions. All three of `is_chase`/`is_zone_swing`/`is_swinging_strike` had real correctness bugs fixed this session (see `tests/hit_predictor/test_pipeline.py`): `is_chase` was gated on whiffs only (missed fouls/in-play swings outside the zone), `is_zone_swing` string-matched a literal `'In Play'` that never appears in real API data, and `is_swinging_strike` was missing `'Missed Bunt'`.

### Outs Above Average (OAA)
- **What it measures:** defensive value — outs converted above/below what an average fielder would given play difficulty (distance, time, direction).
- **How calculated:** each batted ball gets a "catch probability" from historical outcomes of similarly-difficult plays; OAA = sum of (actual outcome − catch probability).
- **Why this approach:** older defensive metrics (fielding%, range factor) ignore play difficulty/positioning; OAA normalizes for it.
- **In this repo:** ❌ gap — no fielder positioning/route data ingested; out of scope for `hit_predictor` (offense-focused) regardless.

### Catcher Framing (Strike Rate Above Average)
- **What it measures:** a catcher's ability to get borderline pitches called strikes relative to an average catcher in the same location/count/umpire.
- **Why it matters:** real, sticky year-over-year value invisible to traditional catcher defense stats.
- **In this repo:** ❌ gap — would need catcher identity joined to pbp plus an umpire-zone model; not currently modeled.

---

## 3. Sabermetric Rate & Value Stats (context/value layer)

Built from event outcomes (walks, hits, etc.), re-weighted or park/league-adjusted to be more meaningful than raw counting stats.

### wOBA (Weighted On-Base Average)
- **What it measures:** overall offensive contribution on an OBP-like scale, correctly weighting each outcome by its actual run value.
- **How calculated (simplified):**
  `wOBA = (0.69×uBB + 0.72×HBP + 0.89×1B + 1.27×2B + 1.62×3B + 2.10×HR) / (AB + BB − IBB + SF + HBP)`
  (weights re-derived each season via linear weights / run expectancy tables)
- **Why this way:** SLG assigns arbitrary weights (a HR is exactly "4x" a single, which isn't how HRs actually add run value). wOBA weights come from measured run-expectancy impact.
- **In this repo:** ⚠️ `BATTER_BOXSCORE_COLUMNS` has `singles`, `doubles`, `triples`, `hr`, `bb`, `ab` — but no `hbp`, `ibb`, or `sf` columns. Computable with fixed published weights as a v1 approximation; a season-specific-weight version needs those additional counting fields.

### wRC+ (Weighted Runs Created Plus)
- **What it measures:** overall offensive value, park- and league-adjusted, indexed to 100 = league average.
- **Why indexed this way:** wRC+ of 120 = "20% better than league-average, accounting for home park." Makes it comparable across players/eras/parks.
- **In this repo:** ❌ gap — needs league-average wOBA and park factors as *additional* context beyond one player's own rows. Real scope increase, not a column-mapping exercise; defer until park factors exist.

### FIP (Fielding Independent Pitching) / xFIP / SIERA
- **FIP:** estimates ERA from outcomes the pitcher fully controls — BB, K, HBP, HR — excluding balls in play (defense/luck-influenced).
  `FIP = ((13×HR + 3×(BB+HBP) − 2×K) / IP) + constant`
- **xFIP:** same as FIP but replaces actual HR with an *expected* HR total from league-average HR/FB rate — reduces noise from HR/FB variance (a stat with a lot of year-to-year luck).
- **SIERA:** regression-based ERA estimator that also accounts for batted-ball type (GB/LD/FB rates) and situational factors.
- **Why this family exists:** ERA is heavily influenced by team defense, sequencing, and BABIP luck. FIP-family stats isolate the pitcher's actual skill contribution.
- **In this repo:** ✅ FIP is already implemented in `_create_pitcher_pa_outcome_stats` (`fip_hr`, `fip_bb`, `fip_k`, constant `C = 3.10`). xFIP/SIERA are ❌ gap — need league-average HR/FB rate (xFIP) and batted-ball-type regression coefficients (SIERA), neither currently computed.

### BABIP (Batting Average on Balls In Play)
- **What it measures:** AVG excluding home runs and strikeouts — how often balls in play become hits.
- **Formula:** `BABIP = (H − HR) / (AB − K − HR + SF)`
- **Why it's a feature:** league-average BABIP is fairly stable (~.290–.300); a hitter/pitcher far outside that range (absent a persistent-talent explanation like elite speed or Barrel%) is often a regression signal.
- **In this repo:** ⚠️ implemented as an approximation — `_create_boxscore_batter_stats` → `batter_season_babip` (zero-denominator-guarded). Missing `sf` (sac flies) from `BATTER_BOXSCORE_COLUMNS`, so the SF adjustment in the formula above is omitted.

### ISO (Isolated Power)
- **What it measures:** raw power, separate from average.
- **Formula:** `ISO = SLG − AVG` (extra bases per at-bat)
- **Why:** SLG conflates "hits the ball a lot" with "hits for power"; ISO isolates power alone.
- **In this repo:** ✅ implemented — `_create_boxscore_batter_stats` → `batter_season_iso` (`batter_season_slg − batter_season_ba`, zero-AB-guarded).

### K% and BB%
- **What they measure:** strikeouts/walks as % of plate appearances, not raw totals.
- **Why rate over count:** removes playing-time bias — 500 PA vs. 300 PA naturally produces more Ks; K% lets you compare skill directly.
- **In this repo:** ✅ implemented on both sides, with different (correct-for-each-role) denominators — pitcher side: `pitcher_season_k_rate`/`pitcher_season_bb_rate` (denominator `ip`, since a pitcher's workload is measured in innings). Batter side: `_create_batter_pa_outcome_stats` → `batter_season_pa_strikeout_rate`/`batter_season_pa_walk_rate` (denominator PA, the conventional batter-side denominator, via a last-pitch-of-PA dedup on `(gamepk, play_id)`).

### WAR (Wins Above Replacement) — fWAR vs. bWAR
- **What it measures:** total player value (offense + defense + baserunning + positional adjustment + playing time) in wins above a replacement-level player.
- **Why it differs between sources:** fWAR uses FIP-based pitching value and wOBA-based batting value; bWAR uses runs-allowed-based pitching value and win-probability-influenced batting. Different philosophy on what to hold a player accountable for.
- **In this repo:** ❌ gap — full WAR requires defensive value, baserunning value, positional adjustments, and replacement-level baselines, none of which exist here. Out of scope; pick sub-components (e.g. wOBA-based batting value) rather than chasing full WAR.

### Park Factors
- **What they measure:** how much a ballpark inflates/suppresses a specific outcome (runs, HR, doubles) vs. a neutral park, indexed to 100.
- **How calculated:** compare the outcome rate in-park vs. the same teams' rate on the road, over a multi-year rolling window, then normalize.
- **Why multi-year:** single-season park factors are noisy (weather, small sample); 3-year rolling windows are standard.
- **In this repo:** ❌ gap — would need venue identity per game (check whether `game_info` carries a venue/park field) plus multi-season aggregation; a prerequisite for wRC+.

---

## 4. Contextual / Environmental Features

| Feature | Why it's used | In this repo |
|---|---|---|
| **Weather (temp, wind speed/direction)** | Air density affects carry distance; wind direction relative to outfield orientation adds/subtracts HR distance; warmer air carries further | ✅ `weather_temp`, `weather_wind_speed`, `weather_wind_direction` already parsed in `process_game_info` (`pipeline.py`) |
| **Umpire tendencies (zone size, consistency)** | Some umpires call a measurably larger/smaller or less consistent zone, shifting effective K%/BB% for that game | ❌ gap — no umpire identity ingested |
| **Days of rest / bullpen usage (fatigue proxies)** | Recent pitch counts, back-to-back appearances, rest days are leading indicators of stuff decline or injury risk | ⚠️ `pbp` has per-game pitch counts (`_create_pitcher_pitch_count_stats` already computes these); day-of-rest itself needs a date-diff against the player's prior appearance, not yet computed |
| **Platoon splits (vs. L/R)** | Nearly every hitter/pitcher performs meaningfully differently vs. same- vs. opposite-handed opponents — a near-universal matchup feature | ✅ implemented — see §5B |
| **Batted-ball spray angle / pull%** | Combined with EV/LA, indicates whether a power profile is exploitable by shifting/positioning | ⚠️ have `hit_coord_x`/`hit_coord_y` — spray angle is computable from those, not yet implemented |

---

## 5. Batter Feature Status — Implemented

All batter season-stat categories originally scoped in this section are now built in `src/models/hit_predictor/processing/features/season_stats.py`, entry point `build_pbp_batter_feats(pbp)` (pbp-derived) plus `build_batter_stats(batter_boxscore)` (box-score-derived). Every function follows the point-in-time-safe pattern: raw stats are computed under a `{stat}` name, then shifted forward one season and renamed to `last_{stat}` by `_shift_to_last_season` at the top-level `build_*` call only — so a season's stats can only be joined onto *next* season's games.

**Feature → function lookup** (search `season_stats.py` for the function name to see the exact aggregation):

| Category | Representative columns (post-shift, e.g. via `build_pbp_batter_feats`) | Function |
|---|---|---|
| Traditional box-score rates | `batter_last_season_ba`, `_slg`, `_iso`, `_babip` | `build_batter_stats` → `_create_boxscore_batter_stats` |
| PA outcome rates | `batter_last_season_pa_strikeout_rate`, `_walk_rate`, `_hit_rate`, `_hr_rate`, `_single_rate`, `_xbh_rate`, `_hbp_rate` | `_create_batter_pa_outcome_stats` |
| Plate discipline | `batter_last_season_o_swing_rate`, `_z_swing_rate`, `_swinging_strike_rate`, `_zone_rate`, `_contact_rate` | `_create_batter_plate_discipline_stats` |
| In-play contact quality | `batter_last_season_contact_hard_hit_rate`, `_sweet_spot_rate`, `_gb_rate`, `_fb_rate`, `_ld_rate`, `_avg_launch_speed`, `_avg_launch_angle` | `_create_batter_in_play_contact_stats` |
| Foul-ball contact | `batter_last_season_foul_rate`, `_contact_foul_rate` | `_create_batter_foul_contact_stats` |
| Two-strike foul rate | `batter_last_season_two_strike_foul_rate` | `_create_batter_two_strike_foul_stats` |

**Known gaps in what's implemented** (documented inline above, not fixed): `o_swing_rate`/`z_swing_rate` use an all-pitches denominator rather than the standard zone-filtered one (see §2, Whiff/Chase/Z-Swing section); BABIP omits the `sf` adjustment (missing column); wOBA/wRC+ remain unbuilt (§3).

**Everything in the ❌ gap column across §1–§4** (xBA/xwOBA, wRC+, park factors, bat tracking, WAR, OAA, catcher framing, sprint speed) remains explicitly deferred — each needs either a new model, new context data, or new raw ingestion, not just a new feature function.

---

## 5B. Platoon (Handedness) Splits — Implemented

Every stat category from §5 also exists split by opponent handedness — the standard sabermetric platoon split: a batter's season split by the *pitcher's* throwing hand, a pitcher's season split by the *batter's* hand. Plumbing: `pipeline.py`'s `_add_pbp_handedness(df, player_info)` merges two static, per-player columns onto every pbp row — `pitcher_throw_hand` (`pitcher_id` → `player_info.pitchHand`, 'L'/'R') and `batter_bat_side` (`batter_id` → `player_info.batSide`, 'L'/'R'/'S'). `build_pbp_features` now takes `player_info` as a required argument.

**Batter vs. pitcher hand** — `build_pbp_batter_feats_by_pitcher_hand(pbp)`. Same 5 categories as §5, each `_create_batter_*` helper grouped with `pitcher_throw_hand` added (via a new optional `extra_group_cols` param, default `None` — existing callers unaffected), then pivoted wide with `_pivot_by_hand` so a batter/season row has both `batter_last_season_vs_lhp_pa_hit_rate` and `batter_last_season_vs_rhp_pa_hit_rate` side by side, rather than one row per hand. `pitcher_throw_hand` is static per pitcher, so there's no switch-hitter-style ambiguity on this side.

**Pitcher vs. batter hand** — `build_pbp_pitcher_feats_by_batter_hand(pbp, pitcher_role=None)`. Same call signature/role filter as `build_pbp_pitcher_feats` (stacks with `pitcher_role`: sp/bullpen × 3 hand buckets = 6 combos). **Three buckets, not two** — `vs_lhb` / `vs_rhb` / `vs_switch` — because `batter_bat_side` already distinguishes switch-hitters (API code `'S'`) from `player_info`, at zero extra ingestion cost. Switch-hitters get their own bucket rather than being forced into L or R: they aren't a data-quality compromise, they're arguably the *more correct* modeling choice, since a switch-hitter almost always bats opposite-handed to the pitcher specifically to keep the platoon advantage — so "pitcher vs. switch-hitter" is a genuinely distinct population (a pitcher essentially never gets the same-handed advantage against one), not a fuzzy blend of the L/R buckets.

**League-wide context table** — `build_league_handedness_stats(pbp)`. Same 5 categories, but pooled across *every* player: one row per `(game_season, pitcher_throw_hand, batter_bat_side)` — 2×3 = 6 rows/season, no `batter_id`/`pitcher_id`. Exists because a single player's PAs against their less-frequently-faced hand can be as thin as ~75–150 PA in one season — too noisy to trust alone even though platoon splits are comparatively stable. This table is the stable, high-sample-size companion sitting alongside the noisier per-player splits above; no shrinkage/blending is done automatically — both levels are shipped as separate columns and left for the model to weigh.

All three follow the same point-in-time-safe `_shift_to_last_season` pattern as everything else in §5.

**Known limitation:** per-PA batter handedness isn't available for switch-hitters mid-game — `player_info.batSide` is a static per-player attribute, correct for the ~85–90% of batters who don't switch, but the raw MLB API's per-play `matchup.batSide` (which would resolve exactly which side a switch-hitter batted from in a specific PA) isn't captured by `fetch_playbyplay_data` in `src/data/modules/fetchers.py`. This doesn't affect the splits documented above (they only need the *pitcher's* hand, which is static and reliable, or the batter's *bucket* — L/R/S — not the exact side used in a specific PA) — it would only matter for a future, finer-grained feature that needed a switch-hitter's actual per-PA side. Not pursued; would require adding `batSide`/`pitchHand` to `PLAYBYPLAY_COLUMNS` plus a backfill of already-ingested raw playbyplay data.

---

## 6. Practical Notes for Feature Engineering

- **Stability vs. sample size:** Barrel%, K%, BB%, and Hard-Hit% stabilize (become self-predictive) much faster than BA or BABIP. On partial-season data, prefer the "sticky" metrics as leading features and treat outcome-based rate stats (AVG, ERA) with more caution.
- **xStats vs. actual stats as features:** using both the actual and expected version of a stat (once xwOBA exists here) lets a model implicitly capture "performing above/below expected skill" — a regression-candidate signal.
- **Avoid leakage — repo-specific:** every `build_*` function in `season_stats.py` handles this via `_shift_to_last_season`, which shifts `game_season` forward one year and renames any column containing `season_` to `last_season_` (e.g. `batter_season_ba` → `batter_last_season_ba`) so a season's stats can only be joined onto *next* season's games, never the same one. Any new build function must follow the same pattern — shift once, at the top-level `build_*` entry point, not inside a private `_create_*` helper.
- **Data sources:** Baseball Savant (Statcast layer, sections 1–2), FanGraphs (sabermetric layer, section 3, easiest CSV export), Retrosheet (raw play-by-play if computing from scratch). This repo's own raw data comes from the MLB Stats API (`raw_data/games/*`, `raw_data/playbyplay/*`) — confirm which Savant/FanGraphs fields, if any, are already present in that pull before assuming a ❌ gap actually requires a new source.

---

*Compiled as a feature-engineering reference for `hit_predictor`. Cross-check the "In this repo" column against `src/models/hit_predictor/processing/schema.py` and `src/data/modules/preprocessing.py` if either file changes — this glossary reflects their shape as of the completed batter + pitcher season-stats implementation in `season_stats.py`. `season_stats.py` itself points back here in a header comment — the two files are meant to be read together, not duplicated.*
