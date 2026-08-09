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
- **In this repo:** ⚠️ implemented as a v1 approximation — `processing/features/park_factors.py` → `build_park_factors(schedule, batter_boxscore)`. `venue_id` comes from `schedule` (`SCHEDULE_COLUMNS`), joined to `batter_boxscore` via `gamepk` to compute each venue's hit rate (h/ab) for a season relative to the league-wide hit rate that season — `> 1` = hitter-friendly, `< 1` = pitcher-friendly. This is a single-season index (trailing 1 year, `_shift_to_last_season`-shifted like every other season feature here), **not** the standard 3-year rolling window described above, and it's a single hit-rate factor rather than separate runs/HR/2B components or the home-vs-road-split methodology — both are real simplifications, deferred as future work. Still a prerequisite for wRC+ once that's built.

---

## 4. Contextual / Environmental Features

| Feature | Why it's used | In this repo |
|---|---|---|
| **Weather (temp, wind speed/direction)** | Air density affects carry distance; wind direction relative to outfield orientation adds/subtracts HR distance; warmer air carries further | ✅ `weather_temp`, `weather_wind_speed`, `weather_wind_direction` already parsed in `process_game_info` (`pipeline.py`) |
| **Umpire tendencies (zone size, consistency)** | Some umpires call a measurably larger/smaller or less consistent zone, shifting effective K%/BB% for that game | ❌ gap — no umpire identity ingested |
| **Days of rest / bullpen usage (fatigue proxies)** | Recent pitch counts, back-to-back appearances, rest days are leading indicators of stuff decline or injury risk | ⚠️ `pbp` has per-game pitch counts (`_create_pitcher_pitch_count_stats` already computes these); day-of-rest itself needs a date-diff against the player's prior appearance, not yet computed |
| **Times through the order (TTOP)** | Batters gain a real, well-documented advantage each additional time they face the same starter in a game (Lichtman) | ✅ implemented in two complementary pieces. **(1) Per-PA gating** — `expected_times_through_order`, `processing/features/expected_role.py` → `assign_expected_pitcher_role`. An earlier version derived this from the *realized* `batter_pa_number`/`pitcher_role` for that PA, but that's only knowable after the fact (whether the starter is still in by a batter's 2nd/3rd PA depends on real-time pitch count/performance/manager decisions) — a leak for a model meant to predict before the game. The fixed version compares the batter's estimated position in the lineup (`estimated_team_pa_position`, from pre-game-knowable `batting_order`) against *that specific starter's own historical average depth* (`pitcher_last_season_start_avg_batters_faced_per_start`, `season_stats.py` → `build_pitcher_start_depth_stats`, with a league-wide fallback for pitchers with no prior-season starts — `build_league_avg_start_depth`). `expected_times_through_order = min(batter_pa_number, 3)` when the estimate says still-facing-the-starter, `NaN` otherwise. Applied identically to historical training rows, not just future ones, to avoid train/serve skew. Doesn't handle openers/double-switches, same documented limitation as `pitcher_role` itself. **(2) Historical split stats** — `season_stats.py` → `build_pbp_pitcher_feats_by_times_through_order` (per-pitcher, PA-outcome category only) and `build_league_times_through_order_stats` (population-level companion for the thin per-pitcher tto3plus sample, same no-shrinkage-just-ship-both-levels approach as §5B's handedness splits — includes the per-bucket PA count, e.g. `pitcher_last_season_tto3plus_pa_total`, alongside every rate column so a thin sample is visible, not hidden). Built from pbp's realized `times_through_order`/`pitcher_role` — correct for *aggregating a pitcher's own true past performance*, unlike the leaky per-PA gating this replaced; the point-in-time-safety rule is about what's used to *decide which PA a stat applies to*, not what's used to *build* the stat from already-completed history. Attached unconditionally to every PA that pitcher throws as `'sp'` (mirroring how handedness splits attach both `vs_lhb`/`vs_rhb` regardless of the batter's actual side) rather than picked per-PA — the model combines these with `expected_times_through_order` itself to learn which bucket is relevant to a given PA. |
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

## 5C. Pitcher Feature Status — Implemented

Pitcher season stats are built in `season_stats.py`, entry point `build_pbp_pitcher_feats(pbp)` (pbp-derived, 5 categories — the pitcher-side counterpart to §5's `build_pbp_batter_feats`) plus `build_pitcher_stats(pitcher_boxscore)` (box-score-derived traditional rates, counterpart to `build_batter_stats`). Same point-in-time-safe `_shift_to_last_season` pattern as §5 throughout.

**Role-aware pooling** — `build_pbp_pitcher_feats_all_roles` and `build_pitcher_stats_all_roles` each build two variants and stack them: `sp` rows aggregated per individual `pitcher_id` (a starter's identity is known pre-game), `bullpen` rows pooled by `pitcher_team_id` (a specific reliever's identity isn't knowable pre-game — see §6's leakage note). Both halves are renamed onto a common `pitcher_key_id`/`pitcher_role` pair so `model_df` can join on that instead of `pitcher_id` alone; a swingman who both starts and relieves in the same season gets separate sp/bullpen rows rather than one blended aggregate.

**Feature → function lookup** (search `season_stats.py` for the function name to see the exact aggregation):

| Category | Representative columns (post-shift, e.g. via `build_pbp_pitcher_feats`) | Function |
|---|---|---|
| Traditional box-score rates | `pitcher_last_season_whip`, `_k_rate`, `_bb_rate`, `_strike_rate`, `_hr_rate` | `build_pitcher_stats` |
| Stuff (pitch characteristics) | `pitcher_last_season_stuff_start_speed_mean`/`_max`/`_std`, `_end_speed_mean`/`_max`, `_perceived_velo_mean`/`_max`, `_spin_rate_mean`/`_max`, `_movement_magnitude_mean`/`_max`, `_pfx_z_mean`/`_max`, `_extension_mean`/`_max`/`_std`, `_speed_retention_mean` | `_create_pitcher_stuff_command_stats` (via `build_pbp_pitcher_feats`) |
| Command (location/control) | `pitcher_last_season_command_in_play_rate`, `_swinging_strike_rate`, `_plate_x_std`, `_plate_z_normalized_std`, `_zone_rate`, `_ball_rate`, `_strike_rate`, `_called_strike_rate`, `_chase_rate`, `_zone_swing_rate`, `_first_pitch_strike_rate` | `_create_pitcher_stuff_command_stats` |
| PA outcome (incl. FIP) | `pitcher_last_season_pa_strikeout_rate`, `_walk_rate`, `_hbp_rate`, `_hit_rate`, `_hr_rate`, `_single_rate`, `_xbh_rate`, `_fip`, `_avg_final_balls`/`_strikes`, `_full_count_rate` | `_create_pitcher_pa_outcome_stats` |
| Last-inning-pitched | `pitcher_last_season_avg_last_inning`, `_std_last_inning`, `_avg_last_inning_velo`, `_last_inning_ball_rate`/`_strike_rate`, `_last_inning_avg_balls`/`_strikes`, `_last_inning_outs` | `_create_pitcher_last_inning_stats` |
| Pitch count (workload) | `pitcher_last_season_game_avg_pitch_count`, `_std_pitch_count`, `_max_pitch_count` | `_create_pitcher_pitch_count_stats` |
| In-play contact quality allowed | `pitcher_last_season_contact_hard_hit_rate`, `_gb_rate`, `_fb_rate`, `_ld_rate`, `_avg_launch_speed`, `_avg_launch_angle` | `_create_pitcher_contact_quality_stats` |

**Known gaps, cross-referenced from §2 (not re-explained here):** `command_chase_rate`/`command_zone_swing_rate` share batters' all-pitches-denominator gap (Whiff/Chase/Z-Swing section); `contact_hard_hit_rate` uses the categorical `hardness == 'Hard'` field rather than the raw `launch_speed >= 95` threshold, unlike the more precise batter-side version (Hard-Hit Rate section); pitcher contact quality has no sweet-spot-rate counterpart to the batter side (Sweet-Spot % section).

---

## 5D. Rolling-Window Features — Implemented

Every category in §5 and §5C also exists in a point-in-time **rolling** form — a value that updates game-by-game instead of once a season — built in `rolling_stats.py`. Same stat categories and formulas as `season_stats.py` throughout; `rolling_stats.py`'s own header comment points back to this glossary for definitions rather than duplicating them.

**Two window types, one naming convention:**
- `window='season'` → `{entity}_roll_season_{stat}` — expanding sum within `(entity_col, game_season)`, resets at every season boundary (the rolling analog of `_shift_to_last_season`'s season-level shift).
- `window=<int N>` → `{entity}_roll_last{N}g_{stat}` — trailing N-game sum, **carries across season boundaries** (a recent-form window has no reason to reset just because the calendar flipped to a new year). `experiments/v3_interaction_feats/train.py` sets `SHORT_WINDOW_GAMES = 10` for this N.

**Point-in-time safety rule:** every rolling column explicitly excludes the row's own game — `_rolling_sum`/`_rolling_max` compute `.cumsum().shift(1)` (season window) or `.rolling(window).sum().shift(1)` (int window) — so a rolling stat attached to a given game reflects only games strictly *before* it, never that game's own performance. This is enforced at every single game rather than once a year, making it a stricter version of the same no-leakage guarantee `_shift_to_last_season` gives at the season grain.

**Correctness rule (see `rolling_stats.py` module docstring):** roll counts, never rates — every rate is a numerator/denominator pair rolled separately as raw sums, and divided exactly once at the end (`_finalize_rates`). Averaging per-game rates directly would be wrong whenever games have different sample sizes (e.g. a 1-AB game and a 5-AB game shouldn't count equally toward a rolling AVG). Rolling `_std` columns (e.g. `stuff_start_speed_std`) go through `_rolling_pooled_std`, which derives a rolling sample std from rolled per-game `(n, sum, sum_of_squares)` triples — an exact rolling std can't be reconstructed from per-game std/mean alone.

**Entry points:**

| What it rolls | Entry point | Rolling equivalent of |
|---|---|---|
| Batter box-score rates (ba/slg/iso/babip) | `build_batter_rolling_stats(batter_boxscore, window)` | `build_batter_stats` |
| Pitcher box-score rates (whip/k_rate/bb_rate/strike_rate/hr_rate), role-aware | `build_pitcher_rolling_stats_all_roles(pitcher_boxscore, pbp, window)` | `build_pitcher_stats_all_roles` |
| Pitcher pbp-derived (stuff/command/PA-outcome+FIP/last-inning/pitch-count/contact-quality), role-aware | `build_pbp_pitcher_rolling_feats_all_roles(pbp, window)` | `build_pbp_pitcher_feats_all_roles` |
| Batter pbp-derived (PA-outcome/plate-discipline/in-play-contact/foul-contact/two-strike-foul) | `build_pbp_batter_rolling_feats(pbp, window)` | `build_pbp_batter_feats` |

Role-aware rolling functions pool bullpen rows per `(team_id, gamepk)` *before* rolling (a real bullpen outing routinely uses 2+ relievers in one game — rolling per individual reliever first would fan out and corrupt the `shift(1)` exclusion across teammates sharing a game date).

**Sample-size columns are kept as features, not just internal working columns** — `n_pitches`, `pa_total`, `contact_n`, `games_n` (pitcher pbp-rolling) and `plate_appearances`/`ab`/`ip` (box-score rolling) all survive into the final output. This lets a model learn to trust a rolling rate less when it's built from a thin recent window — see §5E, which turns this same signal into an explicit engineered feature.

**Scope note:** rolling features do not currently exist for handedness/platoon splits (§5B) or park factors (§3) — both remain season-level only.

---

## 5E. Engineered Interaction Features (v3) — Implemented

Built in `processing/features/interaction_feats.py`, consumed by `experiments/v3_interaction_feats/train.py`. Two column families, both derived from the short-window-vs-season-window pairs §5D produces.

**Pair-finding helpers:**
- `find_rolling_trend_pairs(columns)` — matches every `*_roll_last{N}g_{stat}` column to its `*_roll_season_{stat}` counterpart (same entity prefix, same stat name).
- `find_sample_size_col(columns, rate_col)` — finds the sample-size denominator sharing a rate's rolling prefix, checked in priority order (`plate_appearances` → `pa_total` → `ab` → `ip` → `n_pitches` — a rate's own true PA denominator first, coarser fallbacks after).

**`build_trend_features(df, pairs)`** — per pair, adds:
- `{prefix}_trend_ratio_{stat}` = short-window value / season value (>1 = running hot vs. own baseline, <1 = cold; zero-season-guarded).
- `{prefix}_trend_direction_{stat}` = `sign(short − season)` — a coarse +1/−1/0 hot/cold/flat indicator, magnitude discarded.

**`build_shrinkage_weight_features(df, rate_to_sample_col, k=10.0)`** — per rate column, adds:
- `{prefix}_shrinkage_weight_{stat}` = `sample / (sample + k)` — a smooth 0→1 confidence weight, 0.5 at `sample == k`.
- `{prefix}_shrunk_{stat}` = `rate × weight` — an explicit product between a rolling rate and its own sample size.

**Why these exist:** built only after a PDP (partial dependence) diagnostic — fitting a Random Forest on raw rolling features alone and checking its 2-way partial dependence surfaces — showed neither interaction was already being learned implicitly (see the PDP diagnostics section of `experiments/v3_interaction_feats/train.py`). An earlier version (`build_trend_diff_features`, a plain `short − season` difference) dominated feature importance but measurably *hurt* held-out val metrics (ROC-AUC, PR-AUC, log loss, Brier, decile spread all worse than the raw-features baseline) — a deterministic linear combination of two already-present columns gives a tree no information it couldn't already reconstruct with an extra split, while diluting `RandomForestClassifier`'s `max_features='sqrt'` random split sampling with a near-duplicate column. Ratio/direction (non-linear/coarsened transforms) were tried instead as a different hypothesis, not a guaranteed fix.

---

## 6. Practical Notes for Feature Engineering

- **Stability vs. sample size:** Barrel%, K%, BB%, and Hard-Hit% stabilize (become self-predictive) much faster than BA or BABIP. On partial-season data, prefer the "sticky" metrics as leading features and treat outcome-based rate stats (AVG, ERA) with more caution.
- **xStats vs. actual stats as features:** using both the actual and expected version of a stat (once xwOBA exists here) lets a model implicitly capture "performing above/below expected skill" — a regression-candidate signal.
- **Avoid leakage — repo-specific:** every `build_*` function in `season_stats.py` handles this via `_shift_to_last_season`, which shifts `game_season` forward one year and renames any column containing `season_` to `last_season_` (e.g. `batter_season_ba` → `batter_last_season_ba`) so a season's stats can only be joined onto *next* season's games, never the same one. Any new build function must follow the same pattern — shift once, at the top-level `build_*` entry point, not inside a private `_create_*` helper.
- **Avoid leakage — realized vs. expected pitcher role:** `pitcher_role` (`'sp'`/`'bullpen'`) reflects who *actually* pitched a PA — only knowable after the game, since a manager might pull his starter after 4 innings or let him go 8. Every pitcher-side feature merge in `experiments/v3_interaction_feats/train.py` (season stats, rolling stats, hand-split stats) is gated on `expected_pitcher_role`/`expected_pitcher_key_id` (`processing/features/expected_role.py`), not the realized `pitcher_role` — a pre-game-knowable estimate built from the starter's own historical depth. Realized `pitcher_role` is still used, deliberately, inside the `build_*` functions in `season_stats.py`/`rolling_stats.py` themselves (a pitcher's own season aggregate should reflect his true starts, not a guess) — the fix is specifically at the join/gating layer, not the aggregation layer. Any new pitcher-side feature merged into `model_df` must key on `expected_pitcher_key_id`/`expected_pitcher_role`, never the realized columns (which are kept in the frame only for diagnostics, e.g. checking `expected_pitcher_role`'s agreement rate against the realized truth on historical data).
- **Data sources:** Baseball Savant (Statcast layer, sections 1–2), FanGraphs (sabermetric layer, section 3, easiest CSV export), Retrosheet (raw play-by-play if computing from scratch). This repo's own raw data comes from the MLB Stats API (`raw_data/games/*`, `raw_data/playbyplay/*`) — confirm which Savant/FanGraphs fields, if any, are already present in that pull before assuming a ❌ gap actually requires a new source.

---

*Compiled as a feature-engineering reference for `hit_predictor`. Cross-check the "In this repo" column against `src/models/hit_predictor/processing/schema.py` and `src/data/modules/preprocessing.py` if either file changes — this glossary reflects their shape as of the completed batter + pitcher season-stats implementation in `season_stats.py` (§5, §5B, §5C), the rolling-window implementation in `rolling_stats.py` (§5D), and the v3 engineered interaction features in `interaction_feats.py` (§5E). `season_stats.py` and `rolling_stats.py` both point back here in a header comment — these files are meant to be read together, not duplicated.*
