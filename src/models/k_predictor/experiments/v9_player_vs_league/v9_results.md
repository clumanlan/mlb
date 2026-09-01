# v9 Results — k_predictor (player-vs-league interaction features)

**Date:** 2026-08-29
**Task:** Binary classification (PA grain), aggregated to batter-game grain for the decision verdict
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC (PA grain, triage filter) → reliability/resolution (game grain, decision metric)
**Data:** s3://mlbdk

## What this pass adds

Tests whether v6's shallow XGBoost (`max_depth=2`) is missing a real interaction
between player-level rates and v8's league-wide rates. A tree this shallow can only
combine two features per tree, so an interaction between a weak-marginal feature
(v8's `league_roll_season_pa_strikeout_rate`, ranked #47 of 56) and a strong one
(the player's own K rate) needs the greedy splitter to stumble onto exactly that
pairing, which is unlikely when the weak feature rarely wins a split on its own.
This pass precomputes **player rate ÷ league rate** explicitly — the sabermetric
"+"-stat pattern (wRC+/ERA+ are literally player-rate-over-league-rate,
era-adjusted).

Two new reusable functions in `interaction_feats.py` (TDD'd, 11 new tests in
`tests/hit_predictor/test_interaction_feats.py`), alongside the file's existing
`find_rolling_trend_pairs`/`build_trend_features`:

1. **`find_player_vs_league_pairs`** — auto-matches every non-league rolling
   column to the `league_roll_{window}_{stat}` column sharing its exact window
   bucket and stat name, regardless of entity prefix (batter/pitcher/opp_team/
   pitching_team/anything). Excludes sample-size denominator columns
   (`pa_total`/`plate_appearances`/`ab`/`ip`/`n_pitches`) from pairing — a real
   bug caught mid-pass, see below.
2. **`build_player_vs_league_features`** — for every pair, adds
   `{entity}_vs_league_ratio_{window}_{stat}` (player/league) and
   `{entity}_vs_league_direction_{window}_{stat}` (sign of the difference) — the
   same ratio+direction shape `build_trend_features` already uses, not a plain
   diff, per this file's own documented finding that a plain difference measurably
   hurts held-out metrics (linear combination of two already-present columns a
   tree can reconstruct with one extra split).

**Comprehensive by construction**: this operates by column-name pattern matching
over the fully-assembled `pa_outcome` frame, not a hand-picked subset. Two merges
were widened/added to expose categories that were already computed but not
previously selected: batter/pitcher PA-outcome rolling now selects walk/hbp/
single/xbh/hr rate too (zero new feature-engineering), and a new merge adds the
batter's own slash-line stats (`batter_box_rolling_obp_slg`, already computed for
v7's best-batter extremum feature, never merged onto `pa_outcome` for the batter's
own value before). **20 pairs auto-discovered**, producing 40→38 derived columns
(see the bug below) across batter, pitcher, opposing-team, and pitching-team
K-rate, plus batter/pitcher walk/hbp/single/xbh/hr rate and batter slash-line.

## A real bug was caught and fixed mid-pass

The first run produced a degenerate pair: `pitcher_vs_league_ratio_season_pa_total`
(mean **0.0037**, constant direction **-1.0000**) — and it wasn't inert, it ranked
**#31 of 111** in feature importance on the buggy run.

**Root cause:** `pa_total` is a raw COUNT (a pitcher's own season PA count, in the
hundreds), not a rate. `find_player_vs_league_pairs` matched it against
`league_roll_season_pa_total` (the league's *cumulative* season PA count, tens of
thousands) purely because both share the stat name `pa_total` — but dividing a
per-entity count by a population-wide count isn't a meaningful "player vs. league"
ratio the way an actual rate comparison is (there's no real-world reading of
"this pitcher faced 0.4% of the league's total plate appearances" as a signal).

**Fix:** excluded `_SAMPLE_SIZE_SUFFIXES` (`plate_appearances`, `pa_total`, `ab`,
`ip`, `n_pitches` — a constant already defined in `interaction_feats.py` for
exactly this "these are counts, not rates" categorization) from both sides of the
pairing. Caught with a new regression test,
`test_find_player_vs_league_pairs_excludes_sample_size_denominator_columns`. All
565 tests pass after the fix. The numbers below are from the corrected re-run
(109 features, 38 derived columns, not 111/40).

## Sanity check on the new columns (corrected run)

All new columns land at 93.8-99.4% non-null (pitcher-side slightly lower —
consistent with pitchers having a lower PA-count denominator than batters early in
a season) and ratios center near 1.0 as expected:

| Feature | Non-null | Mean | Sanity |
|---|---|---|---|
| `batter_vs_league_ratio_season_pa_strikeout_rate` | 98.7% | 0.9947 | ~1.0, as expected for a population-centered ratio |
| `pitcher_vs_league_ratio_season_pa_strikeout_rate` | 93.8% | 0.9804 | ~1.0, plausible |
| `batter_vs_league_ratio_season_ba` | 98.7% | 1.0192 | ~1.0, plausible |
| `opp_team_vs_league_ratio_season_pa_strikeout_rate` | 99.4% | 0.9911 | ~1.0, plausible |

## Results (evaluated on val season 2024, 105,265 PAs)

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.2190 | — | 0.5000 |
| Naive (per-role K rate) | 0.2207 | — | 0.5048 |
| Logistic regression (v6 config) | 0.2829 | +0.0622 | 0.5985 |
| XGBoost (v6 config) | 0.2838 | +0.0631 | 0.6004 |
| (v6 tuned XGBoost, standing best, for reference) | 0.2838 | | |

**PA-grain: exact match to v6** (0.2838 vs. 0.2838, -0.0000) — the closest any
post-v6 version has landed to v6's own number, despite 53 new columns.

## Game-grain check (batter-game "1+ strikeout")

| Metric | Naive (per-role) | XGBoost (v6 config) | v6 (standing best, for reference) |
|---|---|---|---|
| reliability | 0.0005 | 0.0002 | 0.0001 |
| resolution | 0.0045 | 0.0137 | 0.0137 |
| roc_auc | 0.5626 | 0.6372 | — |

**Verdict: `real_improvement`** vs naive — resolution matches v6 exactly, reliability within noise.

## Feature importance — the standout finding of this session

**One derived feature dominates the entire 109-feature model:**
`batter_vs_league_direction_season_pa_strikeout_rate` ranks **#1 of 109**, importance
**0.3801** — more than 5x the #2 feature (`pitcher_vs_league_ratio_season_pa_strikeout_rate`,
#2, 0.0655) and over a third of the model's total gain. `batter_vs_league_ratio_season_pa_strikeout_rate`
also places well (#5, 0.0459).

| Feature | Rank (of 109) | Importance |
|---|---|---|
| `batter_vs_league_direction_season_pa_strikeout_rate` | **#1** | **0.3801** |
| `pitcher_vs_league_ratio_season_pa_strikeout_rate` | #2 | 0.0655 |
| `batter_vs_league_ratio_season_pa_strikeout_rate` | #5 | 0.0459 |
| `batter_vs_league_ratio_season_ba` | #13 | 0.0117 |
| `batter_vs_league_ratio_season_pa_single_rate` | #14 | 0.0096 |
| `batter_roll_season_pa_hr_rate` (raw, new this version) | #15 | 0.0085 |
| `batter_roll_season_pa_single_rate` (raw, new this version) | #16 | 0.0073 |

**But almost every OTHER derived column is dead weight** — of the 38 vs-league
columns, roughly two-thirds show exactly **0.0000** importance, including every
single `direction` column except the K-rate one, and every walk/hbp/single/xbh/hr-
vs-league ratio for the pitcher side except HBP/HR (both still modest, #62/#80).
The signal that DOES exist is almost entirely concentrated in the STRIKEOUT-rate
category on both the batter and pitcher side — exactly the category most directly
relevant to this specific target — not spread across the "comprehensive" set.

**Reading: this confirms the shallow-tree-discovery mechanism, but the confirmed
interaction is redundant with information the model already had, not new
information.** `batter_vs_league_direction_season_pa_strikeout_rate` is almost a
pre-thresholded, binarized version of the batter's own raw K rate (`sign(batter
rate − league rate)`, and the league rate barely moves within a single season) —
XGBoost's shallow trees clearly find this single clean binary split more useful
than reconstructing an equivalent threshold from the raw rate over multiple splits,
which is exactly the mechanism hypothesized going into this experiment. But since
PA-grain PR-AUC and game-grain resolution came back **exactly equal to v6's own
numbers** despite this feature capturing over a third of total gain, the
interaction isn't adding new predictive information — it's a more efficient
*encoding* of information the raw batter K-rate feature already provided, not new
signal on top of it. This is the same "individually top-ranked, redundant at the
margin" pattern seen throughout this project's feature-hunt history (v3's whiff
rate, v7's toughest-out K-rate), but the most extreme version of it yet — over a
third of total gain redirected into one feature with zero net accuracy change.

## Interpretation

**Flat aggregate result, but not an uninformative one.** The hypothesis that
shallow trees might be missing a real player-vs-league interaction is **confirmed
mechanistically** (the model eagerly adopted the precomputed threshold as its
single most important split) but **not confirmed as a source of new predictive
power** (aggregate metrics didn't move) — these are two different claims, and this
result cleanly separates them. The "comprehensive" bet (pairing every available
category, not just K-rate) mostly didn't pay off: only the K-rate-vs-league pairs
(batter and pitcher) carry real weight; walk/hbp/single/xbh/hr-vs-league and most
slash-line-vs-league pairs are unused. This is informative in itself — the
strikeout-specific categories are where this mechanism has any traction, not the
broader offensive-environment categories v8 also found weak on their own.

**v6's tuned XGBoost (PR-AUC 0.2838, `real_improvement`, max_depth=2,
learning_rate=0.03) remains the strongest k_predictor version and standing
production candidate** — v9 matches it exactly but doesn't beat it.

## Next steps

- `find_player_vs_league_pairs`/`build_player_vs_league_features` are real, tested,
  reusable infrastructure (any future model with both player-level and
  league-level rolling tables gets vs-league features for free) regardless of this
  specific result — kept, not reverted.
- Don't pursue the walk/hbp/single/xbh/hr-vs-league or most slash-line-vs-league
  pairs further without a new hypothesis — this pass found them essentially
  unused. The K-rate-vs-league pairs (both direction and ratio) are the one
  confirmed-real thread here, though redundant with the raw K-rate already in the
  model on this evidence.
- A cleaner test of whether the vs-league framing adds anything BEYOND the raw
  level (not attempted this pass): drop the raw `batter_roll_season_pa_strikeout_rate`/
  `pitcher_roll_season_pa_strikeout_rate` from `FEATURE_COLS` and keep only the
  vs-league derived versions, to see whether the model's aggregate performance
  holds up on the normalized version alone — if it does, that's evidence the
  "vs-league" framing is a strict improvement (era-adjustment for free) even
  though it's not a strict *addition* on top of the raw feature.
- `window='season'` only this pass, per the earlier scope decision — a trailing-
  window league table (extending v8's functions with an int `window`) would let
  this same machinery test recent-form-vs-league interactions too, not just
  season-to-date.
