# v10 Results — k_predictor (count-leverage/put-away rate + park strikeout tendency)

**Date:** 2026-08-30
**Task:** Binary classification (PA grain), aggregated to batter-game grain for the decision verdict
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC (PA grain, triage filter) → reliability/resolution (game grain, decision metric)
**Data:** s3://mlbdk

## What this pass adds

Two independent new feature threads, both closing gaps confirmed absent by direct
code inspection (not assumed) before this pass — see `implementation_plan.md`:

1. **Count-leverage / put-away rate.** No prior version had any feature describing
   whether a PA even reaches 2 strikes, or what happens once it does — every
   existing PA-outcome feature (`pa_strikeout_rate`, `pa_full_count_rate`, etc.)
   describes the overall outcome or the *final* count reached. New TDD'd columns
   (`rolling_stats.py`, 4 new tests in `tests/hit_predictor/test_rolling_stats.py`):
   - `pa_two_strike_reach_rate` = PAs that ever reach 2 strikes (max
     `count_strikes` across every pitch in the PA, not just the final pitch —
     a foul-off at 2 strikes keeps `count_strikes` at 2 on every following pitch)
     / total PAs.
   - `pa_put_away_rate` = strikeouts / PAs-that-reached-2-strikes — a *conditional*
     conversion rate, genuinely different information from `pa_strikeout_rate`
     even though both come from the same underlying counts.
   Built for both pitcher and batter sides, landed in hit_predictor's shared
   `rolling_stats.py` (not a k_predictor-local file) — same target-agnostic-
   infrastructure precedent `pa_strikeout_n`/`pa_full_count_n` already set.
2. **Park-specific strikeout tendency.** `park_factors.py` had exactly one
   function, `build_park_factors` — a hit-rate index (H/AB by venue/season vs
   league) — confirmed by reading the file to have no strikeout equivalent. New
   `build_park_strikeout_factor` mirrors its exact shape (season-level,
   `_shift_to_last_season`, keyed on `(venue_id, game_season)`) using
   `pitcher_boxscore`'s k/ip instead of `batter_boxscore`'s h/ab, matching
   `season_stats.py`'s own `k_rate = k/ip` convention.

Both threads land at `window='season'` only this pass, same "season first, N-game
follow-up is a separate decision" convention v8/v9 used. Carries forward v6's
exact winning hyperparameters and v9's full feature set (109 columns, including
the player-vs-league interaction features) unchanged.

## Sanity check on the new columns

| Feature | Non-null | Mean | Sanity |
|---|---|---|---|
| `pitcher_roll_season_pa_two_strike_reach_rate` | 93.8% | 0.5270 | plausible — roughly half of PAs reach 2 strikes |
| `pitcher_roll_season_pa_put_away_rate` | 93.8% | 0.4120 | consistent with 0.527 reach × 0.412 ≈ 0.217 unconditional K rate, matching this dataset's ~21.8% strikeout rate |
| `batter_roll_season_pa_two_strike_reach_rate` | 98.7% | 0.5296 | matches the pitcher-side number closely, as expected (same underlying PAs, opposite entity) |
| `batter_roll_season_pa_put_away_rate` | 98.6% | 0.4141 | same cross-check as above |
| `park_last_season_strikeout_factor` | 84.2% | 0.9994 | centered almost exactly at 1.0 as expected for a ratio-to-league-mean index; lower non-null than the count-leverage columns because it additionally requires a matched prior-season venue-level aggregate to exist |

The reach-rate × put-away-rate ≈ unconditional strikeout-rate identity holding
almost exactly (0.527 × 0.412 = 0.217 vs. the dataset's actual ~0.218-0.219
strikeout rate) is a real correctness check, not just a plausibility eyeball —
confirms the two new rates are internally consistent with each other and with
the existing `pa_strikeout_rate`, not an independent computation that happens to
look reasonable.

## Results (evaluated on val season 2024, 105,265 PAs)

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.2190 | — | 0.5000 |
| Naive (per-role K rate) | 0.2207 | — | 0.5048 |
| Logistic regression (v6 config) | 0.2833 | +0.0627 | 0.5989 |
| XGBoost (v6 config) | 0.2837 | +0.0630 | 0.6000 |
| (v6 tuned XGBoost, standing best, for reference) | 0.2838 | | |

**PA-grain: flat vs v6** (0.2837 vs 0.2838, -0.0001 — well inside this project's
~0.005 "real" threshold), same pattern as every version since v7.

## Game-grain check (batter-game "1+ strikeout")

| Metric | Naive (per-role) | XGBoost (v6 config) | v6 (standing best, for reference) |
|---|---|---|---|
| reliability | 0.0005 | 0.0002 | 0.0001 |
| resolution | 0.0045 | 0.0137 | 0.0137 |
| roc_auc | 0.5626 | 0.6368 | — |

**Verdict: `real_improvement`** vs naive (same as every version since v2) — but
not deeper than v6's own result: resolution matches v6's exactly (0.0137),
reliability is within noise of v6's 0.0001.

## Feature importance — the 5 new columns

| Feature | Rank (of 114) | Importance |
|---|---|---|
| `pitcher_roll_season_pa_two_strike_reach_rate` | #12 | 0.0172 |
| `batter_roll_season_pa_two_strike_reach_rate` | #16 | 0.0146 |
| `batter_roll_season_pa_put_away_rate` | #19 | 0.0113 |
| `park_last_season_strikeout_factor` | #40 | 0.0054 |
| `pitcher_roll_season_pa_put_away_rate` | #112 | 0.0000 |

**A sharp, genuinely counter-intuitive split within the count-leverage thread
itself — not a uniform "reaching 2 strikes doesn't matter" result.** Both
two-strike-*reach*-rate columns place well inside the top quartile of 114
features (#12, #16), ahead of most of v1-v9's own established engineered
features — a pitcher's or batter's propensity to even get a PA to 2 strikes
carries real, non-redundant signal the model didn't have before. But the two
*put-away*-rate columns — the more directly targeted "given 2 strikes, does the
pitcher convert" hypothesis this feature was actually built to test — split in
opposite directions depending on whose put-away rate it is: the **batter's own**
put-away rate (his susceptibility once behind in the count) still places
respectably at #19, while the **pitcher's own** put-away rate — arguably the
single most on-target feature this whole thread could produce for a model
predicting *this specific pitcher's* strikeout total — ranks #112 of 114,
essentially unused by the model. Park strikeout tendency lands in the
unremarkable middle third (#40), no strong signal in either direction.

One plausible mechanism for the pitcher-put-away asymmetry: `pa_strikeout_rate`
(the pitcher's own, already in the feature set since v1) and
`pa_two_strike_reach_rate` (new this pass) together already let a shallow
(`max_depth=2`) tree reconstruct most of what `pa_put_away_rate` would add —
put-away rate is arithmetically `pa_strikeout_rate / pa_two_strike_reach_rate`,
so once both inputs are present the ratio itself is close to redundant for a
tree-based model, which doesn't need the pre-computed quotient the way a linear
model would. This doesn't explain why the *batter*-side put-away rate escaped
the same fate (batter's own `pa_strikeout_rate` is also already in the feature
set) — a real open question, not resolved by this pass.

## Interpretation

**Flat aggregate result — same "additive feature, no PA/game-grain movement"
pattern seen throughout most of this project's feature-hunt history since v3**
(v3-v5, v7, v8 here; v3-v7 on `batters_faced_predictor`). As with v8's league
walk/single rate, this does not close the "does count leverage matter" question
the way a uniformly-low-importance result would have: two-strike reach rate
carries real, top-quartile signal on both sides of the matchup. The specific
"put-away rate" framing this thread was actually designed around is genuine
evidence against the pitcher-side version of that mechanism (not merely
"untested for it") — it's the single worst-ranked new column of the five,
worse than the exploratory park factor. Park-specific strikeout tendency shows
no signal in either direction this pass.

**v6's tuned XGBoost (PR-AUC 0.2838, `real_improvement`, max_depth=2,
learning_rate=0.03) remains the strongest k_predictor version and standing
production candidate**, six follow-up passes (v5-v9 plus this one) in.

## Next steps

- Two-strike reach rate (both sides) is the strongest new signal to come out of
  this project since v2's opposing-lineup K rate — worth a trailing-N-game
  follow-up (this pass only tried `window='season'`) before concluding it's
  fully captured.
- Don't pursue the pitcher's own put-away rate further without a new angle — it
  ranked dead last here, and the likely mechanism (redundant with
  `pa_strikeout_rate` + `pa_two_strike_reach_rate` already present) argues
  against simply re-trying it at a different window.
- The batter-side put-away-rate / pitcher-side put-away-rate asymmetry is an
  open question this pass didn't resolve — worth a dedicated look if a future
  pass revisits this thread.
- Park-specific strikeout tendency is real, tested, reusable infrastructure
  regardless of this flat result — a 3-year rolling window (the same deferred
  improvement `build_park_factors`' own docstring already notes for the hit-rate
  version) is the more likely lever than re-trying the season-level version
  again.
