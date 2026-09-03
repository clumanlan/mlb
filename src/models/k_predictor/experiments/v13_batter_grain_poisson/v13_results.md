# v13 Results — k_predictor (batter-grain Poisson GLM total-K model)

**Date:** 2026-09-02
**Task:** Count regression (one row per real lineup batter per SP start), evaluated via P(total K > line) at both a fixed 2024 threshold and real 2026 DK lines
**Target:** Total strikeouts in a starting pitcher's outing, built as the sum of independent Poisson predictions for each real batter he's expected to face
**Primary metric:** 2026 real-odds reliability/resolution/disagreement-win-rate (decision metric) + 2024 val-season coverage/threshold check (continuity with prior versions)
**Data:** s3://mlbdk

## What this pass adds

v6's PA-grain classifier treats ~22-40 per-slot predictions as independent
Bernoulli trials; v12's start-grain NB2 model collapsed to one row and lost
all batter identity, and came back *worse* calibrated (36x the market's
reliability vs. v6's 13x — ROADMAP.md item 6(d)). v13 is a third grain: one
row per real lineup batter (~9/start), predicting that batter's own
strikeout count against the starter directly, with his expected
plate-appearance count entered as a **Poisson exposure offset** (fixed
coefficient of 1 in log-space) rather than an ordinary feature. A sum of
independent Poisson(mean_i) variables is itself exactly Poisson(sum(mean_i)),
so combining the 9 batters' predictions into one start's total-K
distribution needs no new combination algorithm — just the new
`poisson_pmf` utility applied to the summed mean.

Deliberately a GLM, not XGBoost, at this stage — isolating the grain/
exposure hypothesis from a model-family confound, same reasoning v12 used
relative to v6. Full design rationale (Poisson-additivity, GLM-vs-XGBoost,
why exposure is not a feature) is in the published design artifact,
"Batter-Grain K's" (https://claude.ai/code/artifact/5f696523-9031-46db-bfbd-3e7b640bbe7a).

New TDD'd production code:
- `build_team_bullpen_pa_share` (`game_context.py`) — rolling, point-in-time-safe
  manager quick-hook tendency proxy, a new team-level feature.
- `build_batter_expected_pa` (`game_context.py`) — collapses
  `build_batter_slot_expansion`'s per-slot output to one row per real batter
  (`expected_pa`, `expected_times_through_order_max`).
- `poisson_pmf` (`count_distribution.py`) — thin pmf wrapper, the combination
  step this design's Poisson-additivity property makes trivial.

## A real, severe bug found and fixed mid-build

The first two full runs produced an implausible ~10% implied per-PA
strikeout rate (this repo's documented baseline is ~22%) and a model
predicting roughly half the realized mean K per start. Root cause:
`build_batter_slot_expansion` merges `batting_order` onto
`(gamepk, lineup_position)` with **no team-awareness** — genuinely
ambiguous whenever a `gamepk` has two starts (home SP, away SP), each
needing a *different* team's 9 batters. Passing it the full unscoped
`batting_order` — which every existing caller of this function in the
codebase does, including v6's own `score_2026_test_dates.py` real-odds
backtest script — lets roughly half of all synthetic slots collide against
the WRONG team (a pitcher's own teammates, whom he never actually faces).

Caught via a cheap single-season diagnostic before spending another full
8-season run chasing the wrong hypothesis: match rate against the real
target was 49.9% (an exact home/away-collision signature). Fixed by
team-scoping `batting_order` per start — call the expansion once for
home-team starters against the away lineup, once for away-team starters
against the home lineup, concatenate. Post-fix diagnostic: match rate 99.7%,
summed target matched the true season total exactly (21,423 = 21,423),
implied per-PA rate 0.2198 vs. true 0.2211.

**This is a latent bug in existing production backtest code, not one
introduced by v13.** All 6 existing callers of `build_batter_slot_expansion`
(`score_2026_test_dates.py`, `score_2025_test_dates.py`,
`run_xgboost_uncertainty.py`, `run_naive_batter_uncertainty.py`,
`run_batters_faced_distribution.py`, `run.py`) pass it an unscoped
`batting_order`. v6's own trained PA classifier is unaffected — it fits on
real, correctly-matched `pa_outcome` rows, never on this synthetic
construct — but any script using this function to combine/aggregate
predictions at scoring time, including the one that produced v6's published
13x real-odds reliability gap, may have corrupted batter-identity and
opposing-team features on roughly half its synthetic slots. **Not audited
or fixed in those other scripts — flagged here, out of scope for this plan.**

## Sanity checks

| Check | Result |
|---|---|
| MLE convergence | `Converged: True` |
| Post-fix diagnostic match rate (single-season) | 99.7% (up from 49.9% pre-fix) |
| Post-fix diagnostic implied per-PA rate | 0.2198 vs. true 0.2211 |
| `batter_last_season_pa_strikeout_rate` / `batter_roll_season_pa_strikeout_rate` coefficients | 1.2561 / 1.2733, both p<0.001 — batter identity is used non-trivially, unlike v12 which had none |
| `expected_times_through_order_max` coefficient | -0.2378, p<0.001 — a real, precisely-estimated TTO-penalty-consistent effect (more times through the order, fewer Ks per PA), holding exposure fixed |

## Results

### 2024 val-season coverage check (continuity with `run_xgboost_uncertainty.py`'s check on v6)

| Level (nominal) | v13 empirical | v13 gap | v12 empirical (published) | v6 empirical (published) |
|---|---|---|---|---|
| 50% | 61.6% | +11.6% | 63.1% | 56.1% |
| 80% | 85.4% | +5.4% | 86.7% | 79.9% |
| 95% | 96.7% | +1.7% | 97.4% | 93.7% |

v13's intervals are wider than nominal at every level, same direction as
v12 and v6 — but tighter (closer to nominal) than v12 at every level.

**Correction (2026-09-03):** the `v6 (published)` figures throughout this
section predate the `build_batter_slot_expansion` team-scoping fix actually
being exercised in `run_xgboost_uncertainty.py` (written into the code
2026-09-02, not rerun until 2026-09-03). Corrected v6 coverage, same 4,786
starts: 50% → 57.3%, 80% → 80.8%, 95% → 94.3% — essentially unchanged from
what's quoted above.

### 2024 val-season threshold check — P(total K > 5.5) vs. realized

| Metric | v13 | v12 (published) | v6 (published) |
|---|---|---|---|
| Predicted mean K | 4.784 | — | — |
| Realized mean K | 4.836 | — | — |
| MAE | 1.785 | — | — |
| Reliability (lower=better) | 0.0012 | 0.0009 | 0.0018 |
| Resolution (higher=better) | 0.0295 | 0.0260 | 0.0239 |
| ROC-AUC | 0.7102 | 0.6989 | — |
| PR-AUC | 0.5810 | 0.5680 | — |

**v13 is essentially tied with v12 and beats v6 on this disconnected
check** — and unlike v12, its predicted mean K (4.784) now closely tracks
the realized mean (4.836), a real, working point estimate, not an
artifact of a GLM's intercept trivially matching the mean.

**Correction (2026-09-03):** the `v6 (published)` column above is stale,
same root cause as the coverage-check correction above. Corrected v6, same
4,786 starts: MAE 1.792, reliability 0.0012, resolution 0.0299, ROC-AUC
0.7122, PR-AUC 0.5879. **"v13 beats v6" no longer holds** — corrected v6
now matches or edges out v13 on every metric except MAE (reliability tied
at 0.0012; resolution 0.0299 vs. v13's 0.0295; ROC-AUC 0.7122 vs. 0.7102;
PR-AUC 0.5879 vs. 0.5810), with v13's MAE (1.785 vs. 1.792) the one place
it still edges ahead — essentially a tie, not a win. Combined with v13's
own real-2026-market result (70x the market's reliability, the worst of
the three grains), corrected v6 is now the best or tied-best of the three
on both the disconnected 2024 check and the real 2026 market — see
`ROADMAP.md`'s 2026-09-03 entry for the full picture.

### 2026 real-odds backtest (1,206 matched starts, same 72 dates as v6/v12)

| Metric | v13 | v12 (published) | v6 raw (published) |
|---|---|---|---|
| Reliability (lower=better) | 0.07598 | 0.0388 | 0.0145 |
| Market reliability | 0.00109 | 0.0011 | 0.0011 |
| Reliability ratio vs. market | **70x** | 36x | 13x |
| Resolution (higher=better) | 0.00161 | 0.0026 | — |
| Market resolution | 0.00264 | 0.0022 | — |
| Mean edge (model − market) | -0.2106 | -0.1161 | (smaller magnitude) |
| Disagreement win rate | 46.7% (259/555) | 48.9% (257/526) | ~49.6% (post-cal) |
| Significant vs. coin flip (50%)? | No (p=0.126) | No (p=0.63) | No |
| Significant vs. -110 break-even (52.4%)? | **Yes** (p=0.007) | No (p=0.11) | No |
| Reliability gap vs. market significant? | **Yes** (95% CI [+0.058, +0.092]) | Yes | Yes |
| Resolution gap vs. market significant? | No (95% CI [-0.003, +0.005]) | No | No |

## Interpretation

**The batter-grain hypothesis is not supported — v13 is significantly WORSE
calibrated against real 2026 DK odds than both v6 (13x) and v12 (36x), at
70x the market's reliability.** This is the third distinct grain/model-family
combination tried on this exact problem, and the third to show the same
disconnect: v13 looks *at least as good as* v12 and *better than* v6 on the
disconnected 2024 val-season checks, while being the *worst* of the three on
the real 2026 market. The gap between disconnected-check performance and
real-market performance isn't just present here — it's now a robust,
three-times-replicated pattern (v6's isotonic-recalibration-doesn't-transfer
finding, v12's better-on-val/worse-on-real split, and now v13's same split,
more extreme). Whatever's driving real-market miscalibration is very
unlikely to be explained by "which grain" or "which independence assumption"
alone.

**Unlike v12, this is not a mean-prediction problem** — v13's predicted mean
K (4.784) matches the realized mean (4.836) closely on val-season data,
and the batter-identity features are doing real, statistically significant
work. The failure mode here is specifically that the model's predicted
*probabilities* don't track real market-implied probabilities on 2026 data,
despite tracking realized 2024 outcomes reasonably well — a genuine
train/serve or 2024-vs-2026 distribution-shift signature, not an obviously
broken point estimate.

The disagreement win rate (46.7%) is a new, sharper finding than v6's or
v12's: it's not just indistinguishable from a coin flip, it's **significantly
below the ~52.4% break-even bar** (p=0.007) — the only version of the three
where this specific test rejects in the wrong direction with this much
confidence. **No betting edge — actively worse than following the market
would suggest, on this evidence.**

**v6's tuned XGBoost remains the strongest and standing production
candidate.** Three independent modeling approaches (v6's PA-grain, v12's
start-grain, v13's batter-grain) have now all been tried against the same
real-odds backtest; none clears the bar for betting use, and the grain lever
itself does not appear to be the right one to keep pulling.

## Next steps

- **Do not pursue further grain variations on this same hypothesis.** Three
  distinct grains (PA/slot, start, batter) have now been tried; the pattern
  (looks fine on disconnected val checks, fails on real market) repeats
  regardless of grain, arguing the real problem is elsewhere — most likely
  something about 2026-specific conditions, market sharpness, or a
  methodology issue in how real-market backtest matching itself works, not
  the independence/correlation structure of the count model.
- **The `build_batter_slot_expansion` team-scoping bug found here should be
  fixed in the other 6 callers** (`score_2026_test_dates.py` especially,
  since it produced v6's own published 13x headline number) before trusting
  any of their outputs further — this is a real, separate, higher-priority
  finding independent of v13's own negative result. Not fixed as part of
  this plan; flagged for a follow-up pass.
- `build_team_bullpen_pa_share`, `build_batter_expected_pa`, and
  `poisson_pmf` are real, tested, reusable infrastructure regardless of this
  result — kept, not reverted, same convention as every prior version's new
  columns/utilities.
