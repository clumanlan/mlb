# v12 Results — k_predictor (start-grain Negative Binomial total-K model)

**Date:** 2026-09-01
**Task:** Count regression (one row per SP start), evaluated via P(total K > line) at both a fixed 2024 threshold and real 2026 DK lines
**Target:** Total strikeouts in a starting pitcher's outing
**Primary metric:** 2026 real-odds reliability/resolution/disagreement-win-rate (decision metric) + 2024 val-season coverage/threshold check (continuity with prior versions)
**Data:** s3://mlbdk

## What this pass adds

v6's Poisson-binomial combination models total K as the sum of ~22
independent per-slot Bernoulli trials. A real 1,160-start 2026 DK-odds
backtest (`ROADMAP.md` item 6(c)) found this significantly overconfident —
reliability 0.0145 vs. the market's 0.0011 (13x, bootstrap CI excludes
zero) — consistent with under-modeling the real within-game correlation
across slots (same game, park, weather, bullpen/fatigue trajectory). Eight
prior feature-engineering passes on the per-PA classifier (v3-v11) already
came back flat, so v12 changes what's being modeled instead of adding
another feature: total strikeouts per start, fit directly as its own
Negative Binomial (NB2) count target via
`statsmodels.discrete.discrete_model.NegativeBinomial`, with no per-slot
independence assumption anywhere in the chain. No PA-grain slot expansion,
no Poisson-binomial combine — one MLE fit yields a per-start predicted mean
and a single fitted dispersion alpha directly.

New TDD'd utility: `negative_binomial_pmf(mean, alpha, max_k)` in
`count_distribution.py` (7 new tests, `tests/hit_predictor/test_count_distribution.py`)
— exact NB2 pmf via `scipy.stats.nbinom`, drop-in compatible with
`prob_exceeds_line` and every other pmf consumer, same contract as the
existing `poisson_binomial_pmf`.

Reused wholesale, per the plan's scope decision: the start-grain feature
frame (`expected_batters_faced` cascade, season/rolling pitcher stats,
`pitcher_shrunk_whip`, weather) that `score_2026_test_dates.py` /
`run_xgboost_uncertainty.py` already build in their own Section 4. The one
real gap — that frame previously carried no opposing-*batting*-side K-rate
feature, only the pitcher's own team's — is closed by merging the
already-tested `build_team_batter_strikeout_rolling_feats` table onto
`opp_team_id`/`gamepk` instead of `pitcher_team_id`/`gamepk`. Fit seasons:
`CORE_FIT_SEASONS ∪ {EARLY_STOP_SEASON} = [2018, 2019, 2022, 2023]` (v6's
own fit-data footprint, combined since an MLE fit has no early-stopping
mechanism to hold a season out for).

Categorical features (`pitcher_throw_hand`, `weather_condition`) are
one-hot encoded (`drop_first=True`) rather than ordinal-encoded like the
XGBoost pipeline — appropriate for a GLM linear in the log-mean, where an
arbitrary integer ranking of categories would distort the fit.

## Sanity checks

| Check | Result |
|---|---|
| `opp_team_roll_season_pa_strikeout_rate` non-null rate | 99.3% (mean 0.2218) — matches this pipeline's established ~93-99% rolling-season coverage convention |
| MLE convergence | `converged: True` |
| Fitted alpha | 0.0212 (95% CI [0.016, 0.026], z=8.8, p<0.001) — small but real, non-zero, precisely estimated dispersion |
| `opp_team_roll_season_pa_strikeout_rate` coefficient | 6.4833, p=0.003 — EPIC 2's new feature is used non-trivially by the model, not dead weight |

**A real bug surfaced and fixed during this build, not part of the original
plan:** the first fit attempt used `pd.get_dummies(..., dummy_na=True)`
without dropping a reference category — every row's dummy block for a given
categorical then sums to exactly 1, which is exactly collinear with the
GLM's intercept and left the Hessian singular. Point estimates and
predictions were unaffected (the MLE optimum itself doesn't need an
invertible covariance matrix), but every coefficient's standard error came
back `NaN`, making the fit's own summary table statistically unusable.
Fixed by adding `drop_first=True`; the refit's coefficients are numerically
identical to the first attempt (confirming predictions were never wrong),
and standard errors are now all finite. No test added — this is a
preprocessing correctness fix in an experiment script, not new reusable
methodology, per this repo's own convention that `experiments/v{N}_*/train.py`
files aren't unit tested.

## Results

### 2024 val-season coverage check (continuity with `run_xgboost_uncertainty.py`'s check on v6)

| Level (nominal) | v12 empirical | v12 gap | v6 empirical (published) | v6 gap |
|---|---|---|---|---|
| 50% | 63.1% | +13.1% | 56.1% | +6.1% |
| 80% | 86.7% | +6.7% | 79.9% | -0.1% |
| 95% | 97.4% | +2.4% | 93.7% | -1.3% |

**v12's predicted intervals are wider than nominal at every level, and wider
than v6's own intervals at every level too** — the opposite failure mode
from what the working hypothesis expected. v6 was found overconfident
(narrow) on the real-odds backtest; v12 over-corrects into being too wide
against this disconnected 2024 val-season check.

**Correction (2026-09-03):** the `v6 (published)` column above was generated
before the `build_batter_slot_expansion` team-scoping fix (ROADMAP.md item
6(e)) was actually exercised — `run_xgboost_uncertainty.py` had the fix
written into its code but was never rerun until 2026-09-03. Corrected v6
coverage, same 4,786 starts: 50% → 57.3% (+7.3%), 80% → 80.8% (+0.8%),
95% → 94.3% (-0.7%) — essentially unchanged from the numbers above. v12 is
still wider than v6 at every level; this specific finding survives the fix
intact. See `ROADMAP.md`'s 2026-09-03 entry for the full corrected table.

### 2024 val-season threshold check — P(total K > 5.5) vs. realized

| Metric | v12 | v6 (published) |
|---|---|---|
| Reliability (lower=better) | 0.0009 | 0.0018 |
| Resolution (higher=better) | 0.0260 | 0.0239 |
| ROC-AUC | 0.6989 | — |
| PR-AUC | 0.5680 | — |

**v12 looks better than v6 on this specific disconnected check** — both
directions (reliability down, resolution up). This makes the 2026 real-odds
result below more informative, not less: it directly demonstrates the same
caution `BENCHMARKS.md` already raises about the 2024 val-season check being
a weak proxy for real-market calibration.

**Correction (2026-09-03):** the `v6 (published)` column above is stale —
same pre-fix issue as the coverage table above. Corrected v6, same 4,786
starts: reliability 0.0012 (was 0.0018), resolution 0.0299 (was 0.0239),
ROC-AUC 0.7122, PR-AUC 0.5879 (neither previously reported). **This reverses
the interpretation above**: v12 still wins on reliability alone (0.0009 vs.
0.0012), but now LOSES to corrected v6 on resolution (0.0260 vs. 0.0299) and
ROC-AUC (0.6989 vs. 0.7122) — the opposite of "both directions." v12 is not
clearly better than v6 on this check once the bug is fixed; see
`ROADMAP.md`'s 2026-09-03 entry.

### 2026 real-odds backtest (1,160 matched starts, 72 dates — same sample size and dates as v6's)

| Metric | v12 | v6 raw (published) | v6+isotonic (published) |
|---|---|---|---|
| Reliability (lower=better) | 0.0388 | 0.0145 | — (24x better post-cal, per ROADMAP) |
| Market reliability | 0.0011 | 0.0011 | 0.0011 |
| Reliability ratio vs. market | **36x** | 13x | ~1x (post-cal) |
| Resolution (higher=better) | 0.0026 | — | — |
| Market resolution | 0.0022 | — | — |
| Mean edge (model − market) | -0.1161 | (smaller magnitude, unpublished exact value) | — |
| Disagreement win rate | 48.9% (257/526) | — | ~49.6% |
| Significant vs. coin flip (50%)? | No (p=0.63) | — | No |
| Significant vs. -110 break-even (52.4%)? | No (p=0.11) | — | No |
| Reliability gap vs. market significant? | **Yes** (95% CI [+0.026, +0.050], excludes 0) | Yes (per ROADMAP) | — |
| Resolution gap vs. market significant? | No (95% CI [-0.005, +0.004]) | No (per ROADMAP) | — |

## Interpretation

**The working hypothesis is not supported — v12 is significantly WORSE
calibrated against real 2026 DK odds than v6, not better.** v6's own
aggregate overconfidence was 13x the market's reliability; v12's is 36x.
This is the opposite of what modeling within-game correlation via a
properly-dispersed NB2 count distribution was expected to fix. Resolution
is statistically indistinguishable from the market for both models — same
"doesn't out-discriminate the market" conclusion as v6. The disagreement
win rate (48.9%) is, like v6+isotonic's, statistically indistinguishable
from a coin flip and well short of the ~52.4% break-even bar — **no
betting edge either way.**

The fitted alpha (0.0212) is small — NB2 with this dispersion is close to,
but meaningfully wider than, a plain Poisson (ruled out `z=8.8`,
`p<0.001`, so not spuriously near zero). That the model still comes back
this miscalibrated against real market data, despite a real, precisely-
estimated, non-Poisson dispersion parameter, suggests the actual mechanism
behind v6's overconfidence problem may not be simple mean-variance
underdispersion at all — or that whatever within-game correlation exists
isn't well captured by a single global dispersion parameter shared across
every start regardless of matchup, park, or weather. The large gap between
v12's *better* result on the disconnected 2024 threshold check and its
*worse* result on the real 2026 market check is itself the most useful
finding here: it's a second, independent data point (after v6's own
isotonic-recalibration-doesn't-transfer finding) that this project's
disconnected val-season checks are not a reliable stand-in for real-market
calibration, and shouldn't be trusted alone to green-light a production
switch.

**v6's tuned XGBoost remains the strongest and standing production
candidate.** Neither v6 nor v12 clears the bar for betting use; the
working hypothesis behind this specific lever (correlated-slot count
modeling) is now tested and did not pan out.

## Next steps

- Do not pursue further tuning of the NB2 GLM's feature set or dispersion
  structure on this same hypothesis — the direction of the miss (worse,
  not better, calibration on real odds) argues the underlying mechanism
  assumed here (simple global overdispersion) isn't the right fix, not that
  this particular feature set or fit needs more iteration.
- `negative_binomial_pmf` is real, tested, reusable infrastructure
  regardless of this result — kept, not reverted, same convention as every
  prior version's new columns/utilities.
- The two independent findings that the 2024 val-season check doesn't
  predict real-market calibration (v6's isotonic-recalibration-doesn't-
  transfer result, and now v12's better-on-val/worse-on-real split) argue
  for treating that check as a coarse sanity gate only, not a substitute
  for periodically refreshing the real-odds backtest before trusting any
  future calibration claim.
- A genuinely different lever — e.g. a start-level random effect per
  team/park/weather bucket rather than one global alpha, or abandoning the
  single-shared-dispersion NB2 assumption altogether — would be the next
  thing to try under this same hypothesis, if it's revisited; not more
  feature engineering on the current NB2 GLM.
