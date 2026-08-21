# Benchmarks & Research Notes — hit_predictor

**Date:** 2026-08-19
**Purpose:** answer "what does beating the baseline / a good model actually require here" — both as an evaluation methodology and as a synthesis of external research, so decisions about where to spend effort next are grounded rather than guessed. Read alongside `FEATURE_GLOSSARY.md` (what features exist) and `dashboard_spec.md` (how to inspect a model's errors).

This doc covers `hit_predictor` (per-PA `is_hit` classifier). There is currently no separate pitcher strikeout model — the "pitcher" work so far (`expected_role.py`, the v4 experiment) is pre-game role/TTO gating that *feeds* this same batter model, not an independent target. §5.4 below is written for when a K-prop model gets built.

---

## 1. Current state (local, from `mlruns`)

**Reliability + Resolution (Murphy's decomposition of Brier — `reliability - resolution + uncertainty`) are the headline pair for "did this experiment help," not log_loss/Brier alone.** Calibration in isolation is gameable — a model that predicts the base rate for every row is perfectly calibrated (reliability ≈ 0) and useless (resolution = 0), so a real improvement means reliability flat-or-down *and* resolution up, not either alone. Both are already computed by `evaluate_hit_predictor()` and auto-logged to MLflow for every run (`utils/eval.py`, `utils/mlflow_logging.py`) — they just weren't making it into this table before. ECE is kept alongside as the more human-readable single calibration number (same idea as reliability, absolute-error-weighted instead of squared-error-weighted).

| Run | Val LogLoss | Δ vs naive | Val Brier | Val ROC-AUC | ECE | Reliability | Resolution |
|---|---|---|---|---|---|---|---|
| Naive baseline (predict train hit rate ~0.227) | 0.5281–0.5294 | — | 0.172 | 0.500 | — | — | — |
| Rules-based baseline (prev-BA + shrunk rolling + order-slot blend) | 0.5295 | +0.0001 (worse) | 0.1727 | — | — | — | — |
| v1 — pitcher features | 0.5285 | ~flat | 0.1721 | 0.524 | — | — | — |
| v2 — rolling features | 0.5294 | ~flat | 0.1725 | 0.517 | — | — | — |
| v3 — interaction feats | 0.5293 | ~flat | — | 0.514 | — | — | — |
| v4 — pitcher-expected-adj (inning-based role gating + game context) | not yet run | — | — | — | — | — | — |

ECE/Reliability/Resolution for v1–v3 are not backfilled here — each version's `mlruns/` has 8–15 logged runs (naive/rules/LR/RF/XGB × seasons) with no run-name column visible from the file store alone, and guessing which run matches this table's existing LogLoss/Brier/ROC-AUC numbers risks putting wrong values in a doc meant to be trusted. Backfill only by matching a run's tags/params to confirm it's the same one already summarized here — otherwise leave `—` rather than guess. **Every run from v4 onward: fill all seven columns from `evaluate_hit_predictor()`'s return dict before moving on** — that's the after-every-experiment check this table exists for.

Every iteration so far lands within ~0.001 log_loss of the naive floor, and ROC-AUC tops out around 0.52 — barely above coin-flip discrimination. This is despite already having Statcast-grade inputs (exit velo, launch angle, spin, park factors, platoon splits, TTO gating, rolling windows) per `FEATURE_GLOSSARY.md`. §4 below is about whether that's expected or a red flag.

---

## 2. Evaluation framework — what "beating baseline" should mean

Two separate bars, often conflated:

**Bar 1 — statistical: does the model know something the naive rate doesn't.**
Already instrumented in `utils/eval.py`: log_loss (primary), Brier (secondary), ROC-AUC/PR-AUC, calibration bucket table (`mean_pred` vs `obs_hit_rate`), Murphy decomposition. This is necessary but **not sufficient** for the actual goal.

**Bar 2 — economic: does the model beat the market after vig, and is the edge trustworthy enough to size a bet on.**
Not yet built anywhere in this repo. Requires, at minimum (see §4.3 for the research behind each):
- **Devigged market probability** per prop line (strip the book's hold before comparing).
- **Calibration, not AUC, as the staking-relevant metric** — Kelly sizing needs the raw probability to be trustworthy, not just correctly *ranked*.
- **CLV (closing line value)** tracking once any bets are placed — the standard practitioner proxy for "is this edge real," since realized bet results are too noisy to judge in the short run.
- **Fractional Kelly** stake sizing, not full Kelly, given model uncertainty.

A model can clear Bar 1 (log_loss beats naive) and still be worthless for Bar 2 if the edge it implies is smaller than the vig, or if it's confidently wrong in the tails (poor calibration). Conversely, a model with modest AUC can still be profitable if it's well-calibrated and the vig is thin enough — accuracy and profitability are not the same axis.

---

## 3. Recommended additions to the eval harness

- [ ] Devig helper: convert American odds → implied prob → remove hold (proportional or power method) → compare to model's calibrated prob.
- [ ] Edge-vs-vig report: for each val-set PA/game, `model_prob − devigged_market_prob`, distribution + how much of it is inside the vig band (i.e., not actionable even if the model is right).
- [ ] CLV backtest scaffold (needs historical DK line snapshots — check what's already captured in `raw_data/odds/player_props/`).
- [ ] Calibration-focused model selection: stop optimizing purely for log_loss delta vs naive: a model that's flat on log_loss but tighter-calibrated in the tails could still be more bettable.

---

## 4. External research synthesis

*(Sources cited inline; anything not independently verifiable is flagged as such rather than presented as fact.)*

### 4.1 Published benchmarks for hit-in-PA / hit-in-game prediction — thin

No trustworthy, citable AUC/log-loss number exists for "predict hit/no-hit before the PA happens" that this project can be benchmarked against:

- **"Beat the Streak: Prediction of MLB Base Hits Using Machine Learning"** (Universidade Nova de Lisboa thesis) — closest direct match: per-PA hit prediction, logistic regression/RF/GBM/NN vs. naive baseline. Exact numbers weren't extractable, but the thesis frames per-PA hit prediction as **inherently hard given baseball's stochasticity**, not a "needs more features" problem.
- **FanGraphs "Outcome Machine"** (Jonah Pemstein, 2014) — a "96% accuracy" figure surfaces in searches but the author's own follow-ups flagged real methodological flaws (missing league-average term, overstated accuracy). **Treat as unreliable, not a benchmark.**
- **Baseball Prospectus "Singlearity-PA"** — neural net over all 21 PA outcome types, evaluated via cross-entropy vs. log5; doesn't isolate a comparable single hit/no-hit AUC.
- A 2025 MDPI paper claiming accuracy >0.91 / AUC >0.97 for "baseball outcome prediction" surfaced but the target (PA-level hit vs. likely game win/loss) couldn't be confirmed. **Unverified — do not use as a comparison point.**
- Statcast **xBA** is not a pre-outcome predictor — it scores contact quality (exit velo/launch angle) *after* contact, so it's not comparable to a pre-PA classifier at all.

**Takeaway:** absence of a citable external benchmark means the naive-floor comparison (§1) is the right yardstick to keep using — there's no external number to chase instead.

### 4.2 Why single-PA outcomes are mostly noise

- **Russell Carleton's stabilization research**: batting average needs **~910 PA** to reach split-half reliability of r≈0.7 — a full *season* of BA is noisy; a single PA is close to pure variance from a forecasting standpoint.
- **DIPS theory (Voros McCracken)**: pitchers have essentially no year-over-year control over BABIP — outcomes on balls in play (the bulk of what determines hit-or-out) are governed by defense and luck, not skill, in any short window.

This is a **sourced, credible explanation** for why AUC ≈0.51–0.52 may be close to a real ceiling for a pre-game-features PA classifier — the addressable signal (true talent) is a small slice of a mostly-random single event. It does not fully rule out a pipeline bug, but it substantially lowers the prior that one exists.

### 4.3 Profitability: calibration, CLV, Kelly, vig

- **CLV** is the standard practitioner benchmark for a real long-run edge — more reliable than short-run win/loss record (Closeline, GamblingNerd explainers).
- **Calibration matters more than AUC for staking** — Kelly sizing and edge calculation need the raw probability to be trustworthy; AUC only requires correct ranking.
- **Vig math** (Wizard of Odds): −115 implies ≈53.5% breakeven; a balanced −110/−110 market holds ≈4.8%. Betting profitably requires **devigging the market line first**, then comparing to a calibrated model probability.
- **Player props are reported as structurally softer** (less sharp money, lower limits) than full-game lines (OddsShopper), **but also carry higher hold** (6–10%+ vs. 4–6% on mainlines) — so more calibrated edge is needed to clear vig, not less. Softness and higher cost roughly offset; neither dominates on its own.

### 4.4 Pitcher K props (for when a K model gets built)

No published accuracy/calibration numbers found. Industry sites describe methodology (whiff rate, CSW%, opposing lineup K%, expected innings, umpire zone tendencies) without publishing metrics. The expectation that K props carry **more signal than hit props** is a structural inference (strikeouts are a "three true outcomes" stat pitchers directly control, per DIPS-era findings in §4.2), not a cited benchmark — flag it as a hypothesis to test, not a fact to design around yet.

### 4.5 Per-PA vs. per-game framing

No source directly argues "use per-game because per-PA is too noisy," but the **standard DFS/projection-industry pattern** (SmartFantasyBaseball, FantasyProjectionLab) is: project expected PA count (batting-order/matchup-driven) × per-PA hit probability, then aggregate to player-game P(1+ hits), because that's the grain the actual DK prop resolves at. Aggregating ~4 roughly-independent PA draws should mechanically produce a more separable game-level probability than a single ~0.22-base-rate PA. This is inference from how projection systems are built, not a paper stating it outright — but it is a **concrete, cheap, testable next step**: aggregate existing per-PA predictions to player-game grain and re-measure AUC/calibration before concluding the feature set itself needs more work.

**Result (2026-08-19):** tested. Same `v2_rolling_features` pipeline/model, zero new features — `utils.eval.aggregate_pa_predictions_to_game()` (`game_pred_prob = 1 - ∏(1-p)`, `game_is_hit` = any hit that game) applied to 2024 val predictions (173,067 PA rows → 43,039 batter-game rows, ~4.02 PA/game).

| Metric | PA grain | Game grain |
|---|---|---|
| Resolution | 0.0001 | **0.0135** (~127x) |
| Reliability | 0.0013 | 0.0030 (worse, ~2.4x) |
| ROC-AUC | 0.516 | **0.636** |
| Model vs. naive Brier | worse by 0.0012 | **better by 0.0111** (~4.7%) |

**Verdict: framing, not just features.** Same model, same inputs — at the game grain it clears its own naive baseline for the first time in this project's history, and resolution jumps ~127x. The signal was real; per-PA Bernoulli noise was hiding it, exactly as §4.2's stabilization research predicted. Reliability got worse — the game-grain calibration table shows systematic overconfidence (predicted exceeds observed in nearly every bucket, e.g. bottom bucket predicts 0.464 vs actual 0.362) — attributable to the `1-∏(1-p)` independence assumption: a batter's PAs in one game aren't fully independent (shared pitcher/park/weather push them together), and positive correlation among draws means true P(1+ hit) is lower than the independent-draws estimate implies. Fixable via post-hoc recalibration (isotonic regression at the game grain) — a smaller, different problem than "no signal," and doesn't undercut the resolution finding. Script: `scripts/per_game_aggregation_check.py`; tests: `tests/hit_predictor/test_eval.py`.

### 4.6 Pitcher hits allowed — prediction models & features (research for a possible future model)

*(Requested 2026-08-20: what's out there for predicting how many hits a pitcher allows. No such model exists in this repo yet — this is research to inform one, not a description of anything built.)*

**No citable academic AUC/log-loss benchmark exists** for "predict hits allowed by a starting pitcher in a game" — same absence as §4.1 for batter hits. One directly relevant academic-adjacent source (a preprint, not peer-reviewed) is a negative result:

- **"Predicting Baseball Pitcher Efficacy Using Physical Pitch Characteristics"** (research-archive.org preprint) — neural net + linear regression over 16 game-independent pitch-characteristic features (pitch mix, velocity, spin, etc.), predicting WHIP/BAA/FIP. Result: **models explained <50% of variance** — the authors state their own hypothesis was *not* proven. `ballFrequency` (pitch-mix distribution) was the most important single feature for WHIP; linear regression found no individual feature significantly moved BAA or FIP, though the NN did somewhat better on those two. This is a same-flavor negative result to this repo's own flat log_loss on batter hits — a pitcher's own stuff/characteristics only weakly determine realized hits-allowed-type outcomes in this study.

**DIPS theory (Voros McCracken)** — already cited in §4.2 for batters — has a sharper pitcher-side implication here: pitchers have close to zero year-over-year control over BABIP, meaning WHIP/BAA/hits-allowed are driven mostly by defense and luck, not pitcher skill, especially in single-game samples. Two sourced refinements matter for feature design:
- **Extreme ground-ball pitchers** (~60%+ GB rate) show a real, sustained BABIP suppression — roughly 12 points lower BABIP-on-grounders than their team's average, because grounders are simply easier to field than liners. This is the one evidence-backed exception to "pitchers don't control hits allowed."
- **Knuckleballers** similarly show detectable, real ability to suppress hits on balls in play (McCracken's dERA v2 explicitly models this). Outside those two groups, the effect is described as statistically real but too small to reliably detect pitcher-by-pitcher.
- **Line-drive rate**, by contrast, is reported as largely outside pitcher control — unlike GB/FB tendency, which pitchers do meaningfully influence.

**Industry/betting-vendor framing** (unverified vendor content, not academic — treat directionally, not as fact) converges repeatedly on the same three-factor structure for pitcher hits-allowed props:
1. **Expected innings pitched** — described as the single most emphasized driver, since IP mechanically caps the number of hit opportunities ("a starter who exits in the 5th simply runs out of chances to allow a 6th hit").
2. **Contact profile** — strikeout-heavy pitchers suppress balls in play entirely; pitch-to-contact starters give the lineup more chances per start.
3. **Opponent lineup quality** — high-average, low-strikeout lineups "manufacture hits against anyone," largely independent of pitcher quality.

One vendor (thedatastreak.com) claims, across "15,146 starter logs," that hits-allowed exceeded 5.5 in only 37.7% of starts, framed as evidence the market is structurally over-biased. **Flag as unverified vendor content, not a citable stat** — same treatment as §4.1's MDPI paper and FanGraphs Outcome Machine.

**Innings-pitched projection itself — the biggest single driver per the above — has no rigorously validated public model either.** FanGraphs/The Hardball Times' own published attempt (`projected_IP = pitcher's own avg IP × (opponent avg IP allowed / league avg IP)`) is explicitly self-described by its author as "laughably simple," omits park factors, specific lineup composition, home/away, and recency weighting, and was never back-tested before publication. **This repo's own `build_expected_start_innings`** (`season_stats.py`'s last-season baseline + `game_context.py`'s in-season rolling average, blended via a starts-based shrinkage weight with a league-wide fallback — see Epic E) is already more sophisticated than that published baseline, which explicitly flags shrinkage/recency-weighting as missing from its own version.

**Takeaway for framing a future pitcher-hits-allowed model:** hits allowed in a game is mechanically `opportunities (≈ innings or batters faced) × rate (hits per opportunity)`. Every source above agrees the *opportunities* term is the highest-leverage, most mechanically deterministic driver — and this repo already has a more rigorous version of it built (`build_expected_start_innings`) than anything found in the external research. The *rate* term is BABIP-flavored and close to the same near-irreducible noise floor documented in §4.2 for batter hits, with ground-ball rate as the one evidence-backed, feature-engineerable lever on the rate side (already available via `_create_pitcher_contact_quality_stats`'s `_gb_rate`/`_fb_rate`/`_ld_rate` in `season_stats.py`). This points toward the same per-game-aggregation framing §4.5 recommends for batters — `expected innings × per-PA hit-rate-allowed`, rather than one end-to-end "predict game hits allowed" regression — as the more promising angle to test first, for the same structural reason.

---

## 5. Recommendations / open questions for review

1. **Don't chase an external accuracy number that doesn't exist.** The naive-floor comparison already in `baseline_results.md` is the right yardstick; keep using it.
2. ~~Test the per-game aggregation framing (§4.5) before adding more per-PA features.~~ **Done 2026-08-19 — result in §4.5.** Framing was a real part of the problem: game-grain resolution is ~127x PA-grain, ROC-AUC 0.516→0.636, and the model now beats naive Brier (it didn't, at PA grain). **Game grain should be the primary evaluation target going forward**, and the Tier 1 features in `research/feature_glossary_gap_analysis.md` are better-justified now than before this result.
3. **Build the profitability layer (§3) before treating any model as "good enough."** A log_loss win over naive says nothing about whether the implied edge survives the vig — that requires devig + calibration-in-the-tails + eventually CLV tracking against real DK lines. (Lower urgency per 2026-08-19 conversation — this is a production/feature-store-readiness project right now, not live betting; revisit before any real staking.)
4. **Treat AUC ~0.51–0.52 as plausibly near a real ceiling — at the PA grain.** Per §4.2, this is consistent with sabermetric variance research. §4.5's result shows the ceiling is lower at the game grain, so this caveat now applies specifically to per-PA evaluation, not to the model's ceiling overall.
5. **Recalibrate before trusting aggregated probabilities.** §4.5's game-grain reliability regression (systematic overconfidence from the `1-∏(1-p)` independence assumption) means the raw aggregated `game_pred_prob` isn't ready to read at face value — isotonic regression (or similar) fit at the game grain is the natural next step before this becomes an input to anything downstream.
6. **Don't reuse the unverified numbers** (MDPI 2025 paper, FanGraphs Outcome Machine's 96%) as targets or comparisons — both are flagged unreliable above.
7. For future pitcher K model: same evaluation framework applies (§2–3), with the expectation — untested — that AUC will land meaningfully higher given K-rate's stickiness.
8. For a future pitcher hits-allowed model (§4.6): frame it as `expected innings × per-PA hit-rate-allowed` rather than one end-to-end regression, reuse `build_expected_start_innings` (already more rigorous than any published IP model found), and treat ground-ball rate as the one evidence-backed rate-side lever — but expect the same near-floor AUC ceiling as the batter model, since DIPS/BABIP logic applies with equal or greater force on the pitcher side.

---

## Sources

- "Beat the Streak: Prediction of MLB Base Hits Using Machine Learning" — Universidade Nova de Lisboa thesis (run.unl.pt)
- FanGraphs, "The Run Environment" / "Outcome Machine" — Jonah Pemstein, 2014, + follow-up corrections
- Baseball Prospectus, "Singlearity-PA"
- Baseball Savant — xBA / hit probability methodology
- Russell Carleton, stat stabilization research (summarized via Smart Fantasy Baseball, Twinkie Town)
- Voros McCracken, DIPS theory (FanGraphs Library, SABR)
- Wizard of Odds, "Player Props: Understanding the Math Behind the Lines"
- OddsShopper, "Finding Player Prop Inefficiencies"
- SmartFantasyBaseball, "How to Project Plate Appearances"
- Closeline / GamblingNerd — CLV explainers
- MDPI (2025) baseball outcome prediction paper — found, numbers unverified, not used as benchmark
- "Predicting Baseball Pitcher Efficacy Using Physical Pitch Characteristics" — research-archive.org preprint (Research Archive of Rising Scholars)
- FanGraphs Sabermetrics Library, "DIPS" and "BABIP" entries
- Baseball Prospectus, "Ahead in the Count: Ground-ballers: Better than You Think"
- Wikipedia, "Fielding independent pitching" / "Ground ball pitcher"
- thedatastreak.com, "How to Bet MLB Pitcher Hits Allowed Props" — vendor content, unverified methodology
- The Hardball Times (FanGraphs), "Projecting innings pitched for individual games"
- mlbprops.com, dimers.com, fantasyteamadvice.com — pitcher hits-allowed / prop-projection methodology descriptions (vendor content, unverified)
