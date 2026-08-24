# Benchmarks & Research Notes — hit_predictor

**Date:** 2026-08-19
**Purpose:** answer "what does beating the baseline / a good model actually require here" — both as an evaluation methodology and as a synthesis of external research, so decisions about where to spend effort next are grounded rather than guessed. Read alongside `FEATURE_GLOSSARY.md` (what features exist) and `dashboard_spec.md` (how to inspect a model's errors).

This doc covers `hit_predictor` (per-PA `is_hit` classifier). There is currently no separate pitcher strikeout model — the "pitcher" work so far (`expected_role.py`, the v4 experiment) is pre-game role/TTO gating that *feeds* this same batter model, not an independent target. §5.4 below is written for when a K-prop model gets built.

---

## 1. Current state (local, from `mlruns`)

**Reliability + Resolution (Murphy's decomposition of Brier — `reliability - resolution + uncertainty`) are the headline pair for "did this experiment help," not log_loss/Brier alone.** Calibration in isolation is gameable — a model that predicts the base rate for every row is perfectly calibrated (reliability ≈ 0) and useless (resolution = 0), so a real improvement means reliability flat-or-down *and* resolution up, not either alone. Both are already computed by `evaluate_hit_predictor()` and auto-logged to MLflow for every run (`utils/eval.py`, `utils/mlflow_logging.py`) — they just weren't making it into this table before. ECE is kept alongside as the more human-readable single calibration number (same idea as reliability, absolute-error-weighted instead of squared-error-weighted). See §2 for the full framework and `utils/eval.py::summarize_verdict()` for the codified version of this exact check.

| Run | Val LogLoss | Δ vs naive | Val Brier | Val ROC-AUC | ECE | Reliability | Resolution |
|---|---|---|---|---|---|---|---|
| Naive baseline (predict train hit rate ~0.227) | 0.5281–0.5294 | — | 0.172 | 0.500 | — | — | — |
| Rules-based baseline (prev-BA + shrunk rolling + order-slot blend) | 0.5295 | +0.0001 (worse) | 0.1727 | — | — | — | — |
| v1 — pitcher features | 0.5285 | ~flat | 0.1721 | 0.524 | — | — | — |
| v2 — rolling features | 0.5294 | ~flat | 0.1725 | 0.517 | — | — | — |
| v3 — interaction feats | 0.5293 | ~flat | — | 0.514 | — | — | — |
| v4 — pitcher-expected-adj (inning-based role gating + game context) | 0.5291 | +0.0029 (worse, ~flat) | 0.1724 | 0.5160 | 0.0280 | 0.0012 | 0.0001 |
| v5 — tier1-features (arsenal entropy, tunnel proxies, extended log5, velocity-decline trend) | 0.5291 | +0.0030 (worse, ~flat) | 0.1724 | 0.5176 | 0.0288 | 0.0013 | 0.0001 |
| Statistical shrinkage baseline (this-season BA-so-far, k=100, shrunk to last-season BA else league avg) | 0.5279 | -0.0002 to -0.0015 (better) | 0.1717 | 0.5206 | 0.0185 | 0.0005 | 0.0002 |

ECE/Reliability/Resolution for v1–v3 are not backfilled here — each version's `mlruns/` has 8–15 logged runs (naive/rules/LR/RF/XGB × seasons) with no run-name column visible from the file store alone, and guessing which run matches this table's existing LogLoss/Brier/ROC-AUC numbers risks putting wrong values in a doc meant to be trusted. Backfill only by matching a run's tags/params to confirm it's the same one already summarized here — otherwise leave `—` rather than guess. **Every run from v4 onward: fill all seven columns from `evaluate_hit_predictor()`'s return dict before moving on** — that's the after-every-experiment check this table exists for.

Every iteration so far lands within ~0.001 log_loss of the naive floor, and ROC-AUC tops out around 0.52 — barely above coin-flip discrimination. This is despite already having Statcast-grade inputs (exit velo, launch angle, spin, park factors, platoon splits, TTO gating, rolling windows) per `FEATURE_GLOSSARY.md`. §4 below is about whether that's expected or a red flag.

**Above table is PA grain only.** Per §4.5, game grain is the primary evaluation target going forward. v4 is the first run with a proper head-to-head naive-vs-model comparison at game grain (`utils/eval.py::run_pa_vs_game_grain_check`, added 2026-08-22 — trains an actual `DummyClassifier` through the same aggregation as the real model, rather than approximating "naive" from Brier's `uncertainty` term):

| Run | Grain | Naive ROC-AUC | Model ROC-AUC | Naive Brier | Model Brier | Naive Reliability | Model Reliability | Naive Resolution | Model Resolution | Verdict (`summarize_verdict`) |
|---|---|---|---|---|---|---|---|---|---|---|
| v4 | PA | 0.500 | 0.5160 | 0.1713 | 0.1724 | — | — | — | 0.0001 | — |
| v4 | Game | **0.6448** | 0.6361 | **0.2230** | 0.2259 | not recorded | not recorded | **0.0182** | 0.0138 | not computable — reliability wasn't captured for this run |
| v5 | PA | 0.500 | 0.5176 | 0.1713 | 0.1724 | — | — | — | 0.0001 | — |
| v5 | Game | **0.6448** | 0.6367 | **0.2230** | 0.2260 | not recorded | not recorded | **0.0182** | 0.0138 | not computable — reliability wasn't captured for this run |
| Statistical shrinkage baseline | PA | 0.500 | 0.5206 | 0.1713 | 0.1717 | — | — | — | 0.0002 | — |
| Statistical shrinkage baseline | Game | **0.6448** | 0.6489 | **0.2230** | 0.2229 (tie) | 0.0041 | **0.0015** | **0.0182** | 0.0159 | **`calibration_only`** |
| + pitcher signal, static (v1) | Game | **0.6448** | 0.6474 | **0.2230** | 0.2231 | 0.0041 | **0.0013** | **0.0182** | 0.0155 | `calibration_only` vs batter-only shrinkage |
| + pitcher signal, in-season blended (v2) | Game | **0.6448** | **0.6507** | **0.2230** | **0.2224** | 0.0041 | **0.0012** | **0.0182** | **0.0162** | **`real_improvement`** vs both batter-only and v1 |

At game grain, the naive baseline beats every ML model tried (v1–v5) on every metric that was actually recorded for those runs — see §4.5's correction for why (the naive prediction, though constant per PA, becomes non-constant once aggregated, because `n_pa` varies by game). **v5 (4 Tier 1 features added on top of v4 — arsenal entropy, pitch-tunneling proxies, extended log5 matchup, velocity-decline trend) moved nothing**: game-grain ROC-AUC 0.6367 vs v4's 0.6361, Brier 0.2260 vs 0.2259 — within noise, naive still wins on every recorded metric by the same margin. Consistent with backlog item 9's framing: these features add signal *within* the existing per-PA-rate paradigm, but the naive baseline's edge comes from `n_pa` (plate-appearance count), which none of them touch. See `ROADMAP.md` — the `n_pa`-prediction angle (backlog item 1) is now being worked on separately, outside this repo's experiment-folder pattern.

**Known gap surfaced 2026-08-22, while building `summarize_verdict()` (§2): v4/v5's game-grain runs never logged reliability, only resolution/ROC-AUC/Brier.** No `summarize_verdict` verdict can be computed for them retroactively — don't backfill a guessed number. Future experiments (v6+) get this for free since `run_pa_vs_game_grain_check`'s `game_metrics` dict already includes it; this table's blanks are a historical artifact of v4/v5 predating the codified check, not evidence either way.

**New (2026-08-22): the statistical shrinkage baseline — corrected verdict below.** `baseline/statistical/run_baseline.py` — a single empirical-Bayes cascade, `(cum_hits_before + k*shrink_target) / (cum_pa_before + k)` with `k=100`, where `shrink_target` is last-season BA (falling back to that season's league-average hit rate for rookies) — run through the same `run_pa_vs_game_grain_check` harness as everything above, now with a live `summarize_verdict()` call against naive baked into the script itself (not eyeballed after the fact). **Verdict: `calibration_only`, not `real_improvement`.** Reliability is genuinely better than naive (0.0015 vs 0.0041 — more honest probabilities), but resolution is *worse* than naive (0.0159 vs 0.0182), not better as ROC-AUC alone (0.6489 vs 0.6448) suggested. This is exactly the trap §2 describes: ROC-AUC moved in the "beats naive" direction while the actual decision metric (resolution) didn't. It does still beat v5's *resolution* (0.0159 vs 0.0138) — real discrimination improvement over every ML model tried — but a full verdict against v5 isn't computable (see gap above). Full results: `baseline/statistical/baseline_results.md`.

**Correction to this doc's own earlier framing (same day, 2026-08-22):** an earlier version of this section read the ROC-AUC/Brier numbers above as "beats naive at game grain" before `summarize_verdict()` existed to check reliability/resolution together — exactly the failure mode §2 now exists to prevent. Left the ROC-AUC/Brier numbers in the table above rather than deleting them (they're real, just not decisive), and corrected the verdict language everywhere else in this doc/`ROADMAP.md` to `calibration_only`.

This still reframes backlog item 9's open question, just with the corrected verdict: **the ML models (v1–v5) aren't just failing to beat naive — v5's *resolution* specifically is worse than a two-line statistical formula that doesn't even use `n_pa`.** The shrinkage baseline's reliability edge over naive comes from real per-batter signal (last-season BA and this-season form), not the `n_pa`-driven aggregation effect naive exploits — but naive's resolution edge is *also* real, and the shrinkage baseline hasn't closed it. That leaves *three* separate open levers, not two: (a) predicting `n_pa` pre-game (item 1 — **v1 regression attempt returned a negative result 2026-08-22, but a reframed binary classifier (`low_pa = n_pa<=3`) is positive as of 2026-08-23** — see `ROADMAP.md`), (b) whatever ~910-PA-stabilization-grade per-batter signal the shrinkage formula captures that five iterations of feature-engineered gradient-boosted/RF models aren't fully using (item 2), and (c) closing the shrinkage baseline's own resolution gap vs naive — it's closer than the ML models but still behind. A natural next step once (a) lands: combine (a) and (b) — this shrinkage-based per-PA rate × a predicted `n_pa`, aggregated the same way — as a new floor to beat.

**Update 2026-08-23: lever (c) — pitcher signal, added and verified in two iterations.** `baseline/statistical/run_matchup_baseline.py` extends the shrinkage formula with a log5-combined pitcher term (`shrinkage.py::add_matchup_shrinkage_component` / `add_matchup_shrinkage_component_blended`, 12 tests, TDD; still no regression or fitting anywhere — both are closed-form, same as the base formula). Before adding it, `baseline/statistical/matchup_extremity_check.py` (`utils/matchup_slicing.py::slice_by_matchup_extremity`) pooled ~150K val-season PAs into a 3×3 batter×pitcher grid and confirmed real, if modest, pitcher signal exists in the data (weak-batter-vs-dominant-pitcher pooled hit rate 0.202, vs. 0.242 for the reverse extreme) that the batter-only shrinkage baseline wasn't using at all.

- **v1 (static pitcher term — last season's hit-rate-allowed, frozen for the whole season): flat.** Verdict `calibration_only` vs. the batter-only baseline — reliability nudged better (0.0013 vs 0.0015) but resolution went *down* (0.0155 vs 0.0159), same ROC-AUC-moved-but-resolution-didn't trap as the original naive-vs-shrinkage check. Diagnosed concretely using Blake Snell's actual 2024 game log (`baseline/statistical/pitcher_case_study.py` — identified mechanically as that season's most dominant qualified starter by realized hit-rate-allowed among ≥15-start pitchers, turned out to be the real NL Cy Young winner): his static pitcher term (0.157) applies the same "dominant" discount to every start including his shaky April–May stretch, when his real in-season form didn't yet justify it.
- **v2 (in-season-blended pitcher term — pitcher's own cumulative hits-allowed-so-far this season, shrunk toward his last-season rate via the identical cascade the batter side already uses): `real_improvement`.** Beats the batter-only baseline on every metric in the table above (reliability 0.0012, resolution 0.0162, ROC-AUC 0.6507, Brier 0.2224), and beats v1 static too (verdict `real_improvement`, resolution delta +0.0006). The Snell worked-example table confirms the mechanism directly: early-season predictions sit visibly above v1's (hasn't earned the discount yet), converging to match v1 by his 20th start once enough in-season sample has accumulated — exactly the asymmetry diagnosed above, now fixed.
- **Still doesn't close the gap to naive** (resolution 0.0162 vs. naive's 0.0182) — real, measured progress on lever (c), not a full close. Item (c) stays open but is now the most-improved of the three levers.
- Two batter-side case studies (`baseline/statistical/batter_case_study.py`) — worst (Mitch Garver, 0.146 hit rate/418 PA) and best (Bobby Witt Jr., 0.301/692 PA, finished 2nd in AL MVP voting that season) qualified 2024 batters, again identified mechanically not picked — show the v2 model settling into cleanly separated, non-overlapping predicted-probability bands for both players within a few weeks of the season, despite per-game raw hit rate (only ~4 PA/game) bouncing between 0 and 0.6–0.8 the whole time. A concrete, player-level illustration of the PA-grain-noise-vs-signal distinction in §4.2 below.
- Full interactive walkthrough (season charts + hand-traceable per-PA worked tables for all three case studies): `.lavish/eval-metrics-cleanup.html`.

**Update 2026-08-23: what is the model actually good and bad at? First real answer, via `baseline/statistical/slice_diagnostic.py`.** Reuses `shared/model_dashboard/`'s existing, tested slicing/contribution/bootstrap tooling (built per `CLAUDE.md`'s Model Layer section for exactly this — "used to debug any PA-grain model, not just this one" — but never previously wired to `hit_predictor`'s actual predictions) against the v2 (pitcher-blended) model's real 2024 val predictions. Slice-average log_loss with a bootstrap 95% CI is the right lens even at PA grain: unlike resolution it isn't fooled by n=1 noise as long as a slice pools enough PAs — `min_n` raised to 500 from the dashboard's default of 50 for exactly that reason, same principle as the matchup-extremity grid above.

- **`times_through_order=3` (batter's 3rd+ PA vs. the same starter) is a confirmed real weak spot**, not noise: log_loss 0.5469, 95% CI [0.5403, 0.5536], entirely above the overall 0.5286. `=2` is also a real, smaller gap; `=1` sits below overall but its CI overlaps it (not confidently better). Matches Lichtman's TTOP research, already cited in `pipeline.py`'s `_add_pbp_times_through_order` docstring — direct evidence the *effect* is real in this data and the *model* isn't fully capturing it, not just a literature citation.
- **`pitcher_role` is the single largest confirmed-real split, ranked #1 by contribution** — predictions against starters are measurably worse (log_loss 0.5338, real gap) than against bullpen (0.5204, also real, opposite direction). Not yet investigated why; a plausible thread given the TTO finding above (TTO effects are structurally starter-only, since `times_through_order` is NaN for bullpen PAs by construction) but not confirmed as the cause.
- **One expected effect that did *not* show up as a confirmed gap: `platoon_matchup=same_hand`** (batter/pitcher same-hand — a well-documented real baseball effect) — CI overlaps the overall log_loss at this sample size. Worth re-checking with more pooling or a different metric before concluding the model handles platoon splits adequately.
- Tool is reusable for any future slice (new candidate features, other models) or model version, not a one-off diagnostic.

---

## 2. Evaluation framework — what "beating baseline" should mean

Two separate bars, often conflated:

### Bar 1 — statistical: does the model know something the naive rate doesn't

**The decision metrics are reliability + resolution together, evaluated at game grain — nothing else.** This is the single source of truth for "did this experiment help"; every other number in `utils/eval.py`'s output (log_loss, Brier, ROC-AUC, PR-AUC, ECE) is a diagnostic, not a decision criterion. `config.yaml`'s `decision_metrics`/`diagnostic_metrics` fields document this split; `utils/eval.py::summarize_verdict(baseline_metrics, new_metrics)` codifies the actual comparison so it's a function call, not an eyeball check across two printouts.

Two questions, and both must be answered before calling anything an improvement:

- **Can the number be trusted?** (`reliability`, lower = better, 0 = perfect) — if the model says 22%, is 22% roughly what actually happens across every row that got that prediction? This is Murphy's decomposition of Brier score's calibration-error term.
- **Is there a real range of predictions?** (`resolution`, higher = better) — does the model actually say different things for different rows, and do those different outputs correspond to genuinely different outcomes? A model that predicts the base rate for every single row is **perfectly calibrated** (reliability = 0) and **completely useless** (resolution = 0) — reliability alone is gameable by refusing to differentiate at all.

Why both, not either alone: two models can produce the *exact same ROC-AUC* — same rank ordering, same real discriminative power — while one is honest about its confidence and the other isn't. A model that takes the same real separation and pushes its own predicted probabilities further apart than reality actually spreads (e.g. saying 55% where the true rate is 33%) will have materially worse reliability at identical resolution and identical ROC-AUC. ROC-AUC alone would call these two models interchangeable; only reliability catches that the second one is confidently wrong exactly where a bet would be sized largest. `summarize_verdict()` returns one of four verdicts:

| Verdict | Resolution vs baseline | Reliability vs baseline | Reading |
|---|---|---|---|
| `real_improvement` | up | flat or better | the clean win — more differentiation, at least as honest |
| `overconfidence_risk` | up | worse | more spread in predictions, but the spread itself is dishonest — don't trust the resolution gain until recalibrated |
| `calibration_only` | flat/down | better | more honest probabilities, no new signal — useful, but not "the model learned something new" |
| `no_improvement` | flat/down | flat/worse | neither — this is where naive currently beats every ML model tried (v1–v5) at game grain, see §1 |

**Calibration pooling caveat:** a calibration bucket (and therefore reliability/resolution) pools *every row in the validation set that landed at a similar predicted probability* — not one batter's PAs. Checking "is the model calibrated for player X specifically" isn't statistically answerable from one season of any single batter's PAs (§4.2's ~910-PA stabilization threshold), so this is close to the only way to measure calibration at all, not a shortcut.

**Everything else in the eval output is diagnostic, not decisional:**
- **Log_loss / Brier** — composite scores useful for tracking overall trend across runs (and for the naive-floor delta table in §1), but neither one tells you *why* it moved. A run can have flat log_loss with reliability up and resolution down (a calibration-only change dressed up as "no measurable difference") or vice versa — log_loss alone can't distinguish those.
- **ROC-AUC / PR-AUC** — ranking quality, necessary but not sufficient (see the overconfidence example above — AUC can't see it).
- **ECE** — a single human-readable calibration number (absolute-error-weighted, same idea as reliability which is squared-error-weighted). Useful for a quick gut check, not the decision metric itself.

### Bar 2 — economic: does the model beat the market after vig, and is the edge trustworthy enough to size a bet on

Not yet built anywhere in this repo. Requires, at minimum (see §4.3 for the research behind each):
- **Devigged market probability** per prop line (strip the book's hold before comparing).
- **Calibration, not AUC, as the staking-relevant metric** — Kelly sizing needs the raw probability to be trustworthy, not just correctly *ranked*. Same reliability metric as Bar 1, at a stricter bar.
- **CLV (closing line value)** tracking once any bets are placed — the standard practitioner proxy for "is this edge real," since realized bet results are too noisy to judge in the short run.
- **Fractional Kelly** stake sizing, not full Kelly, given model uncertainty.

A model can clear Bar 1 (`real_improvement` verdict) and still be worthless for Bar 2 if the edge it implies is smaller than the vig. A model can also clear Bar 1 with `calibration_only` and still not be worth acting on if it never differentiated in the first place. Accuracy and profitability are not the same axis — Bar 1 is necessary, not sufficient.

---

## 3. Recommended additions to the eval harness

- [ ] Devig helper: convert American odds → implied prob → remove hold (proportional or power method) → compare to model's calibrated prob.
- [ ] Edge-vs-vig report: for each val-set PA/game, `model_prob − devigged_market_prob`, distribution + how much of it is inside the vig band (i.e., not actionable even if the model is right).
- [ ] CLV backtest scaffold (needs historical DK line snapshots — check what's already captured in `raw_data/odds/player_props/`).
- [x] Calibration-focused model selection — **done 2026-08-22** via `summarize_verdict()` + the `decision_metrics`/`diagnostic_metrics` split above: stop reading log_loss delta as the verdict; reliability+resolution together are.

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

**Verdict: framing, not just features.** Same model, same inputs — at the game grain, resolution jumps ~127x and ROC-AUC goes from barely-above-coinflip to genuinely discriminative. The signal was real; per-PA Bernoulli noise was hiding it, exactly as §4.2's stabilization research predicted. Reliability got worse — the game-grain calibration table shows systematic overconfidence (predicted exceeds observed in nearly every bucket, e.g. bottom bucket predicts 0.464 vs actual 0.362) — attributable to the `1-∏(1-p)` independence assumption: a batter's PAs in one game aren't fully independent (shared pitcher/park/weather push them together), and positive correlation among draws means true P(1+ hit) is lower than the independent-draws estimate implies. Fixable via post-hoc recalibration (isotonic regression at the game grain) — a smaller, different problem than "no signal," and doesn't undercut the resolution finding. Script: `scripts/per_game_aggregation_check.py`; tests: `tests/hit_predictor/test_eval.py`.

**Correction (2026-08-22): the "beats naive Brier" line above does not hold — see below.** ~~The model beats its own naive baseline at game grain for the first time.~~

The "Model vs. naive Brier: better by 0.0111" row above was never a naive-*model* comparison — `per_game_aggregation_check.py` only ever trained the Random Forest (confirmed from its saved `per_game_aggregation_check_results.json`, which has no naive/dummy entry at all). "Naive" there meant `evaluate_hit_predictor`'s Murphy-decomposition `uncertainty` term — the Brier score of predicting one constant probability for every *game* — and 0.2372 (uncertainty) − 0.2261 (model Brier) ≈ 0.0111, the exact number in the table.

That's a fair naive floor at PA grain (predicting the constant per-PA base rate *is* the naive baseline there — no aggregation happens, so `uncertainty` and "naive model Brier" are the same number by construction; the PA-grain "worse by 0.0012" row is unaffected). It is **not** a fair floor at game grain. A naive model that predicts one constant probability per *PA* (e.g. `DummyClassifier(strategy="prior")` — the same constant for literally every plate appearance, regardless of batter or game) stops being constant once aggregated through the same `1-∏(1-p)` formula the real model goes through, because **n_pa (how many times a batter came to bat) varies game to game**, and a batter with more plate appearances is mechanically more likely to log at least one hit — real, if shallow, signal the naive model picks up for free, that the flat `uncertainty` floor never captures.

`utils/eval.py::run_pa_vs_game_grain_check` (added 2026-08-22, see `experiments/v4_pitcher_expected_adj/train.py`) fixes this by running an actual `DummyClassifier` through the identical PA→game aggregation as the real model, in the same script, so the comparison is apples-to-apples. v4's run (same features as v2 plus v4's own additions, itself flat vs. v2 — see §1) surfaced the corrected picture:

| Metric | Naive (game grain) | Model (game grain) |
|---|---|---|
| ROC-AUC | **0.6448** | 0.6361 |
| Resolution | **0.0182** | 0.0138 |
| Brier | **0.2230** | 0.2259 |
| LogLoss | **0.6371** | 0.6434 |

**Revised verdict: the naive baseline wins on every metric at game grain.** The "framing, not just features" finding above still holds — game grain genuinely reveals more separable signal than PA grain does, resolution really did jump, per-PA noise really was masking something — but that something is currently mostly the mechanical n_pa-driven aggregation effect, not the model's own features beating a fair baseline. The model has not yet demonstrated it adds value on top of what "more PAs → more hit chances" already gives away for free. **Recalibration (already flagged as a to-do above) doesn't fix this** — it's a resolution/discrimination gap, not a calibration one. The real open question this raises: does the model's *feature signal* (rolling stats, matchups, etc.) add anything once you control for `n_pa` itself, e.g. by including expected-PA-count as an explicit feature, or by evaluating against this corrected naive floor instead of a flat base rate. Not yet answered — flagged in §5 and `ROADMAP.md`.

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
2. ~~Test the per-game aggregation framing (§4.5) before adding more per-PA features.~~ **Done 2026-08-19, corrected 2026-08-22 — result in §4.5.** Framing was a real part of the problem: game-grain resolution is ~127x PA-grain and ROC-AUC 0.516→0.636. **The "model beats naive Brier" part of the original finding was wrong** — that comparison used the wrong naive floor (Brier's `uncertainty` term, not an actual naive model run through the same aggregation). Corrected via `run_pa_vs_game_grain_check` + v4's run: a properly-aggregated naive baseline actually **beats** the model on every game-grain metric. **Game grain is still the right evaluation target** (the framing insight holds), but the open question is now whether the model's features add anything on top of the naive `n_pa`-driven signal — not yet answered.
   - ~~The Tier 1 features in `research/feature_glossary_gap_analysis.md` are still worth building, but "beat this corrected naive floor" is the bar now, not the old uncorrected one.~~ **4 of 12 Tier 1 features built and tested 2026-08-22 (v5 — arsenal entropy, pitch-tunneling proxies, extended log5 matchup, velocity-decline trend).** Result: flat vs. v4 on every metric, naive baseline still wins at game grain by the same margin. This is now reasonably strong evidence that per-PA feature engineering alone won't close the gap — the naive baseline's edge is structurally about `n_pa`, not about anything a rate-based feature can capture. **Backlog item 1 (predicting `n_pa` pre-game) is the more promising remaining lever** and is now the active thread (being worked on separately, outside this repo's experiment-folder pattern, as of 2026-08-22).
3. **Build the profitability layer (§3) before treating any model as "good enough."** A log_loss win over naive says nothing about whether the implied edge survives the vig — that requires devig + calibration-in-the-tails + eventually CLV tracking against real DK lines. (Lower urgency per 2026-08-19 conversation — this is a production/feature-store-readiness project right now, not live betting; revisit before any real staking.)
4. **Treat AUC ~0.51–0.52 as plausibly near a real ceiling — at the PA grain.** Per §4.2, this is consistent with sabermetric variance research. §4.5's result shows the ceiling is lower at the game grain, so this caveat now applies specifically to per-PA evaluation, not to the model's ceiling overall.
5. **Recalibrate before trusting aggregated probabilities.** §4.5's game-grain reliability regression (systematic overconfidence from the `1-∏(1-p)` independence assumption) means the raw aggregated `game_pred_prob` isn't ready to read at face value — isotonic regression (or similar) fit at the game grain is the natural next step before this becomes an input to anything downstream. Note this fixes calibration, not the §4.5 correction's discrimination gap — the model still needs to beat the corrected naive floor on resolution/ROC-AUC/Brier, which recalibration alone won't do.
6. **Don't reuse the unverified numbers** (MDPI 2025 paper, FanGraphs Outcome Machine's 96%) as targets or comparisons — both are flagged unreliable above.
7. ~~For future pitcher K model: same evaluation framework applies (§2–3), with the expectation — untested — that AUC will land meaningfully higher given K-rate's stickiness.~~ **Confirmed, 2026-08-23** — `src/models/k_predictor/` (baseline + v1) reused this evaluation framework directly (`run_pa_vs_game_grain_check`, `summarize_verdict()`) and, unlike `hit_predictor`, the game-grain aggregation check came back `real_improvement` vs. naive (beats it on both reliability and resolution) — the first such verdict anywhere in this project. See `ROADMAP.md`'s Mid-term section for full numbers.
8. For a future pitcher hits-allowed model (§4.6): frame it as `expected innings × per-PA hit-rate-allowed` rather than one end-to-end regression, reuse `build_expected_start_innings` (already more rigorous than any published IP model found), and treat ground-ball rate as the one evidence-backed rate-side lever — but expect the same near-floor AUC ceiling as the batter model, since DIPS/BABIP logic applies with equal or greater force on the pitcher side.
9. **Question from the §4.5 correction: does the model add anything beyond n_pa? Partially answered, 2026-08-23.** The corrected naive baseline's entire game-grain edge comes from `n_pa` varying by game (more plate appearances → more chances at a hit) — a mechanical fact the model should already have access to (batting order slot, lineup position) but apparently isn't exploiting better than the naive aggregation does. **Update 2026-08-22:** v5's 4 new Tier 1 features moved nothing, evidence that the answer is "no, not via rate-based per-PA features within the existing model architecture." **Update 2026-08-23:** the statistical shrinkage baseline (not a v{N} ML model, built separately) *does* add something beyond `n_pa` — its reliability edge over naive is real per-batter/per-pitcher signal, and the in-season-blended pitcher addition (§1's new table rows) improved resolution too, closing part (not all) of naive's remaining edge without touching `n_pa` at all. So the answer for a closed-form statistical model is "yes, partially" — whether that transfers to the gradient-boosted v{N} architecture remains untested (see backlog item 2 in `ROADMAP.md`). The `n_pa` classifier (reframed as `low_pa = n_pa<=3`, positive result 2026-08-23) is a separate, still-open lever — see `ROADMAP.md` backlog item 1.

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
