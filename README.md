# MLB Prop Research

An ML system for MLB prop research (DraftKings), built around a specific bet: **the way to find real signal isn't to model outcomes better, it's to pick outcomes that have less irreducible noise in them.**

---

## The thesis

The first model here (`hit_predictor`) threw 100+ engineered features and XGBoost at "will this batter get a hit in this plate appearance," aggregated up to the game grain a DraftKings prop actually resolves on. Five model versions in, it still hasn't beaten a naive baseline. Not because the features are wrong — because **a single plate appearance's outcome is mostly noise no feature set can remove.**

Once contact is made, what happens next — exit velocity, launch angle, exactly where nine fielders happen to be standing, whether a liner finds a glove or a gap — is close to a coin flip even for the best hitters in the league (this is why BABIP is famous in sabermetrics: it barely varies by skill). A "run scored" is worse: it's several of those noisy events chained together across an inning, so the uncertainty compounds instead of averaging out.

So the project's direction shifted: instead of trying to out-model that noise, **decompose the prediction problem into sub-problems where the outcome is more dependent on data we have before the game starts** — a roster decision, a workload constraint, or a skill matchup that doesn't depend on where a ball happens to land. Model those instead. `BENCHMARKS.md` and `ROADMAP.md` have the full experimental history; this doc is the "why," kept short.

---

## The dividing line: decided-before-the-game vs. resolved-by-contact

Every candidate prediction target here gets sorted along one axis: **how much of its variance is settled by a legible, pre-game mechanism, versus by the physics of a ball in play (plus defense, park, luck) once contact happens?**

| | Resolved by contact / compounding luck | Resolved by pre-game mechanism / skill matchup |
|---|---|---|
| **What decides it** | Exit velo, launch angle, fielder positioning, park geometry — none of it knowable pre-game, most of it not attributable to either player's skill | A roster decision already made (batting order, bullpen usage) or a repeatable skill duel that never requires a ball in play (strikeout, walk) |
| **Noise floor** | High — BABIP-driven outcomes are close to random even for skilled hitters; a run adds *multiple* such draws in sequence | Low — the outcome doesn't route through contact/defense/luck at all, or is close to reading off a decision a team already made |
| **Sample stabilization** | Batting average needs ~800 PA to stabilize | Walk rate stabilizes ~120 PA, strikeout rate ~60 PA (no batted-ball luck in the numerator) |
| **Examples tried/considered** | Hit/no-hit per PA (`hit_predictor` v1–v5), run scored | `low_pa` (built), K-prop (built), BB-prop (built), starter early exit, bullpen day |

That's the whole reframe: **hit-per-PA and runs-scored sit on the noisy side of this line; everything in the current sub-problem menu sits on the mechanism side.**

**Why so many of these are pitcher-side targets:** it's not a coincidence — predicting a pitcher's outcome is itself a way of controlling for a single batter's plate-appearance variance. A starting pitcher faces far more batters in a game than any one batter gets plate appearances (~20+ vs. ~4), so a pitcher's game-level line is effectively averaged over many more "PA-equivalent" events. Same noise-reduction logic as the dividing line above, applied to *who* you pick as the unit of prediction, not just *what* you predict.

---

## Sub-problem menu

| Target | Mechanism | Status |
|---|---|---|
| **`low_pa`** — batter gets ≤3 plate appearances in a game | Low lineup slot + a starter who typically goes deep + pinch-hit risk — a batting-order decision interacting with a pitcher's known workload pattern | **Built, positive.** XGBoost @ 0.85 confidence → 64.7% precision (95% CI [55.6%, 72.8%]), ~116 qualifying predictions/season. Not yet wired into `hit_predictor`; de-vig check against real DK lines still required. |
| **Strikeout probability (K-prop)** | Pitcher stuff/command vs. batter contact skill — resolved without a ball ever being put in play | **Built, positive.** Baseline LR beats both naive floors (PR-AUC 0.270 vs. 0.226, ROC-AUC 0.582 vs. 0.506); v1 pitcher-workload rolling features flat vs. baseline. Game-grain aggregation check (1+ K vs. the starter) is `real_improvement` vs. naive — beats it on both reliability and resolution, the first such verdict anywhere in this project. Scoped to starter-only PAs (a bullpen reliever isn't identifiable pre-game). Reused most of `hit_predictor`'s existing infra (rolling stats, TTO/role gating) with a different target column, as expected. |
| **Walk probability (BB-prop)** | Pitcher control/zone% vs. batter chase rate — same "three true outcomes" logic as K | **Built, mixed.** Baseline LR beats both naive floors (PR-AUC 0.096 vs. 0.076, ROC-AUC 0.577 vs. 0.501). Scoped to starting pitchers and starting batters only (same population-scoping as K-prop). v1 (rolling walk rate + times_through_order) essentially flat vs. baseline at PA grain (PR-AUC 0.100), though batter rolling walk rate is the single most important feature by a wide margin. Game-grain check ("1+ walk") came back `overconfidence_risk`, not `real_improvement` like K-prop's — real added discrimination (ROC-AUC 0.608 vs. naive's 0.557) but slightly worse calibration than the naive floor. Not yet experiment-ready. |
| **Short outing / bullpen day (≤4 IP or an explicit opener)** | Recent workload/pitch-count trend, opponent's platoon-advantage depth, bullpen rest state, rotation/IL status, day after a doubleheader — one model, since a planned bullpen day is really the extreme end of the same short-outing spectrum as an early pull | Candidate. `expected_role.py` already computes the underlying workload stat as a gating feature. Likely the lowest noise floor of the remaining candidates — closest to reading off a decision already made. |

Weaker candidates considered and set aside for now: player rest-day / will-not-start (useful as a supporting feature, not a standalone target), stolen-base attempts (narrow population, smaller market), home-run props (tempting because power is skill-driven, but still has real batted-ball/weather variance in the tail — not clearly better than the hit problem that already stalled).

**Still open: the batter-side hit-probability decomposition.** The mechanism reframe doesn't retire the original problem — `hit_predictor` still needs real batter-side signal beyond the current statistical baseline (this-season batting average shrunk toward last-season's, k=100). Five bundled ML feature sets (v1–v5) all lose to that two-line formula, and why is still unresolved (`ROADMAP.md` backlog item 2). Not abandoned — this runs alongside the pitcher-side sub-problems above, not instead of them.

---

**Read next**, depending on what you're after:
- `ROADMAP.md` — the living plan: current priorities, backlog, decision log
- `src/models/hit_predictor/BENCHMARKS.md` — full experiment results table, what "beating baseline" means here
- `src/models/hit_predictor/FEATURE_GLOSSARY.md` — every feature, implemented or not
- `CODEBASE.md` — infra reference: ingestion → processing → feature store, S3 layout
- `CLAUDE.md` — repo conventions (TDD requirement, commands, test layout)

**Non-goal right now:** proving this beats the market. The bar is "real, defensible signal" — the economic/CLV layer (comparing model output against DraftKings' actual de-vigged prices) is deliberately deferred until there's a model worth staking behind.
