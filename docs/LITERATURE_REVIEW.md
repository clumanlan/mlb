# Deep Learning / ML in Baseball — Literature Review

Survey of academic and near-academic work on machine learning and deep learning applied to baseball, collected to evaluate methodology before adopting techniques in this repo's models (`hit_predictor`, `k_predictor`, `n_pa_predictor`, `batters_faced_predictor`, `short_outing_predictor`, `bb_predictor`). Entries are grouped by evidence tier — peer-reviewed venue first, since that's the axis this review was built to prioritize — and each entry states what would need independent verification before trusting its numbers.

This is a snapshot as of 2026-09-15. Several entries are 2026 preprints with unreviewed claims; treat those results as leads, not conclusions.

---

## How to read the tiers

- **Tier 1 — peer-reviewed journal.** Passed external review at a named journal. Still worth checking sample size, leakage handling, and whether the journal is a reputable outlet.
- **Tier 2 — arXiv / non-peer-reviewed preprint with a documented methodology.** No external review. Some of these have been sitting on arXiv for years without landing in a journal, which is itself a signal — noted per entry.
- **Tier 3 — industry write-up, competition paper, or blog.** Useful for ideas and implementation detail, not citable as validated research. MIT Sloan Sports Analytics Conference papers sit here because they're competitively selected but not peer-reviewed in the journal sense.

---

## Tier 1 — Peer-reviewed

### Brill, Deshpande & Wyner — "A Bayesian analysis of the time through the order penalty in baseball"
**Journal of Quantitative Analysis in Sports (JQAS)**, accepted 2023 (arXiv:2210.06724 is the preprint version).

- **Method:** Bayesian multinomial regression over plate-appearance outcomes, controlling for pitcher quality, batter quality, handedness, and home-field advantage, explicitly modeled to separate *continuous* pitcher-performance decay across a game from a *discrete* jump between times-through-the-order.
- **Finding:** After controlling for confounders, evidence for a real discontinuity at the "third time through the order" threshold is weak — the decline looks like gradual, continuous fatigue rather than a step change at TTO=3.
- **Relevance to this repo — highest of anything found.** This directly interrogates the premise behind `game_context.py`'s TTO-gated features (`k_predictor`, `short_outing_predictor`) and the workload-shrinkage cascade `build_expected_batters_faced`/`build_expected_start_innings` rest on. If the JQAS finding holds, a smooth pitch-count/fatigue feature (which `batters_faced_predictor`'s v2 trailing-pitch-count feature already gestures at) may be better-specified than a hard TTO indicator. Worth reading in full before any further TTO feature work.
- **To verify before relying on it:** confirm the MLB seasons covered and whether the multinomial framework's independence assumptions hold under this repo's own point-in-time feature constraints.

### Lee & Kim — "Pitcher Performance Prediction Major League Baseball (MLB) by Temporal Fusion Transformer"
**Computers, Materials & Continua**, Vol. 83, No. 3 (May 2025).

- **Method:** Temporal Fusion Transformer (TFT) — an attention-based architecture purpose-built for multi-horizon time-series forecasting — applied to pitcher ERA forecasting from Statcast data.
- **Finding:** TFT outperformed RNN-based baselines and existing (non-ML) projection systems on ERA prediction accuracy.
- **Relevance:** Directly on point for `k_predictor` and any future move toward the "attention model" named as the Layer 5 target in this repo's architecture doc. TFT specifically is designed for exactly the kind of point-in-time, multi-window rolling-feature setup this repo already builds by hand in DuckDB — it's a plausible architecture to prototype against the current XGBoost baseline.
- **To verify:** exact dataset size/years, feature list, and the specific RNN baselines used weren't recoverable from the abstract alone (publisher blocked full-text fetch) — read the full PDF via Computers, Materials & Continua before citing numbers.

### Zhao et al. — "Machine Learning in Baseball Analytics: Sabermetrics and Beyond"
**MDPI Information**, Vol. 16, No. 5 (2025).

- **Method:** Systematic review, not a new model — builds a taxonomy of ML use cases in baseball (play prediction, player performance, player valuation, injury prediction, game-outcome forecasting) and catalogs which data repositories (mainly Baseball Savant, Baseball Reference) and techniques each use case draws on.
- **Relevance:** Best current map of the field — use this as the index to find follow-up papers in whichever sub-area is being worked on, rather than reading it for its own methodology.
- **Caveat:** MDPI's peer-review rigor is inconsistent across its journal portfolio and it's been the subject of legitimate quality criticism in some fields; treat this as a useful index rather than a fully authoritative synthesis, and check its citations against their original sources.

### "Application of Machine Learning Models for Baseball Outcome Prediction"
**MDPI Applied Sciences**, Vol. 15, No. 13 (2025), DOI 10.3390/app15137081.

- **Method:** Head-to-head comparison of decision tree, logistic regression, a (shallow) neural network, random forest, and XGBoost on game-outcome prediction, evaluated via 5-fold CV on accuracy/F1/sensitivity/specificity/AUC-ROC.
- **Relevance:** A clean template for the evaluation methodology this repo already partially follows (`baseline/` naive-vs-ML comparison pattern) — worth comparing its metric choices against this repo's `summarize_verdict()` reliability/resolution approach ([[feedback_hit_predictor_eval_metrics]] in memory) to see if anything's missing.
- **To verify:** same MDPI-portfolio caveat as above; full text was paywalled from this pass, so the "which model won" result wasn't independently confirmed here.

### Kremer et al. (PMC/AJSM) — "Pitch-Tracking Metrics as a Predictor of Future Shoulder and Elbow Injuries in Major League Baseball Pitchers: A Machine-Learning and Game-Theory Based Analysis"
Peer-reviewed sports-medicine journal (PMC-indexed, 2024).

- **Method:** XGBoost classifier predicting next-season shoulder/elbow injury from pitch-tracking metrics, with SHAP (Shapley additive explanations) used to rank feature importance and interaction effects — not a deep learning method, but methodologically rigorous and directly measurable.
- **Finding:** ~84% accuracy; ball-tracking metrics (pitch velocity across all pitch types, slider usage rate, fastball spin rate, fastball horizontal movement) outpredicted workload/demographic features.
- **Relevance:** Tangential but real — if `short_outing_predictor` or `batters_faced_predictor` ever wants a pitcher-durability signal beyond rest days and trailing pitch counts, this says velocity/movement decay is a stronger signal than raw workload counting.
- **To verify:** full text was blocked by a bot-check on this pass; re-fetch directly or via PubMed before citing the 84% figure precisely.

### "Data-driven approaches for predicting Tommy John Surgery risk in major league baseball pitchers"
**Journal of Big Data** (Springer), 2025.

- **Method:** Deep learning framework with joint classification (injury risk) and regression heads, trained on 2016–2023 MLB pitching data; classification arm reports 0.73 F1 detecting injury risk up to 100 days out.
- **Relevance:** Same category as the Kremer paper above — durability/injury signal, not outcome prediction, but useful if workload-based features get revisited for the pitcher-side models.
- **To verify:** full text was paywalled (Springer auth wall) on this pass — confirm the F1/AUC numbers and exact feature set from the original article before citing.

### "A context-enhanced deep learning approach to predict baseball pitch location from ball tracking release metrics"
**Sports Engineering** (Springer Nature), 2025.

- **Method:** Multi-output deep neural network predicting final pitch location from release-point ball-tracking metrics plus game context, trained on 2M+ pitches from NCAA Division I games (not MLB).
- **Relevance:** Lower priority for this repo — NCAA data, and pitch-location prediction isn't a current sub-problem here — but the "context-enhanced" framing (conditioning a DL model on situational features alongside raw tracking data) is the same pattern this repo already uses (rolling stats + role gating + game context).
- **To verify:** paywalled on this pass; note it's NCAA not MLB data before treating any accuracy numbers as transferable.

### "Machine Learning Applications in Baseball: A Systematic Literature Review"
**Applied Artificial Intelligence** (Taylor & Francis), 2018.

- **Method:** Earlier systematic review, pre-dating most deep learning adoption in this space.
- **Relevance:** Useful only as a historical baseline for how much the field has shifted toward deep learning since 2018 — the Zhao et al. 2025 review above supersedes it for current state.

---

## Tier 2 — arXiv preprints, not peer-reviewed, but methodologically documented

### Ahn, Du, Zhang & Kang — "Neural Sabermetrics with World Model: Play-by-play Predictive Modeling with Large Language Model"
arXiv:2602.07030, submitted 2026-02-02.

- **Method:** Casts an entire MLB game as one long autoregressive token sequence and continually pretrains a single LLM on 10+ years of MLB tracking data (~7M pitch sequences, ~3B tokens). Evaluated on next-pitch-type prediction within a plate appearance and batter swing-decision classification.
- **Reported results:** ~64% accuracy on next-pitch prediction, ~78% on swing-decision classification, beating "strong neural baselines from prior work" (unnamed in the abstract).
- **Relevance:** The most architecturally ambitious paper found — a genuine "world model" framing rather than a per-target classifier — but unreviewed, six weeks old at time of writing, and the comparison baselines aren't identified from the abstract alone. Interesting as a research direction, not something to adopt into production without reading the full paper and checking whether "beats baselines" holds up under this repo's own point-in-time-safety standard.
- **To verify:** base LLM identity, exact baseline models, and whether their point-in-time handling matches this repo's shift/gating discipline — full text needed.

### Sun, Lin & Tsai — "Performance Prediction in Major League Baseball by Long Short-Term Memory Networks"
arXiv:2206.09654, submitted 2022-06-20.

- **Method:** LSTM predicting player home-run counts, compared against traditional ML baselines and the Szymborski (ZiPS) projection system.
- **Finding claimed:** LSTM beats both.
- **Caution flag:** submitted mid-2022, still shows no journal publication as of this review (2026-09) — over four years unpublished is a real signal that the result didn't hold up to review, or was never resubmitted. Treat any specific numbers from this one skeptically until independently reproduced.

### "Counterfactual Optimization of Baseball Pitch Sequences and Estimation of Its Impact on Season-Level Statistics"
arXiv:2606.17345 (2026).

- **Method:** Generates counterfactual pitch sequences that minimize predicted in-play probability, then estimates the season-level statistical impact of those alternate sequences.
- **Relevance:** Not a direct match for any current sub-problem here (this repo doesn't do pitch-sequencing optimization), but the counterfactual-estimation methodology is a reasonable pattern to borrow if `batters_faced_predictor` or `short_outing_predictor` ever wants to estimate "what would have happened under different pitcher usage" rather than just predicting realized outcomes.

### "Cross-individual generalizability of machine learning models for ball speed prediction in baseball pitching"
arXiv:2605.05487 (2026).

- **Method:** Tests whether ML models trained on one set of pitchers generalize to held-out pitchers for ball-speed prediction — a generalization-gap study, not an accuracy leaderboard.
- **Relevance:** Directly useful methodology reference regardless of subject — this repo's models are all evaluated in-sample-by-time (train/val/test season splits) but haven't explicitly tested cross-player generalization gaps. Worth reading for the evaluation design, separate from its specific ball-speed subject matter.

---

## Tier 3 — Industry, competition, and blog work (ideas only, not citable evidence)

### Kneita — "Transformer-Based Baseball Modeling for Pitch Outcome Prediction and Strategy Optimization"
MIT Sloan Sports Analytics Conference research paper (year not stated on the page); companion Medium series ("Predicting Baseball Pitch Outcomes with Transformer Models") walks through the Statcast data collection and training process in more implementation detail than the paper itself.

- **Method:** Transformer model predicting pitch outcome and hit location, conditioned on recent batter performance and game context.
- **Relevance:** Sloan papers go through a competitive selection process (useful signal of quality) but not formal peer review — read for implementation ideas (the Medium posts are more actionable than the paper abstract), not as validated science.

### "Singlearity: Using A Neural Network to Predict the Outcome of Plate Appearances"
Baseball Prospectus (industry).

- **Method:** Neural network predicting a full outcome distribution (K, BB, HBP, GO, FO, 1B, 2B, 3B, HR) per plate appearance, given batter and pitcher as inputs.
- **Relevance:** Closest published system to what `hit_predictor`/`k_predictor` are actually trying to do — full PA-outcome-distribution modeling rather than a single binary target — but it's a proprietary industry write-up with no published methodology detail to independently verify. Useful for the framing, not for borrowing numbers.

---

## Recommended reading order for this repo specifically

1. **Brill, Deshpande & Wyner (JQAS, TTO penalty)** — read in full first. It's peer-reviewed, directly challenges an assumption already baked into `game_context.py`, and is short enough (it's a stats paper, not a systems paper) to fully absorb in one sitting.
2. **Lee & Kim (CMC, Temporal Fusion Transformer)** — read in full next, specifically to evaluate TFT as the "attention model" candidate named in this repo's own Layer 5 architecture target. Get the full PDF rather than relying on the abstract summarized here.
3. **Zhao et al. (MDPI Information, survey)** — skim as an index, not for depth; use it to find additional papers in whichever specific sub-area (injury, valuation, outcome prediction) becomes the next priority.
4. **Neural Sabermetrics with World Model (arXiv, 2026)** — read for architectural inspiration only; it's unreviewed and six weeks old, so don't let it drive a production decision yet, but the "one autoregressive sequence model per game" framing is a genuinely different approach from this repo's per-target classifier pattern and worth understanding.
5. Everything else in Tier 1 (injury/workload papers) — read opportunistically if/when a durability signal is added to `short_outing_predictor` or `batters_faced_predictor`; not urgent for current work.

## What this review didn't cover

- Full-text access was blocked (paywall, bot-check, or auth wall) for several Tier 1 entries — Kremer et al. (PMC), the Tommy John Journal of Big Data paper, the Sports Engineering pitch-location paper, and the MDPI Applied Sciences comparison. Their summaries here rely on search-result abstracts, not the full paper — re-fetch and verify before citing specific numbers from any of them.
- This pass didn't search for batting-average/BABIP-specific deep learning papers, defensive-positioning/fielding models, or pitch-type classification-from-video work (one hit — the CS231N course project on video-based pitch outcome prediction — surfaced but wasn't investigated; it's a student project, not peer-reviewed, and out of scope for this repo's Statcast-based approach anyway).
