# Implementation plan — n_pa_predictor v1 feature-ablation study

Goal: find the bare-minimum feature set for the frozen v1 `low_pa` classifier
before it becomes the first model walked through the repo's FTI feature-store
build order (`ROADMAP.md`'s "Production plan" section). Method: correlation-
cluster the 9 frozen features into redundant families, keep the highest-
importance member of each family, then backward-eliminate remaining weak
features — checking each drop against the locked production CI
(XGBoost @ 0.85 threshold, precision 64.7%, 95% Wilson CI [55.6%, 72.8%],
n=116, val season 2024) rather than trusting raw importance rank alone.

Audit gate (run before this plan started): 631/632 tests passed. The one
failure (`test_integration_odds_fetch.py::test_handler_reraises_when_quota_exceeded`)
is pre-existing and unrelated (stale fixture from the 2026-09-05 QUOTA_LIMIT
fix) — not addressed here.

## Epic: Feature-ablation study for n_pa_predictor's low_pa classifier

### Story 1 — Correlation-based redundancy clustering
Layer: processing (pure function)

- Task 1.1 `cluster_correlated_features()`
  - RED: `tests/n_pa_predictor/test_ablation.py::test_cluster_correlated_features_groups_perfectly_correlated_columns`
  - GREEN: `src/models/n_pa_predictor/processing/features/ablation.py`, Spearman corr + threshold clustering
  - RED: `test_cluster_correlated_features_returns_singletons_when_uncorrelated`
  - GREEN/REFACTOR

### Story 2 — Swap-ablation evaluation harness (reuses the locked protocol exactly)
Layer: experiments / utils

- Task 2.1 Extract `wilson_ci()` out of `train.py` into `src/models/n_pa_predictor/utils/threshold_eval.py`
  - RED: `tests/n_pa_predictor/test_threshold_eval.py::test_wilson_ci_matches_locked_production_ci`
  - GREEN: move function, update `train.py` import
  - Verification (not pytest): rerun `train.py`, confirm identical printed sweep table
- Task 2.2 `evaluate_feature_subset()`
  - RED: `tests/n_pa_predictor/test_ablation.py::test_evaluate_feature_subset_returns_expected_keys_and_ranges`
  - GREEN: implement using Task 2.1's `wilson_ci`, v1's exact XGBoost params
  - REFACTOR

### Story 3 — Redundancy pruning + backward elimination, CI-guarded
Layer: experiments

- Task 3.0 Persist `xgb_model.feature_importances_` as data (currently only exists as a plot) — no new test, straight persistence of an existing attribute
- Task 3.1 `pick_family_representatives()`
  - RED: `test_pick_family_representatives_keeps_highest_importance_member`
  - GREEN/REFACTOR
- Task 3.2 `run_ablation_sweep()` — baseline -> de-duplicated set -> backward elimination, one row per step, stop when a drop's CI stops overlapping the running baseline
  - RED: `test_run_ablation_sweep_output_has_one_row_per_variant`
  - GREEN/REFACTOR
- Task 3.3 `experiments/v1_low_pa_classifier/ablation/run_ablation.py` — thin script reusing `train.py`'s S3 loading + `batter_game` assembly, calls Task 3.2's function, writes `ablation_results.md`. Not unit-tested itself (I/O), per "experiments" layer convention.
- Task 3.4 Write `ablation_results.md` from the real sweep run: correlation clusters, family picks, backward-elimination table, final recommended minimal `FEATURE_COLS` with rationale.

## Before marking any task complete
- [ ] Watched the test FAIL before writing any code
- [ ] Test failed for the right reason
- [ ] Wrote minimal code — nothing extra
- [ ] All tests pass after GREEN
- [ ] No new warnings or errors in pytest output
- [ ] Mocks only where I/O is unavoidable (real code otherwise)
