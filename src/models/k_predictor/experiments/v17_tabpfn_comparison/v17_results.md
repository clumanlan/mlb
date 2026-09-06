# v17 Results — k_predictor (TabPFN vs. XGBoost)

**Date:** 2026-09-05
**Task:** Binary classification (PA grain) — is this a real production-candidate pass, testing TabPFN as an alternative to XGBoost
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC / ROC-AUC (PA grain), same protocol every version since v7
**Data:** `experiments/count_distribution_check/_model_cache/model_df_v6.parquet` (cached, not rebuilt from S3)

## Why this experiment

Prompted by a session-level research question: is there a real case for deep learning / neural approaches over XGBoost for k_predictor? The research finding (Grinsztajn et al., NeurIPS 2022) is that GBTs still beat from-scratch neural nets on tabular data at this project's scale. TabPFN is a different comparison — a transformer *pretrained* on millions of synthetic tabular tasks (in-context learning, not trained from scratch here) — with published wins specifically in the small-to-medium regime it targets (official ceiling: 10,000 training rows, 500 features). This is the cheap, honest way to check whether that transfers to a real strikeout target, using v6's exact 42 pre-game features with zero new feature engineering.

## Design — apples-to-apples

TabPFN's row ceiling (10,000) is far below k_predictor's full fit pool (411,861 PA rows across FIT_SEASONS). Comparing TabPFN-on-a-subsample against v6's full-data XGBoost fit would confound "different model" with "different training set size." So this experiment trains two things on the identical 8,000-row stratified subsample (`stratified_subsample`, TDD'd in `utils/sampling.py`, preserves the ~21.8% strikeout base rate, `random_state=42`):

1. **TabPFN** — fit on the 8,000-row subsample directly (no early-stopping split needed).
2. **XGBoost (same subsample)** — v6's exact tuned hyperparameters (`max_depth=2`, `learning_rate=0.03`), with a 90/10 stratified split of the subsample providing the early-stopping monitor.

A third model, **XGBoost v6 (full pool)**, is v6's actual cached fit on all 411,861 rows — kept as a reference line only, not the primary verdict, since it differs in training set size as well as nothing else being held constant.

All three are evaluated on the identical, untouched VAL_SEASON=2024 population (105,265 PA rows). TEST_SEASON (2025) stays untouched per `config.yaml`.

## Results (VAL_SEASON=2024, 105,265 PA rows, 42 pre-game features)

| Model | Train rows | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| TabPFN (subsample) | 8,000 | 0.2790 | 0.5957 |
| XGBoost (same subsample) | 8,000 | 0.2743 | 0.5891 |
| XGBoost v6 (full pool, reference) | 411,861 | 0.2838 | 0.5996 |

**Apples-to-apples delta (TabPFN − XGBoost, same 8,000-row subsample): PR-AUC +0.0047, ROC-AUC +0.0066.**

## Interpretation

TabPFN beats XGBoost when both are restricted to the same 8,000-row subsample — a real, directionally consistent edge on both metrics, though the PR-AUC delta (+0.0047) sits right at this project's own ~0.005 "real" threshold rather than clearly clearing it. This matches TabPFN's published niche: at small sample sizes, its pretraining prior helps more than a tree ensemble can extract from scratch.

But that edge doesn't survive contact with the actual production question. XGBoost trained on the **full** 411,861-row pool (0.2838 PR-AUC) still beats TabPFN's small-sample result (0.2790) by a larger margin (+0.0048) than TabPFN's own edge over XGBoost-on-the-same-subsample. In other words: more data beats TabPFN's small-sample advantage, and k_predictor already has more data than TabPFN can use. This is exactly the outcome the research finding predicted — GBTs win at k_predictor's real scale; TabPFN's advantage is confined to a regime this project has already outgrown.

**v6's tuned XGBoost on the full pool (PR-AUC 0.2838) remains the standing production candidate.** TabPFN is not worth pursuing further for the current pre-game total-strikeouts prop use case — its ceiling caps it below what the existing model already achieves with the data actually available.

## Operational notes

- **A real deadlock, not a slow run:** the first attempt hung for 2h42m using 5 seconds of CPU time. A stack sample (`sample <pid>`) showed the main thread stuck in `torch::autograd::THPVariable_linalg_qr` → an OpenMP join barrier — a known PyTorch/OpenMP conflict on macOS when more than one OpenMP runtime (PyTorch's bundled one, pyarrow's own thread pool) loads into the same process. Fixed by setting `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)` before any TabPFN call — all set at the top of `train.py` before `torch` is imported. Worth remembering for any future PyTorch use in this repo on this machine.
- **TabPFN requires a one-time external gate**: a Prior Labs account (ux.priorlabs.ai) with the license explicitly accepted on the Licenses tab (a separate step from generating the API key), plus `TABPFN_TOKEN` set in the environment. Not something automatable end-to-end.
- **Inference is slow on CPU**: ~1,050s (~17.5 minutes) to score all 105,265 VAL_SEASON rows in batches of 2,000, vs. XGBoost's near-instant inference. TabPFN's cost scales with train_size × test_size per ensemble member — a real operational cost even setting aside the accuracy result.

## Next steps

- Not pursued further — the result is a clean, real negative (or at best a wash) for TabPFN as a production candidate here, consistent with the broader literature. `stratified_subsample` (`utils/sampling.py`) is kept as reusable infrastructure for any future experiment needing a row-ceiling-safe subsample.
- If a future k_predictor sub-problem ever has a genuinely small row count (well under 10,000, e.g. a rare-event slice), TabPFN would be worth revisiting there specifically — this result doesn't rule that out, it just rules out replacing the current full-data XGBoost pipeline.
