# Baseball PA Model Dashboard — Spec

**Status: implemented.** Built at top-level `batter_pa_model/` (driver) + `shared/model_dashboard/` (portable package), per the acceptance criteria at the bottom of this doc. Kept as a reference for the design rationale (why each module is/isn't tested, the 3-step error-analysis framework) rather than as a pending to-do — read it to understand *why* the dashboard is shaped this way, not as a build checklist.

A reusable Streamlit dashboard for diagnosing per-plate-appearance binary classification models (batter hit/no-hit, pitcher K/no-K). Targeted TDD on the math that's hard to eyeball; everything else built and verified visually. Built portable from the start — the same dashboard is meant to serve every model in the planned multi-model system (see `CLAUDE.md`'s "Model Layer" section), not just `hit_predictor`.

## Goals

- **EDA**: feature distributions, missingness, signal, player time series
- **Error analysis**: find where the model fails using a 3-step framework (Howard / Ng / Shankar)
- **Calibration**: verify predicted probabilities are trustworthy for betting use
- **Portable**: same dashboard works across batter PA, pitcher PA, future PA-grain models via config object

---

## TDD scope — what gets tested and why

This is a solo dashboard built to speed up modeling work. Full TDD coverage would slow that down without proportional payoff. Test only what's hard to verify by eyeballing the dashboard.

| Module | Tested? | Why |
|---|---|---|
| `logic/bootstrap.py` | **Yes — full suite** | Math is subtle, reproducibility matters, performance test catches Python-loop trap |
| `logic/contribution.py` | **Yes — math + ranking** | Whole prioritization depends on the formula; ranking is testable with known-biased data |
| `logic/metrics.py` (per-row log loss) | **One sklearn-match test** | Cheap insurance against a silent bug that would corrupt everything downstream |
| Everything else (plots, binning, slicing, missing, calibration bins, config) | **No** | You'll see bugs the moment you open the dashboard. Don't write tests just to write tests. |

If something breaks later in a non-obvious way, *that's* when you add a regression test for it.

---

## Architecture

```
project/
├── batter_pa_model/
│   ├── dashboard.py              # ~15-line driver: CONFIG + run_dashboard(CONFIG)
│   ├── data.csv
│   └── ...
├── pitcher_pa_model/
│   ├── dashboard.py              # same shape, different CONFIG
│   └── ...
├── shared/
│   └── model_dashboard/
│       ├── __init__.py
│       ├── logic/                # PURE functions — no streamlit, no plotly
│       │   ├── __init__.py
│       │   ├── metrics.py        # per_row_loss, slice_log_loss
│       │   ├── slicing.py        # single + interaction slice generation
│       │   ├── contribution.py   # TESTED — contribution math + ranking
│       │   ├── bootstrap.py      # TESTED — bootstrap CI with fixed seed
│       │   ├── binning.py        # numeric/categorical binning
│       │   ├── calibration.py    # reliability bins
│       │   └── missing.py        # null counts per column
│       ├── plots/                # plotly figure builders
│       │   ├── __init__.py
│       │   ├── distribution.py
│       │   ├── feature_vs_outcome.py
│       │   ├── time_series.py
│       │   ├── forest.py
│       │   ├── reliability.py
│       │   └── missing.py
│       ├── components/           # streamlit-aware UI pieces
│       │   ├── __init__.py
│       │   ├── description_block.py
│       │   ├── header.py
│       │   └── tables.py
│       ├── config.py             # CONFIG dataclass (no validation tests)
│       └── app.py                # run_dashboard() — streamlit orchestration
└── tests/
    └── model_dashboard/
        ├── __init__.py
        ├── conftest.py           # shared fixtures
        ├── test_bootstrap.py     # full suite
        ├── test_contribution.py  # full suite
        └── test_metrics.py       # one sklearn-match test
```

---

## Data assumptions — read this before building

The dashboard expects a **single joined CSV per model folder** containing features + target + predictions. All columns are already in this CSV — the dashboard does NOT fetch anything from S3, an API, or any other source.

**Important context for `batting_order`**: this column is already in the joined CSV. It comes from boxscore data that was joined into the play-by-play dataframe upstream. Do not look for a separate lineups dataset.

Example schema for batter model:
- `is_hit` (target, 0/1)
- `pred_prob` (model output, 0–1)
- `batter_name`, `game_date`
- `batting_order`, `batSide`, `pitcher_hand`, `game_season`
- `weather_condition`, `weather_temp`
- `weight`, `height_in_inches`, `strikeZoneTop`, `strikeZoneBottom`, `last_season_ba`

---

## Driver file (per model folder)

```python
from shared.model_dashboard import run_dashboard

CONFIG = {
    "model_name": "Batter PA — hit/no hit",
    "target": "is_hit",
    "pred": "pred_prob",
    "entity": "batter_name",
    "date": "game_date",
    "slice_cols": ["batSide", "pitcher_hand", "batting_order", "weather_condition"],
    "interaction_pairs": [
        ("batSide", "pitcher_hand"),
        ("batting_order", "pitcher_hand"),
    ],
    "data_path": "data.csv",
}

run_dashboard(CONFIG)
```

Switching to pitcher model: copy the folder, change `target` (e.g. `is_strikeout`) and `entity` (e.g. `pitcher_name`), point at new CSV.

---

## Stack

- **Streamlit** — UI / layout / dropdowns / tabs
- **Plotly** — all charts
- **Pandas / NumPy** — data manipulation
- **scikit-learn** — `log_loss` reference (used in tests)
- **pytest** — testing (only for tested modules above)

---

## Shared test fixtures

For the few modules that get tests, put fixtures in `tests/model_dashboard/conftest.py`.

```python
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def toy_pa_data():
    """8-row deterministic PA dataset for fast unit tests."""
    return pd.DataFrame({
        "is_hit":        [0, 1, 0, 1, 0, 1, 0, 1],
        "pred_prob":     [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        "batSide":       ["L", "L", "R", "R", "L", "L", "R", "R"],
        "pitcher_hand":  ["L", "R", "L", "R", "L", "R", "L", "R"],
    })

@pytest.fixture
def biased_slice_data():
    """Data where batSide=L slice has clearly worse log loss than overall.
    Used to verify slice ranking puts the bad slice on top."""
    rng = np.random.default_rng(42)
    n = 1000
    bat_side = rng.choice(["L", "R"], size=n)
    is_hit = rng.binomial(1, 0.3, size=n)
    pred_prob = np.where(
        bat_side == "L",
        rng.uniform(0.7, 0.9, size=n),   # confidently wrong on L
        rng.uniform(0.2, 0.4, size=n),   # well-calibrated on R
    )
    return pd.DataFrame({
        "is_hit": is_hit, "pred_prob": pred_prob, "batSide": bat_side,
    })
```

---

## The 3 tested modules

### `logic/metrics.py` — one test only

**TASK**: Per-row log loss matches sklearn.

RED — `tests/model_dashboard/test_metrics.py`
- `test_per_row_loss_matches_sklearn(toy_pa_data)` — for each row, our `per_row_loss(y, p)` should equal `sklearn.metrics.log_loss([y], [p], labels=[0,1])`

GREEN — `shared/model_dashboard/logic/metrics.py`
- Implement `per_row_loss(y, p, eps=1e-15) -> float` with clipping

Also implement `slice_log_loss(df, target_col, pred_col)` in the same file — no test for it (it's a one-line mean over per-row losses).

---

### `logic/contribution.py` — math + ranking

RED — `tests/model_dashboard/test_contribution.py`
- `test_contribution_formula(biased_slice_data)` — contribution = pct × (slice_loss − overall_loss) for the batSide=L slice; verify with hand calculation
- `test_contribution_ranking_puts_bad_slice_on_top(biased_slice_data)` — the deliberately-biased L slice ranks #1
- `test_contribution_filters_below_min_n()` — slices with N < min_n excluded (default min_n=50)
- `test_contribution_returns_expected_columns()` — output has `[feature, value, n, pct, slice_loss, delta, contribution]`

GREEN — `shared/model_dashboard/logic/contribution.py`
- `compute_contribution(df, slices, target_col, pred_col, min_n=50) -> pd.DataFrame` sorted desc by contribution

---

### `logic/bootstrap.py` — full suite (most important)

> The performance test catches a Python-loop implementation. If first attempt uses `for _ in range(n_iter):`, it fails the 5-second budget on 50k rows. Vectorize by generating all sample indices at once: `rng.integers(0, n, size=(n_iter, n))`, then compute log loss per row of the resulting matrix in numpy.

RED — `tests/model_dashboard/test_bootstrap.py`
- `test_bootstrap_ci_reproducible_with_seed(biased_slice_data)` — same seed → identical (lo, hi) across calls
- `test_bootstrap_ci_ordering()` — lo < point_estimate < hi
- `test_bootstrap_ci_widens_with_small_n()` — same population, n=50 sample has wider CI than n=10000 sample
- `test_bootstrap_ci_performance()` — 50k rows × 1000 iterations completes in under 5 seconds

GREEN — `shared/model_dashboard/logic/bootstrap.py`
- `bootstrap_ci(df, target_col, pred_col, n_iter=1000, seed=42) -> tuple[float, float, float]` returning (point, lo, hi)
- Must vectorize per teaching note above

---

## Everything else — build without tests

Build these in this order, eyeball-verify each by running `streamlit run` and clicking around. No unit tests.

### `logic/` (untested)
- `slicing.py::generate_single_slices(df, slice_cols)` + `generate_interaction_slices(df, interaction_pairs)` — pandas groupby wrappers
- `binning.py::bin_feature_by_outcome(df, feature_col, outcome_col)` — numeric → 10 quantile bins, categorical → all values; returns DataFrame `[bin_label, mean_outcome, n]`
- `missing.py::missing_value_report(df)` — returns `[feature, pct_missing]` sorted desc
- `calibration.py::reliability_bins(df, target_col, pred_col, n_bins=10)` — returns `[bin_mid, pred_mean, actual_rate, n]`

### `plots/` (untested)
All take a DataFrame + maybe options, return a `plotly.graph_objects.Figure`:
- `distribution.py::build_distribution_figure(df, column)` — auto-detect numeric → histogram, categorical → bar
- `feature_vs_outcome.py::build_feature_vs_outcome_figure(binned_df, overall_rate)` — bars green if > overall_rate, red if below, dashed reference line
- `time_series.py::build_time_series_figure(df, entity, metric, slice_by=None)` — multi-line if slice_by set, dots colored by outcome
- `forest.py::build_forest_figure(slice_ci_df, overall_loss)` — horizontal CI lines, vertical reference line at overall
- `reliability.py::build_reliability_figure(reliability_df)` — scatter + diagonal reference + confidence histogram subplot
- `missing.py::build_missing_figure(missing_df)` — horizontal bar chart

### `components/` (untested)
- `description_block.py::render_description(what, why, math=None, read=None)` — styled `st.markdown` block, label column ~56px, monospace for `math`
- `header.py::render_header(config, summary_stats)` — model name + config pills + 4 metric cards
- `tables.py::render_top_losses_table(df)`, `render_contribution_table(df)` — wrap `st.dataframe` with formatting

### `app.py` (untested)
`run_dashboard(config_dict)`:
1. Validate config inline (just dict.get with defaults)
2. Load CSV from `config["data_path"]`
3. Render header
4. Render `st.tabs(["EDA", "Error analysis", "Calibration"])`
5. Each tab calls its plot/table builders

---

## Layout

### Header
- Model name from config
- Config pills showing target + entity
- 4 summary metric cards: Total PAs, Positive rate, Overall log loss, Top slice to fix

### Description block
Each panel sits above this block:
```
what  →  one-line summary of the chart
why   →  why it matters for diagnosis
math  →  formula (only when relevant)
read  →  how to interpret what you see
```

### EDA tab — 4 sections
1. **Feature distribution** — dropdown → histogram or bar
2. **Missing values** — horizontal bar chart of % null per feature
3. **Feature vs outcome** — dropdown → binned mean plot, green/red bars, reference line at overall rate
4. **Entity time series** — player + metric + slice-by + season dropdowns → multi-line plot, dots colored by outcome

### Error analysis tab — 3 sections
1. **Step 1 — top losses (Howard)** — top 20 highest-loss rows in `st.dataframe`; toggle for "high-confidence wrong only"
2. **Step 2 — slice contribution (Ng)** — contribution table sorted desc, "fix this" badge
3. **Step 3 — CI check (Shankar)** — forest plot of top 15 slices from Step 2

Display label: "95% CI (bootstrap, assumes within-season exchangeability)"

### Calibration tab — 1 section
**Reliability diagram + confidence histogram** sharing x-axis.

---

## Build order

1. **Audit gate**: `pytest --tb=short -q` and `flake8` — fix any existing failures before adding new code
2. **TDD round** (one task at a time, RED → GREEN → REFACTOR):
   - Per-row log loss
   - Contribution math + ranking
   - Bootstrap CI (most important)
3. **Build the rest of `logic/`** without tests
4. **Build `plots/`** without tests
5. **Build `components/` and `app.py`** without tests
6. **Smoke test** with `streamlit run batter_pa_model/dashboard.py`
7. **Verify portability** by copying to pitcher folder and changing CONFIG

---

## What's NOT in v1

- Feature correlation matrix
- Outcome rate over time
- Prediction distribution histogram
- ECE / Brier score summary numbers
- Calibration by slice
- Block bootstrap (more rigorous CI for autocorrelated data)
- Auto column detection — config object replaces this
- Tests beyond the three modules above — add regression tests later if real bugs surface

---

## Acceptance criteria

The dashboard is done when:

1. `pytest tests/model_dashboard/` passes (3 test files, all green)
2. `streamlit run batter_pa_model/dashboard.py` launches and all three tabs render
3. Switching to pitcher model = copy folder, change ~3 CONFIG lines, point at new CSV
4. Bootstrap CI completes in < 5 seconds for 50k row test set
5. Every panel has a description block above it
