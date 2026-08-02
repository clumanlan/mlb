# Implementation Plan — MLflow tracking for hit_predictor

## EPIC: MLflow tracking for hit_predictor
Why: every `evaluate_hit_predictor` run currently only prints to stdout — nothing is
comparable across models/runs. MLflow gives params/metrics/tags/artifacts per run
in one queryable experiment.
Depends on: none

---

### STORY 1: Dependency wired in
Acceptance: `mlflow` importable in the venv; `mlruns/` never committed.
Layer: config (no production code, no TDD cycle needed)

TASK 1.1: Add mlflow to requirements.txt, install, gitignore mlruns/
  - No RED/GREEN — this is dependency/config, not testable behavior.
  - `pip install mlflow`, pin version in requirements.txt, add `mlruns/` to `.gitignore`.

---

### STORY 2: `mlflow_logging.py` utility module
Acceptance: pure, testable functions that turn an `evaluate_hit_predictor` result
into an MLflow run. Zero MLflow references added to `eval.py` itself.
Layer: experiments (hit_predictor/utils)
File: `src/models/hit_predictor/utils/mlflow_logging.py`
Tests: `tests/hit_predictor/test_mlflow_logging.py`

Design decisions locked in during planning:
- `get_experiment_name()` resolves off `mlflow_logging.py`'s own `__file__`
  (`.parent.parent.name`, i.e. `utils/`'s parent) — stable regardless of how deep
  a calling training script is nested, unlike deriving it from the caller's `__file__`.
- `log_evaluation_to_mlflow()` owns `mlflow.set_experiment()` + `mlflow.start_run()`
  internally — call sites (train.py) just call it once per model with the raw
  `evaluate_hit_predictor()` return dict, params dict, tags dict, and an optional
  list of extra artifact file paths (PNG, `baseline_results.md`). Filtering for
  "if present" (e.g. `baseline_results.md`) is the caller's job — the function logs
  whatever paths it's given.
- `calibration_df` is stripped out of the metrics dict before `mlflow.log_metrics`
  (it's a DataFrame, not a scalar) and written to a temp CSV logged as an artifact
  instead.
- `get_git_sha()` never raises — wrapped in try/except per spec, returns `None` on
  failure so a run outside a git repo (or git missing) doesn't hard-fail.
- Tracking URI (`file:./mlruns` vs. a future remote server) is NOT set inside
  `mlflow_logging.py` — that's the call site's responsibility (one line in train.py),
  which is also what makes the module trivially testable against a tmp tracking URI.

TASK 2.1: `get_experiment_name()` resolves to "hit_predictor"
  RED:
    - File: tests/hit_predictor/test_mlflow_logging.py
    - Test: test_get_experiment_name_returns_hit_predictor
    - Assert: `get_experiment_name() == "hit_predictor"`
    - Run pytest, confirm it fails with ImportError (function doesn't exist yet)
  GREEN:
    - File: src/models/hit_predictor/utils/mlflow_logging.py
    - `EXPERIMENT_NAME = Path(__file__).resolve().parent.parent.name`
    - `def get_experiment_name(): return EXPERIMENT_NAME`
  REFACTOR: none needed yet

TASK 2.2: `get_git_sha()` — happy path + failure path
  RED:
    - Test: test_get_git_sha_returns_full_hex_commit_hash
      Assert: running inside this repo returns a 40-char lowercase hex string
      matching `git rev-parse HEAD`.
    - Test: test_get_git_sha_returns_none_on_subprocess_failure
      Mock `subprocess.run` to raise `FileNotFoundError` (git missing) — assert
      `get_git_sha()` returns `None`, not an exception.
    - Confirm both fail (ImportError) before implementation.
  GREEN:
    - `get_git_sha()` wraps `subprocess.run(["git","rev-parse","HEAD"], capture_output=True, text=True)`
      in try/except; also treats non-zero returncode as failure → `None`.
  REFACTOR: none needed yet

TASK 2.3: `log_evaluation_to_mlflow()` creates a run under the "hit_predictor"
experiment with correct tags and params
  RED:
    - Test: test_log_evaluation_creates_run_with_tags_and_params
    - Fixture: point `mlflow.set_tracking_uri` at `tmp_path` before calling.
    - Assert via `MlflowClient`: run's experiment name == "hit_predictor",
      run tags include the given tags, run params include the given params
      (string-coerced, as MLflow stores them).
  GREEN: implement `log_evaluation_to_mlflow(metrics, params, tags, artifact_paths=None)`
    calling `mlflow.set_experiment(get_experiment_name())`, `mlflow.start_run()`,
    `mlflow.set_tags(tags)`, `mlflow.log_params(params)`.
  REFACTOR: none needed yet

TASK 2.4: scalar metrics logged, `calibration_df` excluded from `log_metrics`
  RED:
    - Test: test_log_evaluation_logs_scalar_metrics_and_skips_calibration_df
    - Pass a metrics dict shaped like `evaluate_hit_predictor`'s return value
      (scalars + a `calibration_df` DataFrame key).
    - Assert every scalar shows up in `run.data.metrics`, and that logging
      does not raise (i.e. the DataFrame never hits `log_metrics`).
  GREEN: split metrics dict into scalar vs. `calibration_df` before logging.
  REFACTOR: none needed yet

TASK 2.5: `calibration_df` logged as a CSV artifact
  RED:
    - Test: test_log_evaluation_logs_calibration_df_as_csv_artifact
    - Assert an artifact ending in `.csv` exists on the run and, when read back,
      round-trips the calibration_df's columns.
  GREEN: write `calibration_df` to a temp file, `mlflow.log_artifact(...)`.
  REFACTOR: none needed yet

TASK 2.6: extra `artifact_paths` (calibration PNG, `baseline_results.md`) logged
  RED:
    - Test: test_log_evaluation_logs_additional_artifact_paths
    - Create a real tmp `.png` file and a tmp `.md` file, pass both via
      `artifact_paths=[...]`, assert both filenames appear in the run's artifacts.
  GREEN: iterate `artifact_paths or []`, `mlflow.log_artifact(path)` each.
  REFACTOR: after 2.3–2.6 are green, tidy `log_evaluation_to_mlflow` into clear
    sections (tags/params → metrics → calibration artifact → extra artifacts);
    tests must stay green throughout.

---

### STORY 3: Wire into `train.py`
Acceptance: each of the two `evaluate_hit_predictor` calls in
`experiments/v1_pitcher_features/train.py` is followed by one MLflow run capturing
that model's tags/params/metrics/artifacts.
Layer: experiments (glue/wiring)

TASK 3.1: Wrap both call sites
  No RED/GREEN — `train.py` is a flat top-to-bottom script that reads real data
  from S3 at import time, so it isn't unit-testable without restructuring it
  (explicitly out of scope per your answer). `log_evaluation_to_mlflow` itself is
  fully covered by Story 2's tests; this task is wiring a tested function into an
  untested script, the same trust boundary as any other script → library call.
  Verified by manual smoke run (see checklist) instead of pytest.
  - `mlflow.set_tracking_uri("file:./mlruns")` once near the top of train.py.
  - Set `STAGE = "v0-baseline"` (or current value) as a module-level var the user
    bumps by hand as the project progresses.
  - For each of "Logistic regression" / "XGBoost": call
    `evaluate_hit_predictor(...)` (unchanged), then
    `log_evaluation_to_mlflow(metrics=..., params={...}, tags={...}, artifact_paths=[...])`.
  - `tags`: `model_type` ("logistic_regression" / "xgboost"), `stage=STAGE`,
    `target="is_hit"`, `val_season=str(VAL_SEASON)`, `git_sha=get_git_sha()`.
  - `params`: whatever hyperparameters were passed to that model's constructor,
    plus `FIT_SEASONS`, `VAL_SEASON`, `TEST_SEASON`, `n_bins`, `min_n`.
  - `artifact_paths`: calibration curve PNG (from `plot_calibration_curve`'s
    `save_path`), plus `baseline_results.md` only if `Path(...).exists()`.

---

## Before marking any task complete:
- [x] Watched the test FAIL before writing any code
- [x] Test failed for the right reason (feature missing, not a typo)
- [x] Wrote minimal code — nothing extra
- [x] All tests pass after GREEN
- [x] No new warnings or errors in pytest output
- [x] Tests use real code — mocks only where I/O is unavoidable (subprocess failure
      path in 2.2 is the one legitimate mock; MLflow runs are tested against a real
      local file-store tracking URI, not mocked, since that I/O is fast and local)

## Deviation found during implementation
mlflow 3.x puts the plain `file:./mlruns` tracking backend in "maintenance mode" and
raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set (or you switch to a database
backend like sqlite). Confirmed with the user: kept `file:./mlruns`, set the env var
in `train.py` (`os.environ.setdefault(...)`) and in the test fixture
(`monkeypatch.setenv`). mlflow was pinned to 3.15.0 rather than the 2.x line because
2.x requires `pyarrow<20`, which conflicts with this project's `pyarrow==20.0.0` pin
(needed by awswrangler); pyarrow was verified to stay at 20.0.0 after the mlflow
install and the full test suite (150 tests) still passes.
