import os

# Must be set before feast or xgboost is ever imported anywhere in the suite
# (root conftest.py runs first, before test collection) — confirmed 2026-09-15
# that importing feast, then later training an XGBoost model in the SAME
# process, segfaults inside xgboost/core.py's OpenMP init (not a hang like
# the earlier documented torch/pyarrow OpenMP conflict — a hard crash,
# reproducible, order-dependent: xgboost-then-feast is fine, feast-then-
# xgboost is not). Once tests/features/test_feature.py imports feast (for
# the FTI feature-store work), any full `pytest tests/` run collects it
# alphabetically before tests/n_pa_predictor/'s XGBoost-training tests, so
# this isn't an edge case — it reproduces on a plain `pytest tests/ -v`.
# See DECISIONS.md's 2026-09-15 entry and
# feedback_pytorch_openmp_macos_deadlock.md.
os.environ.setdefault("OMP_NUM_THREADS", "1")


def pytest_addoption(parser):
    parser.addoption('--date', action='store', default=None, help='Game date YYYY-MM-DD')
