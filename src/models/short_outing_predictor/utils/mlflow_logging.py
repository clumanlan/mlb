"""
utils/mlflow_logging.py

Generic MLflow run logger, copied from bb_predictor's / k_predictor's /
hit_predictor's utils/mlflow_logging.py (anchored to this file's own
location, so it "just works" once copied into a new model directory —
EXPERIMENT_NAME resolves to "short_outing_predictor" here). Not coupled to
any hit_predictor-specific eval function; the caller builds its own
metrics dict.
"""

import subprocess
import tempfile
from pathlib import Path

import mlflow

# Anchored to this file's own location (utils/'s parent), not the caller's
# __file__ — stable regardless of how deeply a training script is nested
# under short_outing_predictor/.
EXPERIMENT_NAME = Path(__file__).resolve().parent.parent.name


def get_experiment_name():
    return EXPERIMENT_NAME


def get_git_sha():
    """Full commit hash of HEAD, or None if unavailable (no git, not a repo)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def create_run_id():
    """Open and immediately close a new MLflow run under this experiment,
    returning its run_id. Lets a caller namespace filesystem artifacts
    (plots) by the REAL run_id before any of those files are written."""
    mlflow.set_experiment(get_experiment_name())
    with mlflow.start_run() as run:
        return run.info.run_id


def log_evaluation_to_mlflow(metrics, params, tags, artifact_paths=None, run_id=None):
    """
    Log one evaluation run to MLflow.

    Parameters
    ----------
    metrics : dict
        Scalar metrics to log. An optional `calibration_df` entry (a
        DataFrame) is logged as a CSV artifact instead of a metric.
    params : dict
        Hyperparameters / run config to log.
    tags : dict
        Run tags (model_type, stage, target, val_season, git_sha, ...).
    artifact_paths : list of str, optional
        Extra files to attach to the run (plots, results.md).
    run_id : str, optional
        An existing run to resume (e.g. from create_run_id()).
    """
    mlflow.set_experiment(get_experiment_name())
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags(tags)
        mlflow.log_params(params)

        calibration_df = metrics.get("calibration_df")
        scalar_metrics = {k: v for k, v in metrics.items() if k != "calibration_df"}
        mlflow.log_metrics(scalar_metrics)

        if calibration_df is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = Path(tmp_dir) / "calibration.csv"
                calibration_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(str(csv_path))

        for path in artifact_paths or []:
            mlflow.log_artifact(str(path))
