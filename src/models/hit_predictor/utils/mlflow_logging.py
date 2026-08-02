"""
utils/mlflow_logging.py

Turns the dict returned by eval.py's evaluate_hit_predictor() into an MLflow
run. eval.py itself stays MLflow-agnostic — this module is the only place
that imports mlflow, so swapping tracking backends later only touches here
and the tracking-URI line at each call site.
"""

import subprocess
import tempfile
from pathlib import Path

import mlflow

# Anchored to this file's own location (utils/'s parent), not the caller's
# __file__ — stable regardless of how deeply a training script is nested
# under hit_predictor/.
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


def log_evaluation_to_mlflow(metrics, params, tags, artifact_paths=None):
    """
    Log one evaluate_hit_predictor() run to MLflow.

    Parameters
    ----------
    metrics : dict
        The dict returned by evaluate_hit_predictor(). Its `calibration_df`
        entry (a DataFrame) is logged as a CSV artifact instead of a metric;
        every other entry is logged as a scalar metric.
    params : dict
        Hyperparameters / run config to log.
    tags : dict
        Run tags (model_type, stage, target, val_season, git_sha, ...).
    artifact_paths : list of str, optional
        Extra files to attach to the run (e.g. calibration curve PNG,
        baseline_results.md). Caller decides which paths to include.
    """
    mlflow.set_experiment(get_experiment_name())
    with mlflow.start_run():
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
