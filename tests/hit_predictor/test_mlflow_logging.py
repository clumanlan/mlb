import re
import subprocess

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from models.hit_predictor.utils.mlflow_logging import (
    get_experiment_name,
    get_git_sha,
    log_evaluation_to_mlflow,
)


@pytest.fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path, monkeypatch):
    """Point MLflow at a throwaway local store so tests never touch ./mlruns
    or leak state into each other."""
    # mlflow 3.x puts the filesystem backend in maintenance mode by default;
    # opt back in rather than switching to a database backend.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    yield
    if mlflow.active_run():
        mlflow.end_run()


def _latest_run(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
    return client, runs[0]


def test_get_experiment_name_returns_hit_predictor():
    assert get_experiment_name() == "hit_predictor"


def test_get_git_sha_returns_full_hex_commit_hash():
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    sha = get_git_sha()

    assert sha == expected
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def test_get_git_sha_returns_none_on_subprocess_failure(monkeypatch):
    def raise_missing_git(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", raise_missing_git)

    assert get_git_sha() is None


def test_log_evaluation_creates_run_with_tags_and_params():
    tags = {"model_type": "xgboost", "stage": "v0-baseline", "target": "is_hit",
            "val_season": "2024"}
    params = {"n_estimators": 100, "max_depth": 6, "VAL_SEASON": 2024}
    metrics = {"brier": 0.123, "log_loss": 0.456}

    log_evaluation_to_mlflow(metrics=metrics, params=params, tags=tags)

    client, run = _latest_run(get_experiment_name())
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == "hit_predictor"
    for key, value in tags.items():
        assert run.data.tags[key] == value
    for key, value in params.items():
        assert run.data.params[key] == str(value)


def test_log_evaluation_logs_scalar_metrics_and_skips_calibration_df():
    calibration_df = pd.DataFrame({
        "bucket": ["0.0-0.5", "0.5-1.0"],
        "n": [10, 10],
        "reliable": [True, True],
        "mean_pred": [0.2, 0.7],
        "obs_hit_rate": [0.18, 0.65],
    })
    metrics = {
        "brier": 0.123,
        "log_loss": 0.456,
        "roc_auc": 0.61,
        "calibration_df": calibration_df,
    }

    log_evaluation_to_mlflow(metrics=metrics, params={}, tags={"model_type": "xgboost"})

    _, run = _latest_run(get_experiment_name())
    assert run.data.metrics["brier"] == pytest.approx(0.123)
    assert run.data.metrics["log_loss"] == pytest.approx(0.456)
    assert run.data.metrics["roc_auc"] == pytest.approx(0.61)
    assert "calibration_df" not in run.data.metrics


def test_log_evaluation_logs_calibration_df_as_csv_artifact():
    calibration_df = pd.DataFrame({
        "bucket": ["0.0-0.5", "0.5-1.0"],
        "n": [10, 10],
        "reliable": [True, True],
        "mean_pred": [0.2, 0.7],
        "obs_hit_rate": [0.18, 0.65],
    })
    metrics = {"brier": 0.123, "calibration_df": calibration_df}

    log_evaluation_to_mlflow(metrics=metrics, params={}, tags={"model_type": "xgboost"})

    client, run = _latest_run(get_experiment_name())
    artifact_names = [f.path for f in client.list_artifacts(run.info.run_id)]
    csv_artifacts = [name for name in artifact_names if name.endswith(".csv")]
    assert len(csv_artifacts) == 1

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=csv_artifacts[0]
    )
    round_tripped = pd.read_csv(local_path)
    assert list(round_tripped.columns) == list(calibration_df.columns)
    assert len(round_tripped) == len(calibration_df)


def test_log_evaluation_logs_additional_artifact_paths(tmp_path):
    png_path = tmp_path / "calibration_curve.png"
    png_path.write_bytes(b"not-a-real-png-but-thats-fine")
    md_path = tmp_path / "baseline_results.md"
    md_path.write_text("# Baseline Results\n")

    log_evaluation_to_mlflow(
        metrics={"brier": 0.1},
        params={},
        tags={"model_type": "xgboost"},
        artifact_paths=[str(png_path), str(md_path)],
    )

    client, run = _latest_run(get_experiment_name())
    artifact_names = {f.path for f in client.list_artifacts(run.info.run_id)}
    assert "calibration_curve.png" in artifact_names
    assert "baseline_results.md" in artifact_names
