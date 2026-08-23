import json

import mlflow
import pytest

from models.hit_predictor.utils.run_comparison import (
    build_comparison_report,
    compute_feature_importance_diff,
    compute_metric_deltas,
    compute_verdict_for_report,
    render_comparison_html,
)


@pytest.fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path, monkeypatch):
    """Same isolation pattern as test_mlflow_logging.py — point MLflow at a
    throwaway local store so this test never touches ./mlruns."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    yield
    if mlflow.active_run():
        mlflow.end_run()


def test_compute_metric_deltas_computes_delta_and_pct_change():
    """delta = new - old, pct_change = delta / old — hand-computable:
    old=0.60, new=0.66 -> delta=0.06, pct_change=0.10 (10% relative gain)."""

    old_metrics = {"roc_auc": 0.60}
    new_metrics = {"roc_auc": 0.66}

    result = compute_metric_deltas(old_metrics, new_metrics)

    assert len(result) == 1
    row = result[0]
    assert row["metric"] == "roc_auc"
    assert row["old"] == pytest.approx(0.60)
    assert row["new"] == pytest.approx(0.66)
    assert row["delta"] == pytest.approx(0.06)
    assert row["pct_change"] == pytest.approx(0.10)


def test_compute_metric_deltas_includes_metrics_present_in_only_one_run():
    """A metric only logged on one run (e.g. a metric added between
    versions) must still appear, with None standing in for the missing
    side rather than the row being silently dropped."""

    old_metrics = {"roc_auc": 0.60}
    new_metrics = {"roc_auc": 0.66, "calib_eval_raw_ece": 0.05}

    result = compute_metric_deltas(old_metrics, new_metrics)
    by_name = {row["metric"]: row for row in result}

    assert by_name["calib_eval_raw_ece"]["old"] is None
    assert by_name["calib_eval_raw_ece"]["new"] == pytest.approx(0.05)
    assert by_name["calib_eval_raw_ece"]["delta"] is None


def test_compute_metric_deltas_sorted_by_metric_name():
    old_metrics = {"zeta": 1.0, "alpha": 1.0}
    new_metrics = {"zeta": 1.0, "alpha": 1.0}

    result = compute_metric_deltas(old_metrics, new_metrics)

    assert [row["metric"] for row in result] == ["alpha", "zeta"]


def test_compute_feature_importance_diff_flags_added_and_removed_features():
    old = {"a": 0.5, "b": 0.3}
    new = {"a": 0.5, "c": 0.2}

    result = compute_feature_importance_diff(old, new)

    assert result["added"] == ["c"]
    assert result["removed"] == ["b"]


def test_compute_feature_importance_diff_top_shifts_sorted_by_abs_delta_desc():
    """Feature 'a' moves +0.10 (0.10->0.20), 'b' moves -0.30 (0.40->0.10) —
    'b' has the larger absolute shift so it must rank first."""

    old = {"a": 0.10, "b": 0.40}
    new = {"a": 0.20, "b": 0.10}

    result = compute_feature_importance_diff(old, new, top_n=5)

    shifts = result["top_shifts"]
    assert [s["feature"] for s in shifts] == ["b", "a"]
    assert shifts[0]["delta"] == pytest.approx(-0.30)
    assert shifts[1]["delta"] == pytest.approx(0.10)


def test_compute_feature_importance_diff_top_shifts_respects_top_n():
    old = {f"f{i}": 0.01 * i for i in range(10)}
    new = {f"f{i}": 0.01 * i + 0.05 for i in range(10)}

    result = compute_feature_importance_diff(old, new, top_n=3)

    assert len(result["top_shifts"]) == 3


def _make_run_data(run_id, roc_auc, importances):
    return {
        "run_id": run_id, "metrics": {"roc_auc": roc_auc}, "params": {},
        "tags": {}, "start_time": 0, "importances": importances,
    }


# ── compute_verdict_for_report ──────────────────────────────────────────────
# Per BENCHMARKS.md §2, game grain is the primary evaluation target, so this
# report should prefer the game_grain_reliability/game_grain_resolution keys
# (as logged by run_pa_vs_game_grain_check + log_evaluation_to_mlflow) over
# PA-grain ones when both runs have them.

def test_compute_verdict_for_report_uses_game_grain_keys_when_present():
    old_metrics = {"reliability": 0.0013, "resolution": 0.0001,
                    "game_grain_reliability": 0.0030, "game_grain_resolution": 0.0182}
    new_metrics = {"reliability": 0.0005, "resolution": 0.0002,
                    "game_grain_reliability": 0.0015, "game_grain_resolution": 0.0159}

    result = compute_verdict_for_report(old_metrics, new_metrics)

    # Game-grain deltas (0.0015-0.0030, 0.0159-0.0182), not PA-grain ones.
    assert result["reliability_delta"] == pytest.approx(0.0015 - 0.0030)
    assert result["resolution_delta"] == pytest.approx(0.0159 - 0.0182)


def test_compute_verdict_for_report_falls_back_to_pa_grain_keys():
    # No game_grain_* keys on either run (e.g. an old pre-game-grain run) —
    # falls back to the plain PA-grain reliability/resolution.
    old_metrics = {"reliability": 0.0013, "resolution": 0.0001}
    new_metrics = {"reliability": 0.0005, "resolution": 0.0003}

    result = compute_verdict_for_report(old_metrics, new_metrics)

    assert result["reliability_delta"] == pytest.approx(0.0005 - 0.0013)
    assert result["resolution_delta"] == pytest.approx(0.0003 - 0.0001)


def test_compute_verdict_for_report_returns_none_when_metrics_missing():
    old_metrics = {"roc_auc": 0.60}
    new_metrics = {"roc_auc": 0.66}

    result = compute_verdict_for_report(old_metrics, new_metrics)

    assert result is None


def test_render_comparison_html_includes_verdict_banner_when_available():
    old_data = _make_run_data("old123", 0.60, {})
    old_data["metrics"].update({"game_grain_reliability": 0.0030, "game_grain_resolution": 0.0138})
    new_data = _make_run_data("new456", 0.66, {})
    new_data["metrics"].update({"game_grain_reliability": 0.0015, "game_grain_resolution": 0.0159})
    metric_deltas = compute_metric_deltas(old_data["metrics"], new_data["metrics"])
    importance_diff = compute_feature_importance_diff({}, {})

    html = render_comparison_html(old_data, new_data, metric_deltas, importance_diff)

    assert "real_improvement" in html


def test_render_comparison_html_omits_verdict_banner_when_unavailable():
    old_data = _make_run_data("old123", 0.60, {})
    new_data = _make_run_data("new456", 0.66, {})
    metric_deltas = compute_metric_deltas(old_data["metrics"], new_data["metrics"])
    importance_diff = compute_feature_importance_diff({}, {})

    html = render_comparison_html(old_data, new_data, metric_deltas, importance_diff)

    assert "real_improvement" not in html
    assert "overconfidence_risk" not in html


def test_render_comparison_html_is_self_contained_and_includes_run_ids():
    old_data = _make_run_data("old123", 0.60, {"a": 0.5, "b": 0.3})
    new_data = _make_run_data("new456", 0.66, {"a": 0.5, "c": 0.2})
    metric_deltas = compute_metric_deltas(old_data["metrics"], new_data["metrics"])
    importance_diff = compute_feature_importance_diff(old_data["importances"], new_data["importances"])

    html = render_comparison_html(old_data, new_data, metric_deltas, importance_diff)

    assert "old123" in html
    assert "new456" in html
    assert "tailwindcss" in html.lower()
    assert "chart.js" in html.lower() or "cdn.jsdelivr.net/npm/chart.js" in html
    # doctype/html/head/body must NOT be present if this were an Artifact page,
    # but this is a standalone local file — it needs its own full document shell.
    assert "<html" in html.lower()


def test_render_comparison_html_lists_added_and_removed_features():
    old_data = _make_run_data("old123", 0.60, {"a": 0.5, "b": 0.3})
    new_data = _make_run_data("new456", 0.66, {"a": 0.5, "c": 0.2})
    metric_deltas = compute_metric_deltas(old_data["metrics"], new_data["metrics"])
    importance_diff = compute_feature_importance_diff(old_data["importances"], new_data["importances"])

    html = render_comparison_html(old_data, new_data, metric_deltas, importance_diff)

    assert ">c<" in html  # added feature name rendered
    assert ">b<" in html  # removed feature name rendered


def test_render_comparison_html_includes_metric_row():
    old_data = _make_run_data("old123", 0.60, {})
    new_data = _make_run_data("new456", 0.66, {})
    metric_deltas = compute_metric_deltas(old_data["metrics"], new_data["metrics"])
    importance_diff = compute_feature_importance_diff({}, {})

    html = render_comparison_html(old_data, new_data, metric_deltas, importance_diff)

    assert "roc_auc" in html


def _log_fake_run(tmp_path, roc_auc, importances=None):
    """Seeds a real MLflow run with a metric and, if given, a feature-
    importance JSON artifact shaped exactly like save_feature_importance_json
    produces — the contract fetch_run_data's artifact-download path relies on."""
    mlflow.set_experiment("hit_predictor")
    with mlflow.start_run() as run:
        mlflow.log_metric("roc_auc", roc_auc)
        if importances is not None:
            path = tmp_path / "random_forest_feature_importance.json"
            path.write_text(json.dumps({"method": "native_gini", "values": importances}))
            mlflow.log_artifact(str(path))
        return run.info.run_id


def test_build_comparison_report_writes_html_file_from_real_runs(tmp_path):
    old_run_id = _log_fake_run(tmp_path, 0.60, {"a": 0.5, "b": 0.3})
    new_run_id = _log_fake_run(tmp_path, 0.66, {"a": 0.5, "c": 0.2})

    output_path = tmp_path / "reports" / "comparison.html"
    result_path = build_comparison_report(
        old_run_id, new_run_id, mlflow.get_tracking_uri(), output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    html = output_path.read_text()
    assert old_run_id in html
    assert new_run_id in html
    assert ">c<" in html


def test_build_comparison_report_handles_run_with_no_importance_artifact(tmp_path):
    """The naive-baseline run never logs a feature-importance JSON — must
    not crash, just treat that side's importances as empty."""
    old_run_id = _log_fake_run(tmp_path, 0.55, importances=None)
    new_run_id = _log_fake_run(tmp_path, 0.60, {"a": 0.5})

    output_path = tmp_path / "comparison.html"
    build_comparison_report(old_run_id, new_run_id, mlflow.get_tracking_uri(), output_path)

    html = output_path.read_text()
    assert ">a<" in html
