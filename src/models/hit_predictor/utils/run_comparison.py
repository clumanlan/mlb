"""
utils/run_comparison.py

Builds a static, self-contained HTML report comparing two MLflow runs of
this experiment: metric deltas, feature-importance shifts, and newly
added/removed features. Split the same way eval.py/mlflow_logging.py are:
compute_metric_deltas/compute_feature_importance_diff are pure and
MLflow-agnostic (fully unit-testable without a tracking store);
fetch_run_data is the only function that talks to MLflow;
render_comparison_html/build_comparison_report assemble the two into the
report.
"""

import json
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from models.hit_predictor.utils.eval import summarize_verdict


def compute_metric_deltas(old_metrics: dict, new_metrics: dict) -> list[dict]:
    """Row per metric present in either run: old, new, delta (new - old),
    pct_change (delta / old). A metric missing from one side gets None for
    that side and for delta/pct_change, rather than being dropped — a
    metric only the new run logs (e.g. a newly added eval) is exactly the
    kind of change this report exists to surface.

    Returns rows sorted by metric name for a stable, diffable report.
    """
    rows = []
    for metric in sorted(set(old_metrics) | set(new_metrics)):
        old_val = old_metrics.get(metric)
        new_val = new_metrics.get(metric)
        delta = None
        pct_change = None
        if old_val is not None and new_val is not None:
            delta = new_val - old_val
            pct_change = (delta / old_val) if old_val != 0 else None
        rows.append({
            "metric": metric, "old": old_val, "new": new_val,
            "delta": delta, "pct_change": pct_change,
        })
    return rows


def compute_feature_importance_diff(old: dict, new: dict, top_n: int = 20) -> dict:
    """added/removed feature names (set difference), plus the top_n
    features with the largest absolute importance shift among features
    present in both runs — sorted by |delta| descending so the biggest
    movers (in either direction) surface first regardless of sign.
    """
    old_features = set(old)
    new_features = set(new)
    added = sorted(new_features - old_features)
    removed = sorted(old_features - new_features)

    shared = old_features & new_features
    shifts = [
        {"feature": f, "old": old[f], "new": new[f], "delta": new[f] - old[f]}
        for f in shared
    ]
    shifts.sort(key=lambda s: abs(s["delta"]), reverse=True)

    return {"added": added, "removed": removed, "top_shifts": shifts[:top_n]}


def fetch_run_data(run_id: str, tracking_uri: str) -> dict:
    """The one MLflow-aware function in this module. Returns metrics,
    params, tags, and (if the run logged one) the parsed feature-importance
    JSON artifact's "values" dict — {} if that artifact isn't present
    (e.g. the naive-baseline run, which never computes feature importance).
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)

    importances = {}
    artifact_names = {f.path for f in client.list_artifacts(run_id)}
    if "random_forest_feature_importance.json" in artifact_names:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="random_forest_feature_importance.json",
                dst_path=tmp_dir,
            )
            importances = json.loads(Path(local_path).read_text())["values"]

    return {
        "run_id": run_id,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
        "start_time": run.info.start_time,
        "importances": importances,
    }


def compute_verdict_for_report(old_metrics: dict, new_metrics: dict) -> dict | None:
    """Runs summarize_verdict on a run-comparison pair, preferring game-grain
    reliability/resolution (game_grain_reliability/game_grain_resolution, as
    logged by log_evaluation_to_mlflow after run_pa_vs_game_grain_check) over
    PA-grain ones, since game grain is the primary evaluation target per
    BENCHMARKS.md §2. Falls back to plain reliability/resolution when the
    game-grain keys aren't present on both runs (e.g. comparing against an
    old run from before game-grain evaluation existed).

    Returns None if neither key pair is available on both runs — the report
    should just omit the verdict banner rather than error.
    """
    for reliability_key, resolution_key in (
        ("game_grain_reliability", "game_grain_resolution"),
        ("reliability", "resolution"),
    ):
        if (
            reliability_key in old_metrics and resolution_key in old_metrics
            and reliability_key in new_metrics and resolution_key in new_metrics
        ):
            return summarize_verdict(
                {"reliability": old_metrics[reliability_key], "resolution": old_metrics[resolution_key]},
                {"reliability": new_metrics[reliability_key], "resolution": new_metrics[resolution_key]},
            )
    return None


# Metrics where a LOWER value is the improvement — everything else defaults to
# higher-is-better. Substring match (not exact) so game_grain_/calib_eval_*_
# variants of the same base metric are covered without listing every prefix.
_LOWER_IS_BETTER_SUBSTRINGS = ("loss", "brier", "ece", "reliability")


def _is_lower_better(metric_name: str) -> bool:
    return any(s in metric_name for s in _LOWER_IS_BETTER_SUBSTRINGS)


def _delta_color_class(metric_name: str, delta) -> str:
    if delta is None:
        return "text-slate-400"
    improved = (delta < 0) if _is_lower_better(metric_name) else (delta > 0)
    if delta == 0:
        return "text-slate-400"
    return "text-emerald-600" if improved else "text-rose-600"


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_comparison_html(old_data: dict, new_data: dict, metric_deltas: list[dict], importance_diff: dict) -> str:
    """Render a static, self-contained HTML report — Tailwind CDN for
    styling, Chart.js CDN for the feature-importance-shift bar chart. Meant
    to be opened directly as a local file (not published as an Artifact),
    so it owns its full <html>/<head>/<body> document shell.
    """
    metric_rows = "\n".join(
        f'<tr class="border-b border-slate-200">'
        f'<td class="py-2 pr-4 font-mono text-sm text-slate-700">{row["metric"]}</td>'
        f'<td class="py-2 pr-4 text-right tabular-nums text-slate-500">{_fmt(row["old"])}</td>'
        f'<td class="py-2 pr-4 text-right tabular-nums text-slate-500">{_fmt(row["new"])}</td>'
        f'<td class="py-2 pr-4 text-right tabular-nums font-semibold {_delta_color_class(row["metric"], row["delta"])}">'
        f'{_fmt(row["delta"])}</td>'
        f'</tr>'
        for row in metric_deltas
    )

    def _feature_badges(names, color_classes):
        if not names:
            return '<span class="text-slate-400 text-sm">none</span>'
        return " ".join(
            f'<span class="inline-block px-2 py-0.5 mr-1 mb-1 rounded {color_classes} text-xs font-mono">{n}</span>'
            for n in names
        )

    added_html = _feature_badges(importance_diff["added"], "bg-emerald-100 text-emerald-800")
    removed_html = _feature_badges(importance_diff["removed"], "bg-rose-100 text-rose-800")

    shift_rows = "\n".join(
        f'<tr class="border-b border-slate-200">'
        f'<td class="py-2 pr-4 font-mono text-sm text-slate-700">{s["feature"]}</td>'
        f'<td class="py-2 pr-4 text-right tabular-nums text-slate-500">{_fmt(s["old"])}</td>'
        f'<td class="py-2 pr-4 text-right tabular-nums text-slate-500">{_fmt(s["new"])}</td>'
        f'<td class="py-2 pr-4 text-right tabular-nums font-semibold {"text-emerald-600" if s["delta"] > 0 else "text-rose-600"}">'
        f'{_fmt(s["delta"])}</td>'
        f'</tr>'
        for s in importance_diff["top_shifts"]
    )

    chart_labels = json.dumps([s["feature"] for s in importance_diff["top_shifts"]])
    chart_deltas = json.dumps([s["delta"] for s in importance_diff["top_shifts"]])

    # Verdict banner — per BENCHMARKS.md §2, "is this a real improvement" is
    # answered by reliability + resolution together, not by eyeballing the
    # metric-delta table below. Omitted entirely when neither run has grain
    # (game or PA) reliability/resolution logged.
    verdict = compute_verdict_for_report(old_data["metrics"], new_data["metrics"])
    verdict_labels = {
        "real_improvement": ("Real improvement", "bg-emerald-100 text-emerald-800 border-emerald-300"),
        "overconfidence_risk": ("Overconfidence risk", "bg-rose-100 text-rose-800 border-rose-300"),
        "calibration_only": ("Calibration only", "bg-amber-100 text-amber-800 border-amber-300"),
        "no_improvement": ("No improvement", "bg-slate-200 text-slate-700 border-slate-300"),
    }
    verdict_html = ""
    if verdict is not None:
        label, classes = verdict_labels[verdict["verdict"]]
        verdict_html = f"""
  <div class="mb-8 px-4 py-3 rounded border {classes} text-sm">
    <span class="font-mono font-semibold">{verdict["verdict"]}</span> — {label}.
    reliability &Delta; {_fmt(verdict["reliability_delta"])} (trustworthy: {verdict["trustworthy"]}),
    resolution &Delta; {_fmt(verdict["resolution_delta"])} (differentiated: {verdict["differentiated"]}).
  </div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Run comparison: {old_data["run_id"][:8]} vs {new_data["run_id"][:8]}</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-50 text-slate-900 font-sans">
<div class="max-w-4xl mx-auto px-6 py-10">

  <h1 class="text-2xl font-bold mb-1">Run comparison</h1>
  <p class="text-sm text-slate-500 mb-8 font-mono">
    old: {old_data["run_id"]} &nbsp;&rarr;&nbsp; new: {new_data["run_id"]}
  </p>
{verdict_html}
  <h2 class="text-lg font-semibold mb-3">Metric deltas</h2>
  <table class="w-full mb-10 text-sm">
    <thead>
      <tr class="border-b-2 border-slate-300 text-left text-xs uppercase tracking-wide text-slate-400">
        <th class="py-2 pr-4">Metric</th>
        <th class="py-2 pr-4 text-right">Old</th>
        <th class="py-2 pr-4 text-right">New</th>
        <th class="py-2 pr-4 text-right">Delta</th>
      </tr>
    </thead>
    <tbody>
      {metric_rows}
    </tbody>
  </table>

  <h2 class="text-lg font-semibold mb-3">Feature importance shifts (top movers)</h2>
  <div class="mb-6 bg-white rounded-lg border border-slate-200 p-4">
    <canvas id="shiftChart" height="120"></canvas>
  </div>
  <table class="w-full mb-10 text-sm">
    <thead>
      <tr class="border-b-2 border-slate-300 text-left text-xs uppercase tracking-wide text-slate-400">
        <th class="py-2 pr-4">Feature</th>
        <th class="py-2 pr-4 text-right">Old</th>
        <th class="py-2 pr-4 text-right">New</th>
        <th class="py-2 pr-4 text-right">Delta</th>
      </tr>
    </thead>
    <tbody>
      {shift_rows}
    </tbody>
  </table>

  <h2 class="text-lg font-semibold mb-2">Added features ({len(importance_diff["added"])})</h2>
  <div class="mb-6">{added_html}</div>

  <h2 class="text-lg font-semibold mb-2">Removed features ({len(importance_diff["removed"])})</h2>
  <div class="mb-6">{removed_html}</div>

</div>
<script>
  new Chart(document.getElementById('shiftChart'), {{
    type: 'bar',
    data: {{
      labels: {chart_labels},
      datasets: [{{
        label: 'Importance delta (new - old)',
        data: {chart_deltas},
        backgroundColor: {chart_deltas}.map(d => d >= 0 ? '#059669' : '#e11d48'),
      }}],
    }},
    options: {{
      indexAxis: 'y',
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ x: {{ beginAtZero: true }} }},
    }},
  }});
</script>
</body>
</html>
"""


def build_comparison_report(old_run_id: str, new_run_id: str, tracking_uri: str, output_path) -> Path:
    """Orchestrates fetch -> compute -> render -> write. Returns the
    written Path so callers (e.g. a script that then shells out to `open`)
    have it without re-deriving the path.
    """
    old_data = fetch_run_data(old_run_id, tracking_uri)
    new_data = fetch_run_data(new_run_id, tracking_uri)

    metric_deltas = compute_metric_deltas(old_data["metrics"], new_data["metrics"])
    importance_diff = compute_feature_importance_diff(old_data["importances"], new_data["importances"])

    html = render_comparison_html(old_data, new_data, metric_deltas, importance_diff)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
