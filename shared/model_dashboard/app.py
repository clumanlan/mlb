import pandas as pd
import streamlit as st

from shared.model_dashboard.components.description_block import render_description
from shared.model_dashboard.components.header import render_header
from shared.model_dashboard.components.tables import render_contribution_table, render_top_losses_table
from shared.model_dashboard.logic.binning import bin_feature_by_outcome
from shared.model_dashboard.logic.bootstrap import bootstrap_ci
from shared.model_dashboard.logic.calibration import reliability_bins
from shared.model_dashboard.logic.contribution import compute_contribution
from shared.model_dashboard.logic.metrics import per_row_loss
from shared.model_dashboard.logic.missing import missing_value_report
from shared.model_dashboard.logic.slicing import generate_interaction_slices, generate_single_slices
from shared.model_dashboard.plots.distribution import build_distribution_figure
from shared.model_dashboard.plots.feature_vs_outcome import build_feature_vs_outcome_figure
from shared.model_dashboard.plots.forest import build_forest_figure
from shared.model_dashboard.plots.missing import build_missing_figure
from shared.model_dashboard.plots.reliability import build_reliability_figure
from shared.model_dashboard.plots.time_series import build_time_series_figure


def _get_slice_df(df: pd.DataFrame, feature: str, value) -> pd.DataFrame:
    if "×" in str(feature):
        cols = str(feature).split("×")
        vals = str(value).split("×")
        mask = pd.Series(True, index=df.index)
        for col, val in zip(cols, vals):
            mask &= df[col].astype(str) == val
        return df[mask]
    return df[df[feature].astype(str) == str(value)]


@st.cache_data
def _load(data_path: str, target_col: str, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["_loss"] = df.apply(lambda r: per_row_loss(r[target_col], r[pred_col]), axis=1)
    return df


@st.cache_data
def _compute_contributions(
    data_path: str, target_col: str, pred_col: str,
    slice_cols: tuple, interaction_pairs: tuple,
) -> pd.DataFrame:
    df = _load(data_path, target_col, pred_col)
    slices = (
        generate_single_slices(df, list(slice_cols))
        + generate_interaction_slices(df, list(interaction_pairs))
    )
    return compute_contribution(df, slices, target_col, pred_col)


@st.cache_data
def _compute_slice_cis(
    data_path: str, target_col: str, pred_col: str,
    slice_cols: tuple, interaction_pairs: tuple,
) -> pd.DataFrame:
    df = _load(data_path, target_col, pred_col)
    contrib = _compute_contributions(data_path, target_col, pred_col, slice_cols, interaction_pairs)
    rows = []
    for _, row in contrib.head(15).iterrows():
        subset = _get_slice_df(df, row["feature"], row["value"])
        if len(subset) < 10:
            continue
        point, lo, hi = bootstrap_ci(subset, target_col, pred_col)
        rows.append({"feature": row["feature"], "value": row["value"],
                     "point": point, "lo": lo, "hi": hi})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["feature", "value", "point", "lo", "hi"])


def run_dashboard(config: dict) -> None:
    st.set_page_config(layout="wide", page_title=config.get("model_name", "Model Dashboard"))

    data_path = config["data_path"]
    target_col = config["target"]
    pred_col = config["pred"]
    entity_col = config.get("entity", "entity")
    date_col = config.get("date", "game_date")
    slice_cols = tuple(config.get("slice_cols", []))
    interaction_pairs = tuple(tuple(p) for p in config.get("interaction_pairs", []))

    df = _load(data_path, target_col, pred_col)
    contrib_df = _compute_contributions(data_path, target_col, pred_col, slice_cols, interaction_pairs)

    overall_loss = float(df["_loss"].mean())
    top_slice = (
        f"{contrib_df.iloc[0]['feature']}={contrib_df.iloc[0]['value']}"
        if len(contrib_df) > 0 else "—"
    )
    summary_stats = {
        "total_pa": len(df),
        "positive_rate": float(df[target_col].mean()),
        "overall_loss": overall_loss,
        "top_slice": top_slice,
    }

    render_header(config, summary_stats)

    eda_tab, error_tab, cal_tab = st.tabs(["EDA", "Error analysis", "Calibration"])

    feature_cols = [c for c in df.columns if c not in (target_col, pred_col, "_loss")]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    # ── EDA ──────────────────────────────────────────────────────────────────
    with eda_tab:
        st.subheader("Feature distribution")
        render_description(
            what="Distribution of any column in the dataset.",
            why="Spot skew, outliers, and unexpected value ranges before reading model results.",
            read="Numeric → histogram. Categorical → bar. Narrow or skewed distributions may need transformation.",
        )
        dist_col = st.selectbox("Column", feature_cols, key="dist_col")
        st.plotly_chart(build_distribution_figure(df, dist_col), use_container_width=True)

        st.divider()

        st.subheader("Missing values")
        render_description(
            what="Percentage of null values per feature.",
            why="High missingness causes silent imputation bias that can corrupt feature signal.",
            read="Features above 20% warrant investigation. If missingness correlates with the target, the imputation will skew predictions.",
        )
        st.plotly_chart(build_missing_figure(missing_value_report(df[feature_cols])), use_container_width=True)

        st.divider()

        st.subheader("Feature vs outcome")
        render_description(
            what="Binned mean outcome rate for a selected feature.",
            why="Shows whether a feature has a monotone or non-linear relationship with the target.",
            read="Green bars exceed the overall hit rate; red bars fall below. A clear gradient signals predictive value.",
        )
        fvo_col = st.selectbox("Column", feature_cols, key="fvo_col")
        binned = bin_feature_by_outcome(df, fvo_col, target_col)
        st.plotly_chart(
            build_feature_vs_outcome_figure(binned, float(df[target_col].mean())),
            use_container_width=True,
        )

        st.divider()

        st.subheader("Entity time series")
        render_description(
            what="Selected metric over time for one player.",
            why="Reveals individual trends and catches model drift on specific players.",
            read="Blue dots = no hit, red dots = hit. A flat pred line with wrong-colored dots signals miscalibration for this player.",
        )
        ts_c1, ts_c2, ts_c3, ts_c4 = st.columns(4)
        entities = sorted(df[entity_col].dropna().unique()) if entity_col in df.columns else []
        ts_entity = ts_c1.selectbox("Player", entities, key="ts_entity") if entities else None
        ts_metric = ts_c2.selectbox("Metric", [pred_col] + num_cols, key="ts_metric")
        ts_slice = ts_c3.selectbox("Slice by", ["None"] + list(slice_cols), key="ts_slice")
        seasons = (
            sorted(df["game_season"].dropna().unique(), reverse=True)
            if "game_season" in df.columns else []
        )
        ts_season = ts_c4.selectbox("Season", ["All"] + [str(int(s)) for s in seasons], key="ts_season")

        if ts_entity:
            plot_df = df.copy()
            if ts_season != "All" and "game_season" in plot_df.columns:
                plot_df = plot_df[plot_df["game_season"].astype(str).str.startswith(ts_season.split(".")[0])]
            st.plotly_chart(
                build_time_series_figure(
                    plot_df, ts_entity, ts_metric,
                    date_col=date_col, entity_col=entity_col, outcome_col=target_col,
                    slice_by=(None if ts_slice == "None" else ts_slice),
                ),
                use_container_width=True,
            )

    # ── Error analysis ────────────────────────────────────────────────────────
    with error_tab:
        st.subheader("Step 1 — Top losses (Howard)")
        render_description(
            what="The 20 rows where the model was most wrong.",
            why="Howard's method: start with the worst examples and look for shared patterns — they reveal systematic failures.",
            read="High log loss = confident and wrong. Scan batSide, pitcher_hand, and weather for patterns across the top rows.",
        )
        confonly = st.checkbox(
            "High-confidence wrong only  (pred > 0.6 on no-hit, or pred < 0.4 on hit)"
        )
        loss_df = df.copy()
        if confonly:
            loss_df = loss_df[
                ((loss_df[pred_col] > 0.6) & (loss_df[target_col] == 0))
                | ((loss_df[pred_col] < 0.4) & (loss_df[target_col] == 1))
            ]
        top_loss_df = loss_df.sort_values("_loss", ascending=False).head(20)
        render_top_losses_table(top_loss_df)

        st.divider()

        st.subheader("Step 2 — Slice contribution (Ng)")
        render_description(
            what="Each slice's proportional contribution to total log loss.",
            why="Ng's method: prioritise by contribution, not raw loss — a large mildly-bad slice outranks a tiny very-bad one.",
            math="contribution = pct × (slice_loss − overall_loss)",
            read="Top row is the highest-priority slice to fix. Negative contribution means the model does better on that slice than average.",
        )
        render_contribution_table(contrib_df)

        st.divider()

        st.subheader("Step 3 — CI check (Shankar)")
        render_description(
            what="Bootstrap 95% CIs for the top slices from Step 2.",
            why="Shankar's method: confirm the gap is real before investing in a fix — don't chase noise.",
            math="95% CI (bootstrap, assumes within-season exchangeability)",
            read="If a slice CI doesn't cross the overall loss line, the gap is statistically real. Overlapping CIs suggest the difference may be noise.",
        )
        ci_df = _compute_slice_cis(data_path, target_col, pred_col, slice_cols, interaction_pairs)
        if len(ci_df) > 0:
            st.plotly_chart(build_forest_figure(ci_df, overall_loss), use_container_width=True)
        else:
            st.info("No slices with enough data for bootstrap CI (min n=10 per slice).")

    # ── Calibration ───────────────────────────────────────────────────────────
    with cal_tab:
        st.subheader("Reliability diagram")
        render_description(
            what="Actual hit rate vs mean predicted probability across bins.",
            why="For betting, calibration matters as much as discrimination — an overconfident model loses money even when directionally right.",
            read="Points on the diagonal = perfect calibration. Above = model underestimates (bet more aggressively). Below = overestimates (fade). The bar chart shows PA count per bin.",
        )
        rel_df = reliability_bins(df, target_col, pred_col)
        st.plotly_chart(build_reliability_figure(rel_df), use_container_width=True)
