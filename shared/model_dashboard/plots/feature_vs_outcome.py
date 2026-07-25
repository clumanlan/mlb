import pandas as pd
import plotly.graph_objects as go


def build_feature_vs_outcome_figure(binned_df: pd.DataFrame, overall_rate: float) -> go.Figure:
    colors = [
        "#2ca02c" if v > overall_rate else "#d62728"
        for v in binned_df["mean_outcome"]
    ]

    fig = go.Figure(go.Bar(
        x=binned_df["bin_label"].astype(str),
        y=binned_df["mean_outcome"],
        marker_color=colors,
        text=binned_df["n"],
        textposition="outside",
        hovertemplate="%{x}<br>hit rate: %{y:.3f}<br>n: %{text}<extra></extra>",
    ))

    fig.add_hline(
        y=overall_rate,
        line_dash="dash",
        line_color="black",
        annotation_text=f"overall {overall_rate:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        yaxis_title="hit rate",
        margin=dict(l=40, r=20, t=30, b=60),
        yaxis_range=[0, min(1.0, binned_df["mean_outcome"].max() * 1.25 + 0.05)],
    )
    return fig
