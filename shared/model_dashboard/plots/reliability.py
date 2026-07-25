import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_reliability_figure(reliability_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Top: scatter of actual_rate vs pred_mean
    fig.add_trace(go.Scatter(
        x=reliability_df["pred_mean"],
        y=reliability_df["actual_rate"],
        mode="markers+lines",
        marker=dict(size=8, color="#4C78A8"),
        line=dict(color="#4C78A8"),
        hovertemplate="pred: %{x:.3f}<br>actual: %{y:.3f}<extra></extra>",
        name="reliability",
    ), row=1, col=1)

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="perfect calibration",
        hoverinfo="skip",
    ), row=1, col=1)

    # Bottom: bar of n (confidence histogram)
    fig.add_trace(go.Bar(
        x=reliability_df["pred_mean"],
        y=reliability_df["n"],
        marker_color="#4C78A8",
        opacity=0.6,
        hovertemplate="pred: %{x:.3f}<br>n: %{y}<extra></extra>",
        name="count",
    ), row=2, col=1)

    fig.update_xaxes(title_text="mean predicted probability", row=2, col=1)
    fig.update_yaxes(title_text="actual hit rate", row=1, col=1)
    fig.update_yaxes(title_text="count", row=2, col=1)
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    return fig
