import pandas as pd
import plotly.graph_objects as go


def build_missing_figure(missing_df: pd.DataFrame) -> go.Figure:
    df = missing_df[missing_df["pct_missing"] > 0].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(margin=dict(l=40, r=20, t=20, b=40))
        return fig

    fig = go.Figure(go.Bar(
        x=df["pct_missing"],
        y=df["feature"],
        orientation="h",
        marker_color="#4C78A8",
        hovertemplate="%{y}<br>missing: %{x:.1%}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(title="% missing", tickformat=".0%", range=[0, 1]),
        margin=dict(l=40, r=20, t=20, b=40),
        height=max(200, 30 * len(df)),
    )
    return fig
