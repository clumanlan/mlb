import pandas as pd
import plotly.graph_objects as go


def build_distribution_figure(df: pd.DataFrame, column: str) -> go.Figure:
    col = df[column].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(col)

    if is_numeric:
        fig = go.Figure(go.Histogram(
            x=col,
            marker_color="#4C78A8",
            hovertemplate="%{x}<br>count: %{y}<extra></extra>",
        ))
    else:
        counts = col.value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            marker_color="#4C78A8",
            hovertemplate="%{x}<br>count: %{y}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="count",
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig
