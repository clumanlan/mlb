import pandas as pd
import plotly.graph_objects as go


def build_forest_figure(slice_ci_df: pd.DataFrame, overall_loss: float) -> go.Figure:
    """
    slice_ci_df columns: feature, value, point, lo, hi
    Rows displayed top-to-bottom in the order given (caller sorts by contribution desc).
    """
    df = slice_ci_df.head(15).copy()
    labels = (df["feature"] + " = " + df["value"].astype(str)).tolist()

    fig = go.Figure()

    # CI lines
    for i, row in df.iterrows():
        label = f"{row['feature']} = {row['value']}"
        fig.add_trace(go.Scatter(
            x=[row["lo"], row["point"], row["hi"]],
            y=[label, label, label],
            mode="lines+markers",
            marker=dict(
                symbol=["circle-open", "circle", "circle-open"],
                size=[8, 10, 8],
                color=["#4C78A8", "#E45756", "#4C78A8"],
            ),
            line=dict(color="#4C78A8", width=2),
            hovertemplate=(
                f"{label}<br>"
                "lo: %{x[0]:.4f}<br>"
                "point: %{x[1]:.4f}<br>"
                "hi: %{x[2]:.4f}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Vertical reference line at overall loss
    fig.add_vline(
        x=overall_loss,
        line_dash="dash",
        line_color="black",
        annotation_text=f"overall {overall_loss:.4f}",
        annotation_position="top",
    )

    fig.update_layout(
        xaxis_title="log loss (95% CI, bootstrap)",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(labels))),
        margin=dict(l=20, r=20, t=30, b=40),
        height=max(300, 40 * len(df)),
    )
    return fig
