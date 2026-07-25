import pandas as pd
import plotly.graph_objects as go


_OUTCOME_COLORS = {0: "#4C78A8", 1: "#E45756"}


def build_time_series_figure(
    df: pd.DataFrame,
    entity: str,
    metric: str,
    date_col: str = "game_date",
    entity_col: str = "batter_name",
    outcome_col: str = "is_hit",
    slice_by: str | None = None,
) -> go.Figure:
    player_df = df[df[entity_col] == entity].sort_values(date_col)
    fig = go.Figure()

    groups = player_df[slice_by].unique() if slice_by else [None]

    for group in groups:
        subset = player_df[player_df[slice_by] == group] if slice_by else player_df
        name = str(group) if slice_by else entity

        # Line
        fig.add_trace(go.Scatter(
            x=subset[date_col],
            y=subset[metric],
            mode="lines",
            name=name,
            line=dict(width=1.5),
            hovertemplate=f"{name}<br>%{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>",
        ))

        # Dots colored by outcome
        if outcome_col in subset.columns:
            for outcome_val, color in _OUTCOME_COLORS.items():
                mask = subset[outcome_col] == outcome_val
                fig.add_trace(go.Scatter(
                    x=subset.loc[mask, date_col],
                    y=subset.loc[mask, metric],
                    mode="markers",
                    marker=dict(color=color, size=5),
                    name=f"{name} ({'hit' if outcome_val else 'no hit'})",
                    showlegend=False,
                    hovertemplate=f"{'hit' if outcome_val else 'no hit'}<br>%{{x}}<br>{metric}: %{{y:.3f}}<extra></extra>",
                ))

    fig.update_layout(
        xaxis_title=date_col,
        yaxis_title=metric,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
