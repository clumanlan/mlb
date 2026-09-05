"""
Single-pitcher drill-down for k_predictor v6, TEST_SEASON=2025.

Pick a pitcher -> season line chart (pitches thrown / strikeouts / batters
faced per start) + a PA-level table of every batter he faced with v6's
predicted strikeout probability and the pre-game stats behind it.

Deliberately NOT built on shared/model_dashboard -- that package is for the
3-tab EDA/error-analysis/calibration workflow across a whole model. This is
a narrower, single-pitcher view; sharing infra here would add indirection
for one page, not remove duplication (same judgment call as this project's
backtest/experiment scripts being copy-adapted rather than shared).

Run from repo root with:
    streamlit run pitcher_pa_model/pitcher_view.py
Requires pitcher_pa_model/data.csv and game_log.csv (run build_data.py and
build_game_log.py first).
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).parent

st.set_page_config(layout="wide", page_title="k_predictor v6 — pitcher view")


@st.cache_data
def load_data():
    pa_df = pd.read_csv(_HERE / "data.csv", parse_dates=["game_date"])
    game_log = pd.read_csv(_HERE / "game_log.csv", parse_dates=["game_date"])
    return pa_df, game_log


pa_df, game_log = load_data()

st.markdown("## k_predictor v6 — pitcher view")
st.caption("TEST_SEASON=2025 (held out, never fit on) · one row per plate appearance faced")

pitchers = sorted(pa_df["pitcher_name"].dropna().unique())
pitcher = st.selectbox("Pitcher", pitchers)

pitcher_games = game_log[game_log["pitcher_name"] == pitcher].sort_values("game_date")
pitcher_pas = pa_df[pa_df["pitcher_name"] == pitcher].sort_values(["game_date", "batting_order"])

if pitcher_games.empty:
    st.info("No starts found for this pitcher in TEST_SEASON.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Starts", len(pitcher_games))
c2.metric("Total batters faced", int(pitcher_games["batters_faced"].sum()))
c3.metric("Total strikeouts", int(pitcher_games["strikeouts"].sum()))
c4.metric("K rate (actual)", f"{pitcher_pas['is_strikeout'].mean():.3f}")

st.divider()

st.subheader("Season log")
st.caption("Pitches thrown (right axis) vs. strikeouts and batters faced (left axis), by start.")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pitcher_games["game_date"], y=pitcher_games["batters_faced"],
    name="Batters faced", mode="lines+markers", line=dict(color="#4C78A8"),
))
fig.add_trace(go.Scatter(
    x=pitcher_games["game_date"], y=pitcher_games["strikeouts"],
    name="Strikeouts", mode="lines+markers", line=dict(color="#E45756"),
))
fig.add_trace(go.Scatter(
    x=pitcher_games["game_date"], y=pitcher_games["pitches_thrown"],
    name="Pitches thrown", mode="lines+markers", line=dict(color="#72B7B2", dash="dot"),
    yaxis="y2",
))
fig.update_layout(
    xaxis_title="Game date",
    yaxis=dict(title="Batters faced / Strikeouts"),
    yaxis2=dict(title="Pitches thrown", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    margin=dict(t=30),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Plate appearances")
st.caption(
    "Every batter faced this season, with v6's predicted strikeout probability and the "
    "pre-game stats it saw. actual = 1 means the PA ended in a strikeout."
)

display_cols = [
    "game_date", "batter_name", "platoon_matchup", "pred_prob", "is_strikeout",
    "pitcher_roll_last3g_k_rate", "pitcher_roll_season_k_rate",
    "batter_roll_season_pa_strikeout_rate", "batter_last_season_pa_strikeout_rate",
    "opp_team_roll_season_pa_strikeout_rate", "weather_condition", "expected_times_through_order",
]
display_cols = [c for c in display_cols if c in pitcher_pas.columns]

sort_choice = st.radio(
    "Sort by", ["Date", "Predicted probability (high to low)", "Predicted probability (low to high)"],
    horizontal=True,
)
table_df = pitcher_pas[display_cols].copy()
if sort_choice == "Date":
    table_df = table_df.sort_values("game_date")
elif sort_choice == "Predicted probability (high to low)":
    table_df = table_df.sort_values("pred_prob", ascending=False)
else:
    table_df = table_df.sort_values("pred_prob", ascending=True)

for col in table_df.select_dtypes(include="float").columns:
    table_df[col] = table_df[col].round(4)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "pred_prob": st.column_config.NumberColumn("pred_prob", format="%.3f"),
        "is_strikeout": st.column_config.NumberColumn("actual"),
    },
)
