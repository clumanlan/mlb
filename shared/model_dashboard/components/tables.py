import pandas as pd
import streamlit as st


def render_top_losses_table(df: pd.DataFrame) -> None:
    display = df.copy()
    for col in display.select_dtypes(include="float").columns:
        display[col] = display[col].round(4)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "_loss": st.column_config.NumberColumn("log_loss", format="%.4f"),
        },
    )


def render_contribution_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No slices passed the min_n threshold.")
        return

    display = df.copy()
    for col in ["pct", "slice_loss", "delta", "contribution"]:
        if col in display.columns:
            display[col] = display[col].round(4)

    st.caption("Sorted by contribution (desc) — top row is highest priority.")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "pct":          st.column_config.NumberColumn("pct", format="%.3f"),
            "slice_loss":   st.column_config.NumberColumn("slice_loss", format="%.4f"),
            "delta":        st.column_config.NumberColumn("delta", format="%.4f"),
            "contribution": st.column_config.NumberColumn("contribution", format="%.4f"),
        },
    )
