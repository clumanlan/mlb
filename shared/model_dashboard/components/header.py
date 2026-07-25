import streamlit as st


def render_header(config: dict, summary_stats: dict) -> None:
    st.markdown(f"## {config.get('model_name', 'Model Dashboard')}")

    target = config.get("target", "?")
    entity = config.get("entity", "?")
    pill_style = (
        "display:inline-block;background:#e8f0fe;padding:2px 10px;"
        "border-radius:12px;font-size:0.83em;margin-right:6px"
    )
    st.markdown(
        f"<span style='{pill_style}'>target: <b>{target}</b></span>"
        f"<span style='{pill_style}'>entity: <b>{entity}</b></span>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total PAs", f"{summary_stats['total_pa']:,}")
    c2.metric("Positive rate", f"{summary_stats['positive_rate']:.3f}")
    c3.metric("Overall log loss", f"{summary_stats['overall_loss']:.4f}")
    c4.metric("Top slice to fix", summary_stats["top_slice"])

    st.divider()
