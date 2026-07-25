import streamlit as st


def render_description(
    what: str,
    why: str,
    math: str | None = None,
    read: str | None = None,
) -> None:
    rows = [("what", what), ("why", why)]
    if math:
        rows.append(("math", f"<code style='font-size:0.9em'>{math}</code>"))
    if read:
        rows.append(("read", read))

    cells = "".join(
        f"<tr>"
        f"<td style='width:52px;font-weight:600;color:#aaa;padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap'>{lbl}</td>"
        f"<td style='color:#555;padding:3px 0;line-height:1.4'>{val}</td>"
        f"</tr>"
        for lbl, val in rows
    )
    st.markdown(
        f"<table style='border:none;border-collapse:collapse;font-size:0.83em;margin-bottom:10px'>"
        f"{cells}</table>",
        unsafe_allow_html=True,
    )
