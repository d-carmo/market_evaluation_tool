"""Signal details table component."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_signal_table(signal_details: dict) -> None:
    """Render a coloured signal details table."""
    if not signal_details:
        st.caption("No signal details available.")
        return

    rows = []
    for name, info in signal_details.items():
        value = info.get("value", 0)
        raw = info.get("raw", 0)
        label = {1: "BUY", 0: "HOLD", -1: "SELL"}.get(int(value), str(value))
        rows.append({"Signal": name, "Direction": label, "Raw Value": round(float(raw), 4)})

    df = pd.DataFrame(rows)

    def colour_direction(val: str) -> str:
        if val == "BUY":
            return "color: #00c853; font-weight: bold"
        if val == "SELL":
            return "color: #d50000; font-weight: bold"
        return "color: #ffa000; font-weight: bold"

    styled = df.style.map(colour_direction, subset=["Direction"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
