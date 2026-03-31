"""Candlestick chart component using Plotly."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_candlestick(chart_data: dict) -> None:
    """Render OHLCV candlestick with RSI sub-panel."""
    points = chart_data.get("points", [])
    if not points:
        st.caption("No chart data available.")
        return

    dates = [p["date"] for p in points]
    opens = [p["open"] for p in points]
    highs = [p["high"] for p in points]
    lows = [p["low"] for p in points]
    closes = [p["close"] for p in points]
    rsi = [p.get("rsi_14") for p in points]
    bb_upper = [p.get("bb_upper") for p in points]
    bb_lower = [p.get("bb_lower") for p in points]

    has_rsi = any(v is not None for v in rsi)
    rows = 2 if has_rsi else 1
    row_heights = [0.75, 0.25] if has_rsi else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name="OHLCV", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if any(v is not None for v in bb_upper):
        fig.add_trace(go.Scatter(x=dates, y=bb_upper, name="BB Upper",
                                  line=dict(color="rgba(100,100,255,0.4)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bb_lower, name="BB Lower",
                                  line=dict(color="rgba(100,100,255,0.4)", dash="dot"),
                                  fill="tonexty", fillcolor="rgba(100,100,255,0.05)"), row=1, col=1)

    if has_rsi:
        fig.add_trace(go.Scatter(x=dates, y=rsi, name="RSI 14",
                                  line=dict(color="#ab47bc", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    ticker = chart_data.get("ticker", "")
    fig.update_layout(
        title=f"{ticker} — Price Chart",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)
