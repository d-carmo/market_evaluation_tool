"""Forecast range chart component."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_forecast_chart(
    prediction: dict,
    ticker: str,
    observed: dict[int, float] | None = None,
) -> None:
    """Bar + range chart for short/long forecasts.

    ``observed`` maps horizon_days → actual price (populated once the
    prediction horizon has passed and accuracy has been evaluated).
    """
    sf = prediction.get("short_forecast", {})
    lf = prediction.get("long_forecast", {})
    current = prediction.get("current_price", 0)

    short_h = sf.get("horizon_days")
    long_h  = lf.get("horizon_days")

    labels = [
        "Now",
        f"{short_h}d Forecast",
        f"{long_h}d Forecast",
    ]
    mids  = [current, sf.get("mid", current), lf.get("mid", current)]
    lows  = [None, sf.get("low"), lf.get("low")]
    highs = [None, sf.get("high"), lf.get("high")]

    fig = go.Figure()

    # Mid prices (forecast bars)
    fig.add_trace(go.Bar(
        x=labels, y=mids,
        name="Forecast Mid",
        marker_color=["#78909c", "#26a69a", "#42a5f5"],
        text=[f"${v:.2f}" for v in mids],
        textposition="outside",
    ))

    # Error bars for forecasts
    for i in (1, 2):
        if lows[i] is not None and highs[i] is not None:
            fig.add_trace(go.Scatter(
                x=[labels[i], labels[i]],
                y=[lows[i], highs[i]],
                mode="lines",
                line=dict(color="rgba(255,165,0,0.6)", width=6),
                name=f"{labels[i]} Range",
                showlegend=False,
            ))

    # Observed (actual) prices — shown as markers when available
    if observed:
        obs_x, obs_y = [], []
        for h, label in ((short_h, labels[1]), (long_h, labels[2])):
            if h in observed:
                obs_x.append(label)
                obs_y.append(observed[h])

        if obs_x:
            fig.add_trace(go.Scatter(
                x=obs_x,
                y=obs_y,
                mode="markers+text",
                name="Observed",
                marker=dict(symbol="diamond", size=14, color="#FF5722",
                            line=dict(color="white", width=1.5)),
                text=[f"${v:.2f}" for v in obs_y],
                textposition="top center",
                textfont=dict(color="#FF5722", size=12),
            ))

    fig.update_layout(
        title=f"{ticker} — Price Forecast",
        yaxis_title="Price ($)",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        bargap=0.4,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
