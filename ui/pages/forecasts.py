"""Forecasts page: deep-dive into a single ticker."""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from ui.api_client import APIClient
from ui.components.candlestick import render_candlestick
from ui.components.forecast_chart import render_forecast_chart
from ui.components.signal_table import render_signal_table

logger = logging.getLogger(__name__)


def render(client: APIClient) -> None:
    st.header("Ticker Deep-Dive")

    # Load active tickers for the dropdown
    try:
        active_tickers = sorted(
            t["symbol"] for t in client.get_tickers()
        )
    except Exception as exc:
        logger.error("Failed to load tickers: %s", exc)
        active_tickers = []

    if not active_tickers:
        st.info("No tickers configured. Add tickers on the Overview page.")
        return

    # Sync widget to shared ticker state before instantiation so cross-page
    # navigation (overview click or accuracy tab switch) is reflected here.
    st.session_state["forecast_ticker"] = st.session_state.get("selected_ticker", active_tickers[0])

    col_sym, col_latest, col_analyze = st.columns([2, 1, 1])
    with col_sym:
        ticker = st.selectbox(
            "Select ticker", active_tickers, key="forecast_ticker",
            on_change=lambda: st.session_state.update(selected_ticker=st.session_state["forecast_ticker"]),
        )
    with col_latest:
        st.write("")
        load_latest = st.button("Load Latest", use_container_width=True)
    with col_analyze:
        st.write("")
        run_new = st.button("Run Fresh Analysis", use_container_width=True)

    prediction_data: dict | None = None

    # Auto-load latest when ticker changes or user clicks Load Latest
    if load_latest or ticker:
        with st.spinner("Loading…"):
            try:
                prediction_data = client.get_latest(ticker)
            except Exception as exc:
                logger.error("No stored prediction for %s: %s", ticker, exc)
                st.info(
                    f"No prediction stored for **{ticker}** yet. "
                    "Click 'Run Fresh Analysis' to generate one."
                )

    if run_new:
        with st.spinner(f"Analyzing {ticker}…"):
            try:
                client.analyze(ticker)
                st.success("Analysis complete!")
                prediction_data = client.get_latest(ticker)
            except Exception as exc:
                logger.error("Analysis failed for %s: %s", ticker, exc)
                st.error("Analysis failed. Check the ticker symbol and try again.")

    if prediction_data is None:
        return

    # ── Headline metrics ───────────────────────────────────────────────
    short_dir = prediction_data["short_direction"]
    long_dir  = prediction_data["long_direction"]
    current   = prediction_data["current_price"]
    short_delta = round(prediction_data["short_mid"] - current, 4)
    long_delta  = round(prediction_data["long_mid"]  - current, 4)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", prediction_data["action"])
    c2.metric("Current Price", f"${current:.2f}")
    c3.metric(
        f"{prediction_data['short_horizon']}d Forecast · {short_dir}",
        f"${prediction_data['short_mid']:.2f}",
        delta=short_delta,
    )
    c4.metric(
        f"{prediction_data['long_horizon']}d Forecast · {long_dir}",
        f"${prediction_data['long_mid']:.2f}",
        delta=long_delta,
    )

    st.caption(
        f"Model: {prediction_data['model_used']} · "
        f"Generated: {prediction_data['generated_at'][:16]}"
    )
    st.divider()

    tab_chart, tab_forecast, tab_signals, tab_history = st.tabs(
        ["Chart", "Forecast", "Signals", "History"]
    )

    with tab_chart:
        with st.spinner("Loading chart…"):
            try:
                chart = client.get_chart_data(ticker)
                render_candlestick(chart)
            except Exception as exc:
                logger.error("Chart failed for %s: %s", ticker, exc)
                st.error("Chart data unavailable.")

    with tab_forecast:
        pseudo_pred = {
            "current_price": prediction_data["current_price"],
            "short_forecast": {
                "horizon_days": prediction_data["short_horizon"],
                "mid":  prediction_data["short_mid"],
                "low":  prediction_data.get("short_low",  prediction_data["short_mid"] * 0.97),
                "high": prediction_data.get("short_high", prediction_data["short_mid"] * 1.03),
                "direction": prediction_data["short_direction"],
            },
            "long_forecast": {
                "horizon_days": prediction_data["long_horizon"],
                "mid":  prediction_data["long_mid"],
                "low":  prediction_data.get("long_low",  prediction_data["long_mid"] * 0.97),
                "high": prediction_data.get("long_high", prediction_data["long_mid"] * 1.03),
                "direction": prediction_data["long_direction"],
            },
        }

        # Look up observed prices from accuracy records for this prediction
        observed: dict[int, float] = {}
        pred_id = prediction_data.get("id")
        if pred_id is not None:
            try:
                acc_records = client.accuracy_records(ticker, limit=200)
                for rec in acc_records:
                    if rec.get("prediction_id") == pred_id:
                        observed[rec["horizon_days"]] = rec["actual_price"]
            except Exception:
                pass

        render_forecast_chart(pseudo_pred, ticker, observed=observed or None)

    with tab_signals:
        st.subheader("Signal Details")
        try:
            live = client.analyze(ticker)
            render_signal_table(live.get("score", {}).get("signal_details", {}))
        except Exception as exc:
            logger.error("Signal details unavailable for %s: %s", ticker, exc)
            st.caption("Signal details unavailable. Run a fresh analysis.")

    with tab_history:
        st.subheader("Prediction History")
        try:
            history = client.get_history(ticker, limit=30)
            if history:
                rows = []
                for h in history:
                    rows.append({
                        "Generated": h["generated_at"][:16],
                        "Action": h["action"],
                        "Current $": round(h["current_price"], 2),
                        "Score": round(h["composite_score"], 3),
                        f"{h['short_horizon']}d Mid": round(h["short_mid"], 2),
                        f"{h['long_horizon']}d Mid": round(h["long_mid"], 2),
                        "Model": h["model_used"],
                    })
                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )
            else:
                st.info("No history yet for this ticker.")
        except Exception as exc:
            logger.error("History unavailable for %s: %s", ticker, exc)
            st.error("History unavailable.")
