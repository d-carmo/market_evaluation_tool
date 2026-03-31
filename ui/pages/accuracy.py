"""Accuracy tracking page: direction accuracy and price error over time."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui.api_client import APIClient

logger = logging.getLogger(__name__)

_ROLLING_WINDOW = 20   # bars for the rolling accuracy ribbon

_PERIOD_OPTIONS: dict[str, int | None] = {
    "Past month":    30,
    "Past 3 months": 90,
    "Past 6 months": 180,
    "Past year":     365,
    "All time":      None,
}

# Per-model column names in accuracy records (from simulation)
_PER_MODEL_COLS = {
    "XGBoost":       "xgb_direction",
    "LSTM":          "lstm_direction",
    "Random Forest": "rf_direction",
}


def _accuracy_evolution_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Build a two-panel Plotly figure:
    - Top: rolling accuracy % per horizon with confidence band + 50% baseline.
    - Bottom: cumulative prediction count per horizon (shows data density).
    """
    horizons = sorted(df["horizon_days"].unique())
    colours   = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
        h = hex_colour.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
        subplot_titles=["Rolling Direction Accuracy", "Cumulative Predictions"],
    )

    for i, h in enumerate(horizons):
        col = colours[i % len(colours)]
        sub = df[df["horizon_days"] == h].copy().sort_values("evaluated_at")

        # Rolling accuracy (centred rolling mean over last N preds)
        roll = sub["direction_correct"].rolling(_ROLLING_WINDOW, min_periods=3).mean() * 100
        # Running (cumulative) accuracy for comparison
        running = sub["direction_correct"].expanding().mean() * 100

        # Rolling upper/lower band (±1 std of correctness in window)
        roll_std = sub["direction_correct"].rolling(_ROLLING_WINDOW, min_periods=3).std().fillna(0) * 100
        upper = (roll + roll_std).clip(upper=100)
        lower = (roll - roll_std).clip(lower=0)

        x = sub["evaluated_at"]

        # Confidence band (fill between upper and lower)
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x.iloc[::-1]]),
            y=pd.concat([upper, lower.iloc[::-1]]),
            fill="toself",
            fillcolor=_hex_to_rgba(col, 0.10),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

        # Rolling accuracy line
        fig.add_trace(go.Scatter(
            x=x, y=roll,
            name=f"{h}d rolling ({_ROLLING_WINDOW})",
            line=dict(color=col, width=2),
            mode="lines",
        ), row=1, col=1)

        # Running accuracy (dashed, thinner)
        fig.add_trace(go.Scatter(
            x=x, y=running,
            name=f"{h}d cumulative",
            line=dict(color=col, width=1, dash="dot"),
            mode="lines",
            opacity=0.6,
        ), row=1, col=1)

        # Cumulative prediction count
        fig.add_trace(go.Scatter(
            x=x,
            y=list(range(1, len(sub) + 1)),
            name=f"{h}d count",
            line=dict(color=col, width=1.5),
            mode="lines",
            showlegend=False,
        ), row=2, col=1)

    # 50% baseline
    fig.add_hline(
        y=50, line_dash="dash", line_color="rgba(150,150,150,0.5)",
        annotation_text="50% random", annotation_position="bottom right",
        row=1, col=1,
    )

    date_range = (
        f"{df['evaluated_at'].min().strftime('%Y-%m-%d')}"
        f" → {df['evaluated_at'].max().strftime('%Y-%m-%d')}"
    )
    fig.update_layout(
        title=dict(
            text=f"{ticker} — Accuracy Evolution  <span style='font-size:13px;color:gray'>{date_range}</span>",
            font=dict(size=16),
        ),
        height=480,
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Evaluated date", row=2, col=1)

    return fig


def _per_model_accuracy_chart(
    df: pd.DataFrame,
    ticker: str,
    model_name: str,
    direction_col: str,
) -> go.Figure | None:
    """Rolling accuracy chart for a single model's per-component direction predictions.

    ``direction_col`` is one of xgb_direction / lstm_direction / rf_direction.
    Returns None when no data is available for this model.
    """
    if direction_col not in df.columns:
        return None

    sub = df[df[direction_col].notna()].copy().sort_values("evaluated_at")
    if len(sub) < 3:
        return None

    # Correct when individual model direction matches actual
    sub = sub.copy()
    sub["model_correct"] = (sub[direction_col] == sub["actual_direction"]).astype(int)
    horizons = sorted(sub["horizon_days"].unique())
    colours  = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
        subplot_titles=["Rolling Direction Accuracy", "Cumulative Predictions"],
    )

    for i, h in enumerate(horizons):
        col  = colours[i % len(colours)]
        hsub = sub[sub["horizon_days"] == h].copy()
        if len(hsub) < 3:
            continue

        roll     = hsub["model_correct"].rolling(_ROLLING_WINDOW, min_periods=3).mean() * 100
        running  = hsub["model_correct"].expanding().mean() * 100
        roll_std = hsub["model_correct"].rolling(_ROLLING_WINDOW, min_periods=3).std().fillna(0) * 100
        upper    = (roll + roll_std).clip(upper=100)
        lower    = (roll - roll_std).clip(lower=0)
        x        = hsub["evaluated_at"]

        r, g, b  = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        fill_col = f"rgba({r},{g},{b},0.10)"

        fig.add_trace(go.Scatter(
            x=pd.concat([x, x.iloc[::-1]]),
            y=pd.concat([upper, lower.iloc[::-1]]),
            fill="toself", fillcolor=fill_col,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=roll,
            name=f"{h}d rolling ({_ROLLING_WINDOW})",
            line=dict(color=col, width=2), mode="lines",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=running,
            name=f"{h}d cumulative",
            line=dict(color=col, width=1, dash="dot"),
            mode="lines", opacity=0.6,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=list(range(1, len(hsub) + 1)),
            name=f"{h}d count",
            line=dict(color=col, width=1.5),
            mode="lines", showlegend=False,
        ), row=2, col=1)

    fig.add_hline(
        y=50, line_dash="dash", line_color="rgba(150,150,150,0.5)",
        annotation_text="50% random", annotation_position="bottom right",
        row=1, col=1,
    )

    date_range = (
        f"{sub['evaluated_at'].min().strftime('%Y-%m-%d')}"
        f" → {sub['evaluated_at'].max().strftime('%Y-%m-%d')}"
    )
    n_total  = len(sub)
    n_correct = int(sub["model_correct"].sum())
    fig.update_layout(
        title=dict(
            text=(
                f"{ticker} — {model_name} Accuracy  "
                f"<span style='font-size:13px;color:gray'>{date_range} · "
                f"{n_correct}/{n_total} correct ({n_correct/n_total*100:.1f}%)</span>"
            ),
            font=dict(size=15),
        ),
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Evaluated date", row=2, col=1)
    return fig


def render(client: APIClient) -> None:
    st.header("Prediction Accuracy Tracker")

    # ── Ticker selector + evaluation trigger ─────────────────────────────────
    try:
        active_tickers = sorted(t["symbol"] for t in client.get_tickers())
    except Exception as exc:
        logger.error("Failed to load tickers: %s", exc)
        active_tickers = []

    # Sync widget to shared ticker state before instantiation.
    if active_tickers:
        st.session_state["accuracy_ticker"] = st.session_state.get("selected_ticker", active_tickers[0])

    col_ticker, col_period, col_eval = st.columns([2, 2, 1])
    with col_ticker:
        if active_tickers:
            ticker = st.selectbox(
                "Select ticker", active_tickers, key="accuracy_ticker",
                on_change=lambda: st.session_state.update(selected_ticker=st.session_state["accuracy_ticker"]),
            )
        else:
            st.info("No tickers configured. Add tickers on the Overview page.")
            return
    with col_period:
        selected_period = st.selectbox(
            "Period", list(_PERIOD_OPTIONS.keys()),
            index=len(_PERIOD_OPTIONS) - 1,  # default: All time
            key="accuracy_period",
        )
    with col_eval:
        st.write("")
        if st.button("Run Evaluation", use_container_width=True):
            with st.spinner("Evaluating past predictions…"):
                try:
                    result = client.trigger_evaluation()
                    n = result["evaluated"]
                    if n > 0:
                        st.success(f"Evaluated {n} record(s).")
                    else:
                        st.info(
                            "Nothing new to evaluate. Either all past-horizon predictions "
                            "are already evaluated (e.g. from `training.py --save-to-db`), "
                            "or live predictions haven't reached their horizon yet. "
                            "Accuracy data from the training simulation is shown below."
                        )
                except Exception as exc:
                    logger.error("Accuracy evaluation failed: %s", exc)
                    st.error("Evaluation failed. Try again later.")

    st.divider()

    # ── Fetch all records once, apply period filter ───────────────────────────
    try:
        raw_records = client.accuracy_records(ticker, limit=2000)
    except Exception as exc:
        logger.error("Cannot load accuracy data for %s: %s", ticker, exc)
        st.error("Cannot load accuracy data.")
        return

    if not raw_records:
        st.info(
            f"No accuracy data yet for **{ticker}**. "
            "Predictions need to age past their forecast horizon first. "
            "You can run the training simulation (`python training.py`) "
            "to generate historical data immediately."
        )
        st.divider()
    else:
        df_all = pd.DataFrame(raw_records)
        df_all["evaluated_at"] = pd.to_datetime(df_all["evaluated_at"], format="ISO8601", utc=True)
        df_all = df_all.sort_values("evaluated_at")

        period_days = _PERIOD_OPTIONS[selected_period]
        if period_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
            df = df_all[df_all["evaluated_at"] >= cutoff].copy()
        else:
            df = df_all.copy()

        if df.empty:
            st.info(f"No accuracy records in the selected period ({selected_period}).")
            st.divider()
        else:
            # ── Summary metrics (computed from filtered data) ─────────────────
            cols = st.columns(len(df["horizon_days"].unique()))
            for i, h in enumerate(sorted(df["horizon_days"].unique())):
                hdf = df[df["horizon_days"] == h]
                n = len(hdf)
                acc = hdf["direction_correct"].mean() * 100
                err = hdf["price_error_pct"].abs().mean()
                with cols[i]:
                    st.metric(
                        f"{h}d Direction Accuracy",
                        f"{acc:.1f}%",
                        help=f"Based on {n} evaluated predictions in selected period",
                    )
                    st.metric(f"{h}d Mean Price Error", f"{err:.2f}%")

            st.divider()

            # ── Accuracy evolution chart ──────────────────────────────────────
            if len(df) >= 3:
                st.plotly_chart(
                    _accuracy_evolution_chart(df, ticker),
                    use_container_width=True,
                )
                n_total = len(df)
                n_correct = df["direction_correct"].sum()
                st.caption(
                    f"{n_total} records · "
                    f"{n_correct/n_total*100:.1f}% overall direction accuracy · "
                    f"Rolling window: {_ROLLING_WINDOW} predictions"
                )
            else:
                st.info("Not enough records in the selected period to plot.")

            st.divider()

            # ── Per-model accuracy charts (XGB / LSTM / RF) ──────────────────
            st.subheader("Per-Model Accuracy Evolution")
            st.caption(
                "Shows how each component model's individual direction predictions "
                "compare against actual outcomes. Data is populated from simulation "
                "runs (training.py) where per-model directions are captured."
            )
            any_per_model = any(
                col in df.columns and df[col].notna().any()
                for col in ("xgb_direction", "lstm_direction", "rf_direction")
            )
            if any_per_model:
                for model_name, direction_col in _PER_MODEL_COLS.items():
                    fig_pm = _per_model_accuracy_chart(df, ticker, model_name, direction_col)
                    if fig_pm is not None:
                        with st.expander(f"{model_name} accuracy evolution", expanded=True):
                            st.plotly_chart(fig_pm, use_container_width=True)
                    else:
                        st.info(
                            f"No {model_name} per-component data yet for **{ticker}**. "
                            "Run `python training.py --save-to-db` to generate simulation data."
                        )
            else:
                st.info(
                    "Per-model direction data not yet available. "
                    "Run `python training.py --save-to-db` to populate XGBoost, LSTM, "
                    "and Random Forest individual accuracy from the simulation."
                )

            st.divider()

            # ── Records table ─────────────────────────────────────────────────
            with st.expander("Individual records", expanded=False):
                rows = [{
                    "Evaluated":   pd.Timestamp(r["evaluated_at"]).strftime("%Y-%m-%d %H:%M"),
                    "Horizon":     f"{r['horizon_days']}d",
                    "Predicted":   r["predicted_direction"],
                    "Actual":      r["actual_direction"],
                    "Correct":     "✓" if r["direction_correct"] else "✗",
                    "Pred Mid $":  round(r["predicted_mid"], 2),
                    "Actual $":    round(r["actual_price"], 2),
                    "Price Err %": round(r["price_error_pct"], 2),
                } for r in df.to_dict("records")]
                df_tbl = pd.DataFrame(rows)

                def _colour_correct(val: str) -> str:
                    return "color: #00c853" if val == "✓" else "color: #d50000"

                st.dataframe(
                    df_tbl.style.map(_colour_correct, subset=["Correct"]),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── ML Retraining ─────────────────────────────────────────────────────────
    st.subheader("ML Retraining")
    col_rt, col_train = st.columns(2)
    with col_rt:
        if st.button(f"Retrain {ticker}", use_container_width=True):
            with st.spinner(f"Retraining {ticker}…"):
                try:
                    res = client.retrain_ticker(ticker)
                    if res["success"]:
                        st.success(res["message"])
                    else:
                        st.warning(res["message"])
                except Exception as exc:
                    logger.error("Retrain failed for %s: %s", ticker, exc)
                    st.error("Retrain failed.")
    with col_train:
        if st.button("Train All Tickers", use_container_width=True):
            with st.spinner("Training all tickers…"):
                try:
                    res = client.train_all()
                    ok = sum(1 for r in res["results"] if r["success"])
                    total = len(res["results"])
                    st.success(f"Trained {ok}/{total} ticker-horizon pairs")
                except Exception as exc:
                    logger.error("Train all failed: %s", exc)
                    st.error("Training failed.")
