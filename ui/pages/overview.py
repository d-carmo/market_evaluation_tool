"""Overview page: all tickers latest predictions dashboard."""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from ui.api_client import APIClient

logger = logging.getLogger(__name__)


def _action_colour(action: str) -> str:
    mapping = {"BUY": "#00c853", "SELL": "#d50000", "HOLD": "#ffa000",
               "STRONG_BUY": "#00e676", "STRONG_SELL": "#ff1744"}
    return mapping.get(action.upper(), "#ffffff")


def render(client: APIClient) -> None:
    st.header("Market Overview")

    # ── Ticker management ────────────────────────────────────────────────────
    try:
        tickers: list[dict] = client.get_tickers()
    except Exception as exc:
        logger.error("Failed to fetch tickers: %s", exc)
        st.error("Cannot connect to the API. Make sure the API server is running.")
        return

    with st.expander("Manage Tickers", expanded=not tickers):
        col_sym, col_btn = st.columns([3, 1])
        with col_sym:
            new_sym = st.text_input("Add ticker", placeholder="e.g. NVDA or BTC-USD",
                                    key="add_ticker_input")
        with col_btn:
            st.write("")  # vertical alignment
            if st.button("Add", key="add_ticker_btn") and new_sym.strip():
                try:
                    client.add_ticker(new_sym.strip().upper())
                    st.success(f"{new_sym.strip().upper()} added.")
                    st.rerun()
                except Exception as exc:
                    logger.error("Failed to add ticker %s: %s", new_sym, exc)
                    st.error("Failed to add ticker. Check the symbol and try again.")

        if tickers:
            st.caption("Active tickers — click Remove to deactivate")
            for t in sorted(tickers, key=lambda x: x["symbol"]):
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.write(t["symbol"])
                c2.write(t["market_type"])
                if c3.button("Remove", key=f"rm_{t['symbol']}"):
                    try:
                        client.remove_ticker(t["symbol"])
                        st.success(f"{t['symbol']} removed.")
                        st.rerun()
                    except Exception as exc:
                        logger.error("Failed to remove ticker %s: %s", t["symbol"], exc)
                        st.error("Failed to remove ticker.")
        else:
            st.info("No tickers yet. Add one above.")

    st.divider()

    # ── Controls ─────────────────────────────────────────────────────────────
    ticker_syms = [t["symbol"] for t in sorted(tickers, key=lambda x: x["symbol"])]

    col_refresh, col_sym, col_run, col_all = st.columns([1, 2, 1, 1])
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_sym:
        analyze_sym = (
            st.selectbox("Ticker", [""] + ticker_syms,
                         label_visibility="collapsed", key="analyze_select")
            if ticker_syms else None
        )
    with col_run:
        run_one = st.button("Analyze", use_container_width=True,
                            disabled=not analyze_sym)
    with col_all:
        run_all = st.button("Analyze All", use_container_width=True,
                            disabled=not ticker_syms)

    if run_one and analyze_sym:
        with st.spinner(f"Analyzing {analyze_sym}…"):
            try:
                result = client.analyze(analyze_sym)
                st.success(f"{analyze_sym}: {result.get('score', {}).get('action', '?')}")
                st.rerun()
            except Exception as exc:
                logger.error("Analysis failed for %s: %s", analyze_sym, exc)
                st.error("Analysis failed. Check the ticker symbol and try again.")

    if run_all and ticker_syms:
        prog = st.progress(0, text="Analyzing tickers…")
        errors = []
        for i, sym in enumerate(ticker_syms):
            try:
                client.analyze(sym)
            except Exception as exc:
                logger.error("Analysis failed for %s: %s", sym, exc)
                errors.append(sym)
            prog.progress((i + 1) / len(ticker_syms), text=f"Analyzed {sym}")
        prog.empty()
        if errors:
            st.warning(f"Failed for: {', '.join(errors)}")
        else:
            st.success(f"All {len(ticker_syms)} tickers analyzed.")
        st.rerun()

    st.divider()

    # ── Predictions table ─────────────────────────────────────────────────────
    if not tickers:
        st.info("Add tickers above to get started.")
        return

    try:
        predictions = client.list_predictions()
    except Exception as exc:
        logger.error("Failed to load predictions: %s", exc)
        st.error("Failed to load predictions from the API.")
        return

    if not predictions:
        st.info("No predictions yet. Use 'Run Analysis' above to generate forecasts.")
        return

    # Fetch accuracy summaries for all tickers that have predictions
    pred_tickers = list({p["ticker"] for p in predictions})
    acc_map: dict[str, dict[int, float | None]] = {}
    for sym in pred_tickers:
        try:
            summaries = client.accuracy_summary(sym)
            acc_map[sym] = {s["horizon_days"]: s["direction_accuracy_pct"] if s["n_evaluated"] > 0 else None
                            for s in summaries}
        except Exception:
            acc_map[sym] = {}

    rows = []
    for p in predictions:
        sym = p["ticker"]
        sh, lh = p["short_horizon"], p["long_horizon"]
        acc = acc_map.get(sym, {})
        sh_acc = acc.get(sh)
        lh_acc = acc.get(lh)
        rows.append({
            "Ticker": sym,
            "Type": p["market_type"],
            "Action": p["action"],
            "Current $": round(p["current_price"], 2),
            "Score": round(p["composite_score"], 3),
            f"{sh}d Mid $": round(p["short_mid"], 2),
            f"{sh}d Dir": p["short_direction"],
            f"{lh}d Mid $": round(p["long_mid"], 2),
            f"{lh}d Dir": p["long_direction"],
            f"{sh}d Acc %": f"{sh_acc:.1f}%" if sh_acc is not None else "—",
            f"{lh}d Acc %": f"{lh_acc:.1f}%" if lh_acc is not None else "—",
            "Model": p["model_used"],
            "Generated": p["generated_at"][:16],
        })

    df = pd.DataFrame(rows)

    def _style_cell(val, col: str) -> str:
        if col == "Action":
            return f"color: {_action_colour(str(val))}; font-weight: bold"
        if col.endswith("Dir"):
            colours = {"UP": "color: #00c853", "DOWN": "color: #d50000", "FLAT": "color: #ffa000"}
            return colours.get(str(val), "")
        if col.endswith("Acc %") and isinstance(val, str) and val != "—":
            pct = float(val.rstrip("%"))
            if pct >= 60:
                return "color: #00c853"
            if pct >= 50:
                return "color: #ffa000"
            return "color: #d50000"
        return ""

    styled = df.style.apply(
        lambda col: [_style_cell(v, col.name) for v in col], axis=0
    )
    selection = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = selection.selection.get("rows", [])
    if selected_rows:
        clicked_ticker = df.iloc[selected_rows[0]]["Ticker"]
        st.session_state["selected_ticker"] = clicked_ticker
        st.session_state["current_page"] = "Forecasts"
        st.rerun()
