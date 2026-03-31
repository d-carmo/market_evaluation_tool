"""Bloomberg Terminal data fetcher via blpapi.

Requirements:
  - ``blpapi`` Python package:  pip install blpapi
  - Bloomberg Terminal running locally **or** Bloomberg Server API (B-PIPE / SAPI)
  - Set env vars if non-default:
      BLOOMBERG_HOST  (default: localhost)
      BLOOMBERG_PORT  (default: 8194)

Enable by setting BLOOMBERG_ENABLED=true in .env.
When enabled, Bloomberg is tried first for every ticker; on failure the
standard yfinance fetcher is used as fallback so the app keeps running.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from data.exceptions import FetchError
from data.fetchers.base import AbstractFetcher
from data.models import MarketData, MarketType
from data.normalizer import normalize

logger = logging.getLogger(__name__)

# ── Bloomberg field names (HistoricalDataRequest) ──────────────────────────
_BBG_FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]

# ── Ticker mapping: yfinance format → Bloomberg security identifier ─────────
# Crypto
_CRYPTO_MAP: dict[str, str] = {
    "BTC-USD":   "XBT Curncy",
    "ETH-USD":   "ETH Curncy",
    "SOL-USD":   "SOL Curncy",
    "ADA-USD":   "ADA Curncy",
    "DOGE-USD":  "DOGE Curncy",
    "XRP-USD":   "XRP Curncy",
    "LTC-USD":   "LTC Curncy",
    "LINK-USD":  "LINK Curncy",
    "DOT-USD":   "DOT Curncy",
    "AVAX-USD":  "AVAX Curncy",
    "MATIC-USD": "MATIC Curncy",
    "BNB-USD":   "BNB Curncy",
}

# Continuous futures (front month)
_COMMODITY_MAP: dict[str, str] = {
    "GC=F":  "GC1 Comdty",   # Gold
    "SI=F":  "SI1 Comdty",   # Silver
    "CL=F":  "CL1 Comdty",   # WTI Crude Oil
    "BZ=F":  "CO1 Comdty",   # Brent Crude
    "NG=F":  "NG1 Comdty",   # Natural Gas
    "HG=F":  "HG1 Comdty",   # Copper
    "ZW=F":  "W 1 Comdty",   # Wheat
    "ZC=F":  "C 1 Comdty",   # Corn
    "ZS=F":  "S 1 Comdty",   # Soybeans
    "PL=F":  "PL1 Comdty",   # Platinum
    "PA=F":  "PA1 Comdty",   # Palladium
}


def to_bbg_ticker(ticker: str, market_type: MarketType) -> str:
    """Convert a yfinance-style ticker to its Bloomberg security identifier.

    Falls back to ``<TICKER> US Equity`` for unknown equity symbols.
    Unknown crypto/commodity tickers are passed through unchanged.
    """
    if market_type == "crypto":
        return _CRYPTO_MAP.get(ticker, ticker)
    if market_type == "commodity":
        return _COMMODITY_MAP.get(ticker, ticker)
    # Equities: append exchange qualifier if not already present
    if " " not in ticker:
        return f"{ticker} US Equity"
    return ticker


def _detect_market_type(ticker: str) -> MarketType:
    if ticker.endswith("-USD"):
        return "crypto"
    if ticker.endswith("=F"):
        return "commodity"
    return "stock"


class BloombergFetcher(AbstractFetcher):
    """Fetch OHLCV data from the Bloomberg Terminal via blpapi.

    The session is opened once per process and reused for all requests.
    If the session drops, the next call will attempt to reconnect.
    """

    _session: Any = None
    _connected: bool = False

    # ── Availability ──────────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        """Return True if blpapi is importable (Bloomberg may still be unreachable)."""
        try:
            import blpapi  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def is_enabled(cls) -> bool:
        """Return True when BLOOMBERG_ENABLED=true and blpapi is installed."""
        return (
            os.getenv("BLOOMBERG_ENABLED", "false").strip().lower() in ("1", "true", "yes")
            and cls.is_available()
        )

    # ── Session management ────────────────────────────────────────────────

    @classmethod
    def _open_session(cls) -> Any:
        """Open (or reuse) a connected blpapi session with the refdata service."""
        import blpapi

        if cls._session is not None and cls._connected:
            return cls._session

        host = os.getenv("BLOOMBERG_HOST", "localhost")
        port = int(os.getenv("BLOOMBERG_PORT", "8194"))

        opts = blpapi.SessionOptions()
        opts.setServerHost(host)
        opts.setServerPort(port)

        session = blpapi.Session(opts)
        if not session.start():
            raise FetchError(f"Bloomberg session failed to start ({host}:{port})")
        if not session.openService("//blp/refdata"):
            session.stop()
            raise FetchError("Bloomberg refdata service unavailable")

        cls._session = session
        cls._connected = True
        logger.info("Bloomberg session connected to %s:%d", host, port)
        return session

    @classmethod
    def close_session(cls) -> None:
        """Explicitly close the Bloomberg session (call on app shutdown)."""
        if cls._session is not None:
            try:
                cls._session.stop()
            except Exception:
                pass
            finally:
                cls._session = None
                cls._connected = False

    # ── AbstractFetcher interface ─────────────────────────────────────────

    def fetch(self, ticker: str, days: int) -> MarketData:
        """Fetch *days* of daily OHLCV from Bloomberg for *ticker*.

        Raises FetchError on any Bloomberg error so the caller can fall back.
        """
        import blpapi

        market_type = _detect_market_type(ticker)
        bbg_ticker = to_bbg_ticker(ticker, market_type)

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        try:
            session = self._open_session()
            service = session.getService("//blp/refdata")

            req = service.createRequest("HistoricalDataRequest")
            req.getElement("securities").appendValue(bbg_ticker)
            for field in _BBG_FIELDS:
                req.getElement("fields").appendValue(field)
            req.set("startdt", start_date.strftime("%Y%m%d"))
            req.set("enddt", end_date.strftime("%Y%m%d"))
            req.set("periodicitySelection", "DAILY")
            req.set("adjustmentSplit", True)
            req.set("adjustmentAbnormal", True)
            req.set("adjustmentNormal", True)

            session.sendRequest(req)
            records = self._collect_response(session, bbg_ticker)

        except FetchError:
            # Mark session as potentially broken so next call reconnects
            cls = type(self)
            cls._connected = False
            raise
        except Exception as exc:
            type(self)._connected = False
            raise FetchError(
                f"Bloomberg fetch failed for {bbg_ticker} ({ticker}): {exc}"
            ) from exc

        if not records:
            raise FetchError(f"Bloomberg returned no data for {bbg_ticker}")

        df_raw = pd.DataFrame(records).set_index("date")
        df_raw.index = pd.to_datetime(df_raw.index, utc=True)

        df = normalize(df_raw, source="bloomberg", ticker=ticker)
        return MarketData(
            ticker=ticker,
            market_type=market_type,
            fetched_at=datetime.now(timezone.utc),
            source="bloomberg",
            df=df,
        )

    def supports(self, ticker: str) -> bool:
        return self.is_available()

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _collect_response(session: Any, bbg_ticker: str) -> list[dict]:
        """Drain Bloomberg response events and return parsed OHLCV records."""
        import blpapi

        records: list[dict] = []
        timeout_ms = 30_000  # 30 s per event poll

        while True:
            event = session.nextEvent(timeout_ms)

            if event.eventType() == blpapi.Event.TIMEOUT:
                raise FetchError(f"Bloomberg response timed out for {bbg_ticker}")

            for msg in event:
                if msg.hasElement("responseError"):
                    err_msg = msg.getElement("responseError").getElementAsString("message")
                    raise FetchError(
                        f"Bloomberg response error for {bbg_ticker}: {err_msg}"
                    )

                if not msg.hasElement("securityData"):
                    continue

                sec_data = msg.getElement("securityData")

                if sec_data.hasElement("securityError"):
                    err_msg = sec_data.getElement("securityError").getElementAsString("message")
                    raise FetchError(
                        f"Bloomberg security error for {bbg_ticker}: {err_msg}"
                    )

                field_arr = sec_data.getElement("fieldData")
                for i in range(field_arr.numValues()):
                    row = field_arr.getValue(i)

                    def _float(name: str) -> float:
                        return float(row.getElementAsFloat(name)) if row.hasElement(name) else 0.0

                    records.append({
                        "date": row.getElementAsDatetime("date").date(),
                        "Open":   _float("PX_OPEN"),
                        "High":   _float("PX_HIGH"),
                        "Low":    _float("PX_LOW"),
                        "Close":  _float("PX_LAST"),
                        "Volume": _float("PX_VOLUME"),
                    })

            if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                if event.eventType() == blpapi.Event.RESPONSE:
                    break  # final fragment received

        return records
