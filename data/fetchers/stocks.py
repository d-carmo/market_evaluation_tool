import logging
from datetime import datetime, timezone

import yfinance as yf

from data.exceptions import FetchError
from data.fetchers.base import AbstractFetcher
from data.models import MarketData
from data.normalizer import normalize

logger = logging.getLogger(__name__)


class StockFetcher(AbstractFetcher):
    """Fetches equity OHLCV data via yfinance."""

    def fetch(self, ticker: str, days: int) -> MarketData:
        try:
            raw = yf.Ticker(ticker).history(period=f"{days}d", auto_adjust=True)
        except Exception as exc:
            raise FetchError(f"yfinance fetch failed for {ticker}: {exc}") from exc

        if raw is None or raw.empty:
            raise FetchError(f"No data returned by yfinance for {ticker}")

        df = normalize(raw, source="yfinance", ticker=ticker)
        return MarketData(
            ticker=ticker,
            market_type="stock",
            fetched_at=datetime.now(timezone.utc),
            source="yfinance",
            df=df,
        )

    def supports(self, ticker: str) -> bool:
        return not ticker.endswith("-USD") and not ticker.endswith("=F")
