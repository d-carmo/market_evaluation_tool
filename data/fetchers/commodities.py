import logging
from datetime import datetime, timezone

import yfinance as yf

from data.exceptions import FetchError
from data.fetchers.base import AbstractFetcher
from data.models import MarketData
from data.normalizer import normalize

logger = logging.getLogger(__name__)


class CommodityFetcher(AbstractFetcher):
    """Fetches commodity futures OHLCV data via yfinance (e.g. GC=F)."""

    def fetch(self, ticker: str, days: int) -> MarketData:
        try:
            raw = yf.Ticker(ticker).history(period=f"{days}d", auto_adjust=True)
        except Exception as exc:
            raise FetchError(f"yfinance fetch failed for {ticker}: {exc}") from exc

        if raw is None or raw.empty:
            raise FetchError(f"No data returned by yfinance for {ticker}")

        df = normalize(raw, source="yfinance_futures", ticker=ticker)
        return MarketData(
            ticker=ticker,
            market_type="commodity",
            fetched_at=datetime.now(timezone.utc),
            source="yfinance_futures",
            df=df,
        )

    def supports(self, ticker: str) -> bool:
        return ticker.endswith("=F")
