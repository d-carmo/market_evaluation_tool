import logging
from datetime import datetime, timezone

import yfinance as yf

from data.exceptions import FetchError
from data.fetchers.base import AbstractFetcher
from data.models import MarketData
from data.normalizer import normalize

logger = logging.getLogger(__name__)


class CryptoFetcher(AbstractFetcher):
    """Fetches cryptocurrency OHLCV data via yfinance (e.g. BTC-USD)."""

    def fetch(self, ticker: str, days: int) -> MarketData:
        try:
            raw = yf.Ticker(ticker).history(period=f"{days}d", auto_adjust=True)
        except Exception as exc:
            raise FetchError(f"yfinance fetch failed for {ticker}: {exc}") from exc

        if raw is None or raw.empty:
            raise FetchError(f"No data returned by yfinance for {ticker}")

        df = normalize(raw, source="yfinance_crypto", ticker=ticker)
        return MarketData(
            ticker=ticker,
            market_type="crypto",
            fetched_at=datetime.now(timezone.utc),
            source="yfinance_crypto",
            df=df,
        )

    def supports(self, ticker: str) -> bool:
        return ticker.endswith("-USD")
