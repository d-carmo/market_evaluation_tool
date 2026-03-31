import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.cache import DataCache
from data.exceptions import FetchError
from data.fetchers.bloomberg import BloombergFetcher
from data.fetchers.commodities import CommodityFetcher
from data.fetchers.crypto import CryptoFetcher
from data.fetchers.stocks import StockFetcher
from data.merger import merge_sources
from data.models import MarketData, MarketType

logger = logging.getLogger(__name__)

_FETCHER_MAP = {
    "stock": StockFetcher,
    "crypto": CryptoFetcher,
    "commodity": CommodityFetcher,
}


def fetch_all(
    tickers: dict[str, MarketType],
    days: int,
    cache: DataCache,
    max_workers: int = 8,
) -> dict[str, MarketData]:
    """Fetch all tickers concurrently (ThreadPoolExecutor).

    Cache hits are returned without a network call.

    When BLOOMBERG_ENABLED=true and blpapi is installed, **both** Bloomberg
    and yfinance are queried for every ticker and their results are merged:
      - Bloomberg values take precedence on overlapping dates (higher quality
        corporate-action adjustments).
      - yfinance fills any dates Bloomberg is missing, extending coverage.

    If either source fails entirely the other is used alone.
    If both fail the ticker is excluded with a warning.
    """
    _bbg_enabled = BloombergFetcher.is_enabled()
    if _bbg_enabled:
        logger.info(
            "Dual-source mode: fetching from Bloomberg + yfinance for all tickers"
        )

    def _fetch_yfinance(ticker: str, market_type: MarketType) -> MarketData | None:
        try:
            fetcher = _FETCHER_MAP[market_type]()
            data = fetcher.fetch(ticker, days)
            logger.debug("yfinance fetched %s (%d rows)", ticker, len(data.df))
            return data
        except FetchError as exc:
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
            return None
        except Exception as exc:
            logger.warning("yfinance unexpected error for %s: %s", ticker, exc)
            return None

    def _fetch_bloomberg(ticker: str) -> MarketData | None:
        try:
            data = BloombergFetcher().fetch(ticker, days)
            logger.debug("Bloomberg fetched %s (%d rows)", ticker, len(data.df))
            return data
        except FetchError as exc:
            logger.warning("Bloomberg fetch failed for %s: %s", ticker, exc)
            return None

    def _fetch_one(ticker: str, market_type: MarketType) -> tuple[str, MarketData | None]:
        if cache.is_valid(ticker):
            cached = cache.get(ticker)
            if cached is not None:
                logger.info("Cache hit: %s", ticker)
                return ticker, cached

        yf_data = _fetch_yfinance(ticker, market_type)

        if not _bbg_enabled:
            # Single-source mode: just use yfinance
            if yf_data is not None:
                cache.set(yf_data)
            return ticker, yf_data

        # Dual-source mode: fetch Bloomberg and merge with yfinance
        bbg_data = _fetch_bloomberg(ticker)

        if bbg_data is not None and yf_data is not None:
            merged = merge_sources(primary=bbg_data, secondary=yf_data)
            cache.set(merged)
            logger.info(
                "Fetched %s from bloomberg+yfinance (%d rows total)",
                ticker, len(merged.df),
            )
            return ticker, merged

        # One source failed — use whichever succeeded
        data = bbg_data or yf_data
        if data is not None:
            source_used = "bloomberg" if bbg_data is not None else "yfinance"
            logger.warning(
                "%s: only %s data available (other source failed)",
                ticker, source_used,
            )
            cache.set(data)
        return ticker, data

    results: dict[str, MarketData] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one, ticker, mtype): ticker
            for ticker, mtype in tickers.items()
        }
        for future in as_completed(futures):
            ticker, data = future.result()
            if data is not None:
                results[ticker] = data

    return results
