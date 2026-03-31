from abc import ABC, abstractmethod

from data.models import MarketData


class AbstractFetcher(ABC):
    """Contract every market-specific fetcher must implement."""

    @abstractmethod
    def fetch(self, ticker: str, days: int) -> MarketData:
        """Fetch OHLCV data for *ticker* going back *days* calendar days.

        Returns a MarketData with a normalized DataFrame.
        Raises FetchError on network failure, empty response, or API error.
        """

    @abstractmethod
    def supports(self, ticker: str) -> bool:
        """Return True if this fetcher can handle the given ticker symbol."""
