from datetime import datetime, timezone

import pandas as pd
import pytest

from data.exceptions import FetchError
from data.fetchers.commodities import CommodityFetcher
from data.fetchers.crypto import CryptoFetcher
from data.fetchers.stocks import StockFetcher


def _mock_df():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "Open": [100.0]*5, "High": [105.0]*5, "Low": [95.0]*5,
        "Close": [102.0]*5, "Volume": [1e6]*5,
    }, index=idx)


# ── StockFetcher ──────────────────────────────────────────────────────────────

def test_stock_fetcher_returns_market_data(mocker):
    mocker.patch("data.fetchers.stocks.yf.download", return_value=_mock_df())
    result = StockFetcher().fetch("AAPL", days=5)
    assert result.ticker == "AAPL"
    assert result.market_type == "stock"
    assert result.source == "yfinance"
    assert len(result.df) == 5
    assert set(result.df.columns) == {"open", "high", "low", "close", "volume"}


def test_stock_fetcher_raises_on_empty(mocker):
    mocker.patch("data.fetchers.stocks.yf.download", return_value=pd.DataFrame())
    with pytest.raises(FetchError):
        StockFetcher().fetch("AAPL", days=5)


def test_stock_fetcher_raises_on_exception(mocker):
    mocker.patch("data.fetchers.stocks.yf.download", side_effect=RuntimeError("network"))
    with pytest.raises(FetchError):
        StockFetcher().fetch("AAPL", days=5)


def test_stock_fetcher_does_not_support_crypto():
    assert StockFetcher().supports("BTC-USD") is False


def test_stock_fetcher_does_not_support_futures():
    assert StockFetcher().supports("GC=F") is False


def test_stock_fetcher_supports_equities():
    assert StockFetcher().supports("AAPL") is True


# ── CryptoFetcher ─────────────────────────────────────────────────────────────

def test_crypto_fetcher_supports_usd_pairs():
    assert CryptoFetcher().supports("BTC-USD") is True


def test_crypto_fetcher_does_not_support_equities():
    assert CryptoFetcher().supports("AAPL") is False


def test_crypto_fetcher_returns_market_data(mocker):
    mocker.patch("data.fetchers.crypto.yf.download", return_value=_mock_df())
    result = CryptoFetcher().fetch("BTC-USD", days=5)
    assert result.market_type == "crypto"
    assert result.source == "yfinance_crypto"


# ── CommodityFetcher ──────────────────────────────────────────────────────────

def test_commodity_fetcher_supports_futures():
    assert CommodityFetcher().supports("GC=F") is True


def test_commodity_fetcher_does_not_support_equities():
    assert CommodityFetcher().supports("AAPL") is False


def test_commodity_fetcher_returns_market_data(mocker):
    mocker.patch("data.fetchers.commodities.yf.download", return_value=_mock_df())
    result = CommodityFetcher().fetch("GC=F", days=5)
    assert result.market_type == "commodity"
    assert result.source == "yfinance_futures"
