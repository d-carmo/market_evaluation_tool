from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from data.cache import DataCache
from data.models import MarketData


def _make_market_data(ticker: str = "AAPL") -> MarketData:
    idx = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": [1.0]*10, "high": [2.0]*10, "low": [0.5]*10,
        "close": [1.5]*10, "volume": [1000.0]*10,
    }, index=idx)
    return MarketData(
        ticker=ticker,
        market_type="stock",
        fetched_at=datetime.now(timezone.utc),
        source="test",
        df=df,
    )


def test_cache_miss_returns_none(tmp_path):
    cache = DataCache(tmp_path, ttl_hours=6)
    assert cache.get("MISSING") is None


def test_cache_set_get_roundtrip(tmp_path):
    cache = DataCache(tmp_path, ttl_hours=6)
    data = _make_market_data("AAPL")
    cache.set(data)
    result = cache.get("AAPL")
    assert result is not None
    assert result.ticker == "AAPL"
    assert result.market_type == "stock"
    assert len(result.df) == 10


def test_cache_is_valid_within_ttl(tmp_path):
    cache = DataCache(tmp_path, ttl_hours=6)
    cache.set(_make_market_data("AAPL"))
    assert cache.is_valid("AAPL") is True


def test_cache_is_invalid_after_ttl(tmp_path):
    cache = DataCache(tmp_path, ttl_hours=6)
    cache.set(_make_market_data("AAPL"))
    future = datetime.now(timezone.utc) + timedelta(hours=7)
    with patch("data.cache.datetime") as mock_dt:
        mock_dt.now.return_value = future
        mock_dt.fromisoformat = datetime.fromisoformat
        assert cache.is_valid("AAPL") is False


def test_cache_invalidate_single(tmp_path):
    cache = DataCache(tmp_path, ttl_hours=6)
    cache.set(_make_market_data("AAPL"))
    cache.set(_make_market_data("MSFT"))
    cache.invalidate("AAPL")
    assert cache.is_valid("AAPL") is False
    assert cache.is_valid("MSFT") is True


def test_cache_invalidate_all(tmp_path):
    cache = DataCache(tmp_path, ttl_hours=6)
    cache.set(_make_market_data("AAPL"))
    cache.set(_make_market_data("MSFT"))
    cache.invalidate()
    assert cache.is_valid("AAPL") is False
    assert cache.is_valid("MSFT") is False
