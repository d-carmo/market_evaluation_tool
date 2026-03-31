from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data.models import MarketData


@pytest.fixture
def sample_df():
    """300-row synthetic OHLCV DataFrame with realistic random-walk prices."""
    n = 300
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close + rng.normal(0, 0.2, n)
    volume = np.abs(rng.normal(5e6, 1e6, n))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df.astype(float)


@pytest.fixture
def sample_market_data(sample_df):
    return MarketData(
        ticker="TEST",
        market_type="stock",
        fetched_at=datetime.now(timezone.utc),
        source="test",
        df=sample_df,
    )
