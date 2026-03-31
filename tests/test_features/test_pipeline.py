from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data.exceptions import FeatureError
from data.models import EnrichedData, MarketData
from features.pipeline import build_features, build_features_all

_EXPECTED_FEATURES = {
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist", "adx_14",
    "rsi_14", "stoch_k", "stoch_d", "williams_r",
    "bb_upper", "bb_mid", "bb_lower", "atr_14",
    "obv", "vwap", "mfi_14",
    "returns_1d", "returns_5d", "log_returns",
    "rolling_mean_20", "rolling_std_20", "rolling_skew_20", "rolling_kurt_20",
    "z_score_20", "realized_vol_20",
    "autocorr_lag1", "autocorr_lag5",
}


def test_build_features_returns_enriched_data(sample_market_data):
    result = build_features(sample_market_data)
    assert isinstance(result, EnrichedData)
    assert _EXPECTED_FEATURES.issubset(set(result.df.columns))


def test_build_features_raises_on_short_df():
    idx = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": [1.0]*30, "high": [1.0]*30, "low": [1.0]*30,
        "close": [1.0]*30, "volume": [0.0]*30,
    }, index=idx)
    data = MarketData("X", "stock", datetime.now(timezone.utc), "test", df)
    with pytest.raises(FeatureError):
        build_features(data)


def test_build_features_feature_names_populated(sample_market_data):
    result = build_features(sample_market_data)
    assert len(result.feature_names) > 0


def test_build_features_original_cols_preserved(sample_market_data):
    result = build_features(sample_market_data)
    assert {"open", "high", "low", "close", "volume"}.issubset(set(result.df.columns))


def test_build_features_all_returns_both(sample_market_data):
    market_data = {"TEST": sample_market_data, "TEST2": sample_market_data}
    results = build_features_all(market_data, max_workers=2)
    assert set(results.keys()) == {"TEST", "TEST2"}
