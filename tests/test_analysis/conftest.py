from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data.models import EnrichedData


def make_enriched(n: int = 100, close_trend: str = "up", ticker: str = "TEST") -> EnrichedData:
    """Build synthetic EnrichedData with all required feature columns.

    close_trend: "up" (rising), "down" (falling), "flat" (sideways)
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")

    if close_trend == "up":
        close = np.linspace(100.0, 150.0, n)
    elif close_trend == "down":
        close = np.linspace(150.0, 100.0, n)
    else:
        close = 100.0 + rng.normal(0, 0.3, n)

    df = pd.DataFrame(index=dates)
    df["open"] = close * 0.999
    df["high"] = close * 1.01
    df["low"] = close * 0.99
    df["close"] = close
    df["volume"] = 1e6

    # Trend
    df["sma_20"] = pd.Series(close).rolling(20, min_periods=1).mean().values
    df["sma_50"] = pd.Series(close).rolling(50, min_periods=1).mean().values
    df["sma_200"] = pd.Series(close).rolling(200, min_periods=1).mean().values
    df["ema_12"] = pd.Series(close).ewm(span=12).mean().values
    df["ema_26"] = pd.Series(close).ewm(span=26).mean().values
    df["adx_14"] = np.where(close_trend != "flat", 30.0, 12.0)
    macd_line = pd.Series(close).ewm(span=12).mean() - pd.Series(close).ewm(span=26).mean()
    macd_signal = macd_line.ewm(span=9).mean()
    df["macd"] = macd_line.values
    df["macd_signal"] = macd_signal.values
    df["macd_hist"] = (macd_line - macd_signal).values

    # Momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50).values
    df["stoch_k"] = np.where(close_trend == "up", 25.0, 75.0)
    df["stoch_d"] = np.where(close_trend == "up", 20.0, 80.0)
    df["williams_r"] = -50.0

    # Volatility
    rolling_mean = pd.Series(close).rolling(20, min_periods=1).mean()
    rolling_std = pd.Series(close).rolling(20, min_periods=2).std().fillna(1.0)
    df["bb_upper"] = (rolling_mean + 2 * rolling_std).values
    df["bb_mid"] = rolling_mean.values
    df["bb_lower"] = (rolling_mean - 2 * rolling_std).values
    df["atr_14"] = np.abs(pd.Series(close).diff()).rolling(14, min_periods=1).mean().values

    # Volume
    df["obv"] = np.cumsum(np.sign(pd.Series(close).diff().fillna(0).values) * 1e6)
    df["vwap"] = close
    df["mfi_14"] = 50.0

    # Statistical
    df["returns_1d"] = pd.Series(close).pct_change(1).values
    df["returns_5d"] = pd.Series(close).pct_change(5).values
    ratio = pd.Series(close) / pd.Series(close).shift(1)
    df["log_returns"] = np.log(ratio.clip(lower=1e-10)).values
    df["rolling_mean_20"] = rolling_mean.values
    df["rolling_std_20"] = rolling_std.values
    df["rolling_skew_20"] = pd.Series(close).pct_change().rolling(20, min_periods=2).skew().values
    df["rolling_kurt_20"] = pd.Series(close).pct_change().rolling(20, min_periods=4).kurt().values
    df["z_score_20"] = ((pd.Series(close) - rolling_mean) / rolling_std).values
    log_ret = np.log((pd.Series(close) / pd.Series(close).shift(1)).clip(lower=1e-10))
    df["realized_vol_20"] = log_ret.rolling(20, min_periods=2).std().values * np.sqrt(252)
    df["autocorr_lag1"] = 0.1
    df["autocorr_lag5"] = 0.05

    # Sentiment features (zero-filled; updated by augment_with_sentiment at runtime)
    df["sentiment_score"]    = 0.0
    df["sentiment_velocity"] = 0.0
    df["bullish_ratio"]      = 0.0
    df["engagement_score"]   = 0.0

    feature_names = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    return EnrichedData(ticker=ticker, market_type="stock", df=df, feature_names=feature_names)


@pytest.fixture
def bullish_data() -> EnrichedData:
    return make_enriched(close_trend="up")


@pytest.fixture
def bearish_data() -> EnrichedData:
    return make_enriched(close_trend="down")


@pytest.fixture
def sideways_data() -> EnrichedData:
    return make_enriched(close_trend="flat")
