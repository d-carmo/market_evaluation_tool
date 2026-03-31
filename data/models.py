from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

MarketType = Literal["stock", "crypto", "commodity"]
Direction = Literal["BULLISH", "BEARISH", "SIDEWAYS"]
Action = Literal["BUY", "HOLD", "SELL"]
SignalDir = Literal[-1, 0, 1]
PriceDir = Literal["UP", "DOWN", "FLAT"]


@dataclass(frozen=True)
class MarketData:
    """Normalized OHLCV data from any fetcher.

    df column contract (DatetimeIndex ascending UTC, all float64):
        open, high, low, close, volume  (volume=0.0 if unavailable, never NaN)
    """

    ticker: str
    market_type: MarketType
    fetched_at: datetime
    source: str
    df: pd.DataFrame


@dataclass(frozen=True)
class EnrichedData:
    """MarketData with all computed features appended.

    Guaranteed feature columns (NaN only in warm-up rows, never in last 30):
        sma_20, sma_50, sma_200, ema_12, ema_26
        macd, macd_signal, macd_hist
        rsi_14, stoch_k, stoch_d, williams_r
        bb_upper, bb_mid, bb_lower, atr_14
        adx_14, obv, vwap, mfi_14
        returns_1d, returns_5d, log_returns
        rolling_mean_20, rolling_std_20, rolling_skew_20, rolling_kurt_20
        z_score_20, realized_vol_20
        autocorr_lag1, autocorr_lag5
    """

    ticker: str
    market_type: MarketType
    df: pd.DataFrame
    feature_names: list[str]


@dataclass(frozen=True)
class TrendResult:
    """Current market regime classification."""

    ticker: str
    direction: Direction
    strength: float  # ADX-derived, normalized 0.0–1.0
    ma_aligned: bool  # True when MAs stack consistently with direction
    slope: float  # linear regression slope of last 20 closes, normalized by price
    adx: float  # raw ADX value


@dataclass(frozen=True)
class Signal:
    """Single indicator buy/sell/hold verdict."""

    name: str
    value: SignalDir  # -1=sell, 0=hold, 1=buy
    raw: float  # indicator value that produced this signal
    weight: float  # from config


@dataclass(frozen=True)
class CompositeScore:
    """Weighted ensemble of signals — primary output of the analysis layer."""

    ticker: str
    score: float  # weighted sum normalized to [-1.0, 1.0]
    action: Action
    confidence: float  # 0.0–1.0; fraction of signals agreeing with action
    signals: list[Signal]
    trend: TrendResult


@dataclass(frozen=True)
class PriceForecast:
    """Directional price forecast for a single horizon."""

    horizon_days: int
    low: float
    mid: float
    high: float
    direction: PriceDir
    confidence: float  # 0.0–1.0


@dataclass(frozen=True)
class PredictionResult:
    """Full prediction output across configured horizons."""

    ticker: str
    current_price: float
    short_forecast: PriceForecast
    long_forecast: PriceForecast
    model: str  # "statistical" | "ml"


@dataclass(frozen=True)
class TickerReport:
    """Complete per-ticker output. Input to the output layer."""

    ticker: str
    market_type: MarketType
    score: CompositeScore
    prediction: PredictionResult
    generated_at: datetime


@dataclass(frozen=True)
class SentimentSnapshot:
    """Aggregated social-media sentiment for a ticker over a time window."""

    ticker: str
    window_hours: int          # look-back window used to gather posts
    mention_count: int         # number of posts / comments found
    sentiment_score: float     # weighted average sentiment in [-1.0, 1.0]
    sentiment_velocity: float  # change vs. previous window (can be 0 when unavailable)
    bullish_ratio: float       # fraction of clearly bullish texts (0.0–1.0)
    engagement_score: float    # normalised engagement (upvotes × comments proxy)
    fetched_at: datetime


# ── Storage domain objects (pure Python, not ORM) ────────────────────────────

@dataclass(frozen=True)
class StoredPrediction:
    """Domain object for a row in the predictions table."""

    id: int
    ticker: str
    generated_at: datetime
    current_price: float
    short_horizon: int
    short_mid: float
    short_direction: PriceDir
    long_horizon: int
    long_mid: float
    long_direction: PriceDir
    model_used: str


@dataclass(frozen=True)
class AccuracyRecord:
    """Domain object for a row in the accuracy_records table."""

    id: int
    prediction_id: int
    ticker: str
    horizon_days: int
    evaluated_at: datetime
    actual_price: float
    predicted_direction: PriceDir
    actual_direction: PriceDir
    direction_correct: bool
    predicted_mid: float
    price_error_pct: float
