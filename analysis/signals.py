from __future__ import annotations

import logging

import numpy as np

from data.models import EnrichedData, Signal, SignalDir

logger = logging.getLogger(__name__)


def _last_valid(df, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _last_two(df, col: str) -> tuple[float, float] | None:
    """Return (second-to-last, last) non-NaN values, or None if insufficient."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if len(s) < 2:
        return None
    return float(s.iloc[-2]), float(s.iloc[-1])


def rsi_signal(data: EnrichedData, weight: float) -> Signal:
    """RSI < 30 → BUY, > 70 → SELL, else HOLD."""
    rsi = _last_valid(data.df, "rsi_14")
    if rsi is None:
        return Signal("rsi", 0, 0.0, weight)
    if rsi < 30:
        return Signal("rsi", 1, rsi, weight)
    if rsi > 70:
        return Signal("rsi", -1, rsi, weight)
    return Signal("rsi", 0, rsi, weight)


def macd_cross_signal(data: EnrichedData, weight: float) -> Signal:
    """MACD histogram sign change: negative→positive = BUY, positive→negative = SELL."""
    pair = _last_two(data.df, "macd_hist")
    if pair is None:
        return Signal("macd_cross", 0, 0.0, weight)
    prev, curr = pair
    raw = curr
    if curr > 0 and prev <= 0:
        return Signal("macd_cross", 1, raw, weight)
    if curr < 0 and prev >= 0:
        return Signal("macd_cross", -1, raw, weight)
    return Signal("macd_cross", 0, raw, weight)


def bb_position_signal(data: EnrichedData, weight: float) -> Signal:
    """Price below lower band → BUY; above upper band → SELL."""
    df = data.df
    close = _last_valid(df, "close")
    bb_lower = _last_valid(df, "bb_lower")
    bb_mid = _last_valid(df, "bb_mid")
    bb_upper = _last_valid(df, "bb_upper")

    if any(v is None for v in (close, bb_lower, bb_mid, bb_upper)):
        return Signal("bb_position", 0, 0.0, weight)

    denom = bb_upper - bb_lower
    raw = (close - bb_mid) / denom if denom != 0 else 0.0

    if close < bb_lower:
        return Signal("bb_position", 1, raw, weight)
    if close > bb_upper:
        return Signal("bb_position", -1, raw, weight)
    return Signal("bb_position", 0, raw, weight)


def ma_cross_signal(data: EnrichedData, weight: float) -> Signal:
    """EMA12 crossing SMA50: above → BUY, below → SELL."""
    ema_pair = _last_two(data.df, "ema_12")
    sma_pair = _last_two(data.df, "sma_50")
    if ema_pair is None or sma_pair is None:
        return Signal("ma_cross", 0, 0.0, weight)

    ema_prev, ema_curr = ema_pair
    sma_prev, sma_curr = sma_pair
    raw = (ema_curr - sma_curr) / sma_curr if sma_curr != 0 else 0.0

    if ema_curr > sma_curr and ema_prev <= sma_prev:
        return Signal("ma_cross", 1, raw, weight)
    if ema_curr < sma_curr and ema_prev >= sma_prev:
        return Signal("ma_cross", -1, raw, weight)
    return Signal("ma_cross", 0, raw, weight)


def volume_trend_signal(data: EnrichedData, weight: float) -> Signal:
    """Rising OBV + positive return → BUY; falling OBV + negative return → SELL."""
    df = data.df
    if "obv" not in df.columns or df["obv"].isna().all():
        return Signal("volume_trend", 0, 0.0, weight)

    obv_series = df["obv"].dropna()
    if len(obv_series) < 5:
        return Signal("volume_trend", 0, float(obv_series.iloc[-1]) if not obv_series.empty else 0.0, weight)

    last_obv = float(obv_series.iloc[-1])
    obv_ma5 = float(obv_series.iloc[-5:].mean())
    ret = _last_valid(df, "returns_1d")
    if ret is None:
        return Signal("volume_trend", 0, last_obv, weight)

    if last_obv > obv_ma5 and ret > 0:
        return Signal("volume_trend", 1, last_obv, weight)
    if last_obv < obv_ma5 and ret < 0:
        return Signal("volume_trend", -1, last_obv, weight)
    return Signal("volume_trend", 0, last_obv, weight)


def stoch_signal(data: EnrichedData, weight: float) -> Signal:
    """Stochastic oversold cross-up → BUY; overbought cross-down → SELL."""
    stoch_k = _last_valid(data.df, "stoch_k")
    stoch_d = _last_valid(data.df, "stoch_d")
    if stoch_k is None or stoch_d is None:
        return Signal("stoch", 0, 0.0, weight)
    if stoch_k < 20 and stoch_k > stoch_d:
        return Signal("stoch", 1, stoch_k, weight)
    if stoch_k > 80 and stoch_k < stoch_d:
        return Signal("stoch", -1, stoch_k, weight)
    return Signal("stoch", 0, stoch_k, weight)


def sentiment_signal(snapshot, weight: float) -> Signal:
    """Social-media sentiment: bullish → BUY, bearish → SELL, else HOLD.

    snapshot: SentimentSnapshot | None — if None or too few mentions, returns HOLD.
    Thresholds: score > 0.3 → BUY, score < -0.3 → SELL.
    """
    if snapshot is None or snapshot.mention_count < 5:
        return Signal("sentiment", 0, 0.0, weight)
    score = float(snapshot.sentiment_score)
    if score > 0.3:
        return Signal("sentiment", 1, score, weight)
    if score < -0.3:
        return Signal("sentiment", -1, score, weight)
    return Signal("sentiment", 0, score, weight)


def compute_signals(data: EnrichedData, weights, snapshot=None) -> list[Signal]:
    """Run all signal functions and return the full list.

    Args:
        data:     Enriched ticker data.
        weights:  SignalWeights dataclass (rsi, macd, trend, volume, bb, stoch, sentiment).
        snapshot: Optional SentimentSnapshot for the 7th signal.
    """
    signals = [
        rsi_signal(data, weights.rsi),
        macd_cross_signal(data, weights.macd),
        bb_position_signal(data, weights.bb),
        ma_cross_signal(data, weights.trend),
        volume_trend_signal(data, weights.volume),
        stoch_signal(data, weights.stoch),
        sentiment_signal(snapshot, getattr(weights, "sentiment", 0.5)),
    ]
    return signals
