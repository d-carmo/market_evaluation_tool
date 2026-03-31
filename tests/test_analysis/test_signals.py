import numpy as np
import pandas as pd
import pytest

from analysis.signals import (
    bb_position_signal,
    macd_cross_signal,
    rsi_signal,
    stoch_signal,
    volume_trend_signal,
)
from data.models import EnrichedData
from tests.test_analysis.conftest import make_enriched


def _patch_col(data: EnrichedData, col: str, value) -> EnrichedData:
    """Return new EnrichedData with a column overwritten to a scalar value."""
    df = data.df.copy()
    df[col] = value
    return EnrichedData(ticker=data.ticker, market_type=data.market_type,
                        df=df, feature_names=data.feature_names)


def test_rsi_buy_signal(bullish_data):
    d = _patch_col(bullish_data, "rsi_14", 25.0)
    sig = rsi_signal(d, weight=1.0)
    assert sig.value == 1


def test_rsi_sell_signal(bullish_data):
    d = _patch_col(bullish_data, "rsi_14", 75.0)
    sig = rsi_signal(d, weight=1.0)
    assert sig.value == -1


def test_rsi_hold_signal(bullish_data):
    d = _patch_col(bullish_data, "rsi_14", 50.0)
    sig = rsi_signal(d, weight=1.0)
    assert sig.value == 0


def test_macd_cross_buy(bullish_data):
    df = bullish_data.df.copy()
    df["macd_hist"] = [-0.1] * (len(df) - 1) + [0.1]
    d = EnrichedData(ticker=bullish_data.ticker, market_type=bullish_data.market_type,
                     df=df, feature_names=bullish_data.feature_names)
    sig = macd_cross_signal(d, weight=1.5)
    assert sig.value == 1


def test_macd_cross_sell(bullish_data):
    df = bullish_data.df.copy()
    df["macd_hist"] = [0.1] * (len(df) - 1) + [-0.1]
    d = EnrichedData(ticker=bullish_data.ticker, market_type=bullish_data.market_type,
                     df=df, feature_names=bullish_data.feature_names)
    sig = macd_cross_signal(d, weight=1.5)
    assert sig.value == -1


def test_bb_buy_signal(bullish_data):
    df = bullish_data.df.copy()
    df["close"] = df["bb_lower"] - 1.0
    d = EnrichedData(ticker=bullish_data.ticker, market_type=bullish_data.market_type,
                     df=df, feature_names=bullish_data.feature_names)
    sig = bb_position_signal(d, weight=1.0)
    assert sig.value == 1


def test_bb_sell_signal(bullish_data):
    df = bullish_data.df.copy()
    df["close"] = df["bb_upper"] + 1.0
    d = EnrichedData(ticker=bullish_data.ticker, market_type=bullish_data.market_type,
                     df=df, feature_names=bullish_data.feature_names)
    sig = bb_position_signal(d, weight=1.0)
    assert sig.value == -1


def test_volume_hold_on_nan_obv(bullish_data):
    d = _patch_col(bullish_data, "obv", float("nan"))
    sig = volume_trend_signal(d, weight=0.8)
    assert sig.value == 0


def test_stoch_buy_signal(bullish_data):
    df = bullish_data.df.copy()
    df["stoch_k"] = 15.0
    df["stoch_d"] = 10.0
    d = EnrichedData(ticker=bullish_data.ticker, market_type=bullish_data.market_type,
                     df=df, feature_names=bullish_data.feature_names)
    sig = stoch_signal(d, weight=0.8)
    assert sig.value == 1


def test_stoch_sell_signal(bullish_data):
    df = bullish_data.df.copy()
    df["stoch_k"] = 85.0
    df["stoch_d"] = 90.0
    d = EnrichedData(ticker=bullish_data.ticker, market_type=bullish_data.market_type,
                     df=df, feature_names=bullish_data.feature_names)
    sig = stoch_signal(d, weight=0.8)
    assert sig.value == -1
