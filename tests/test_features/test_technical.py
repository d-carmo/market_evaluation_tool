import numpy as np
import pandas as pd
import pytest

from features.technical import (
    add_momentum_indicators,
    add_trend_indicators,
    add_volatility_indicators,
    add_volume_indicators,
)


def test_trend_indicator_columns(sample_df):
    out = add_trend_indicators(sample_df)
    expected = {"sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
                "macd", "macd_signal", "macd_hist", "adx_14"}
    assert expected.issubset(set(out.columns))


def test_momentum_indicator_columns(sample_df):
    out = add_momentum_indicators(sample_df)
    assert {"rsi_14", "stoch_k", "stoch_d", "williams_r"}.issubset(set(out.columns))


def test_rsi_range(sample_df):
    out = add_momentum_indicators(sample_df)
    valid = out["rsi_14"].dropna()
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 100).all()


def test_volatility_indicator_columns(sample_df):
    out = add_volatility_indicators(sample_df)
    assert {"bb_upper", "bb_mid", "bb_lower", "atr_14"}.issubset(set(out.columns))


def test_bb_ordering(sample_df):
    out = add_volatility_indicators(sample_df)
    valid = out[["bb_lower", "bb_mid", "bb_upper"]].dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_mid"] >= valid["bb_lower"]).all()


def test_atr_nonnegative(sample_df):
    out = add_volatility_indicators(sample_df)
    assert (out["atr_14"].dropna() >= 0).all()


def test_volume_indicator_columns(sample_df):
    out = add_volume_indicators(sample_df)
    assert {"obv", "vwap", "mfi_14"}.issubset(set(out.columns))


def test_volume_graceful_zero(sample_df):
    df_zero_vol = sample_df.copy()
    df_zero_vol["volume"] = 0.0
    out = add_volume_indicators(df_zero_vol)
    assert out["obv"].isna().all()
    assert out["vwap"].isna().all()
    assert out["mfi_14"].isna().all()


def test_no_mutation_trend(sample_df):
    cols_before = list(sample_df.columns)
    add_trend_indicators(sample_df)
    assert list(sample_df.columns) == cols_before


def test_no_mutation_momentum(sample_df):
    cols_before = list(sample_df.columns)
    add_momentum_indicators(sample_df)
    assert list(sample_df.columns) == cols_before


def test_no_mutation_volatility(sample_df):
    cols_before = list(sample_df.columns)
    add_volatility_indicators(sample_df)
    assert list(sample_df.columns) == cols_before
