import logging

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend indicators: SMA, EMA, MACD, ADX.

    Returns a new DataFrame — never mutates input.
    """
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]

    result["sma_20"] = ta.sma(close, length=20)
    result["sma_50"] = ta.sma(close, length=50)
    result["sma_200"] = ta.sma(close, length=200)
    result["ema_12"] = ta.ema(close, length=12)
    result["ema_26"] = ta.ema(close, length=26)

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        result["macd"] = macd_df.iloc[:, 0]          # MACD_12_26_9
        result["macd_signal"] = macd_df.iloc[:, 2]   # MACDs_12_26_9
        result["macd_hist"] = macd_df.iloc[:, 1]     # MACDh_12_26_9
    else:
        result["macd"] = float("nan")
        result["macd_signal"] = float("nan")
        result["macd_hist"] = float("nan")

    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None and not adx_df.empty:
        # ADX_14 column
        adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
        result["adx_14"] = adx_df[adx_col[0]] if adx_col else float("nan")
    else:
        result["adx_14"] = float("nan")

    return result


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators: RSI, Stochastic, Williams %R."""
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]

    result["rsi_14"] = ta.rsi(close, length=14)

    stoch_df = ta.stoch(high, low, close, k=14, d=3)
    if stoch_df is not None and not stoch_df.empty:
        k_col = [c for c in stoch_df.columns if c.startswith("STOCHk_")]
        d_col = [c for c in stoch_df.columns if c.startswith("STOCHd_")]
        result["stoch_k"] = stoch_df[k_col[0]] if k_col else float("nan")
        result["stoch_d"] = stoch_df[d_col[0]] if d_col else float("nan")
    else:
        result["stoch_k"] = float("nan")
        result["stoch_d"] = float("nan")

    willr = ta.willr(high, low, close, length=14)
    result["williams_r"] = willr if willr is not None else float("nan")

    return result


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators: Bollinger Bands, ATR."""
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]

    bb_df = ta.bbands(close, length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        upper_col = [c for c in bb_df.columns if c.startswith("BBU_")]
        mid_col = [c for c in bb_df.columns if c.startswith("BBM_")]
        lower_col = [c for c in bb_df.columns if c.startswith("BBL_")]
        result["bb_upper"] = bb_df[upper_col[0]] if upper_col else float("nan")
        result["bb_mid"] = bb_df[mid_col[0]] if mid_col else float("nan")
        result["bb_lower"] = bb_df[lower_col[0]] if lower_col else float("nan")
    else:
        result["bb_upper"] = float("nan")
        result["bb_mid"] = float("nan")
        result["bb_lower"] = float("nan")

    atr = ta.atr(high, low, close, length=14)
    result["atr_14"] = atr if atr is not None else float("nan")

    return result


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume indicators: OBV, VWAP, MFI.

    If volume is all zeros, fills obv/vwap/mfi_14 with NaN (graceful degradation).
    """
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]

    if volume.sum() == 0:
        logger.warning("Volume is all zeros — skipping volume indicators.")
        result["obv"] = float("nan")
        result["vwap"] = float("nan")
        result["mfi_14"] = float("nan")
        return result

    obv = ta.obv(close, volume)
    result["obv"] = obv if obv is not None else float("nan")

    vwap = ta.vwap(high, low, close, volume)
    if vwap is not None:
        # pandas_ta may return DataFrame or Series
        result["vwap"] = vwap.iloc[:, 0] if isinstance(vwap, pd.DataFrame) else vwap
    else:
        result["vwap"] = float("nan")

    mfi = ta.mfi(high, low, close, volume, length=14)
    result["mfi_14"] = mfi if mfi is not None else float("nan")

    return result
