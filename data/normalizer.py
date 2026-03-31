import logging

import pandas as pd

from data.exceptions import NormalizationError

logger = logging.getLogger(__name__)

# Map source-specific column names → canonical names.
# Lower-cased source names are checked first; originals as fallback.
_COLUMN_MAP: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "volume": "volume",
    # yfinance capitalized variants
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "close",
    "Volume": "volume",
}

_REQUIRED = {"open", "high", "low", "close", "volume"}


def normalize(df: pd.DataFrame, source: str, ticker: str) -> pd.DataFrame:
    """Standardize a raw source DataFrame into the MarketData.df contract.

    Steps:
    1. Rename columns using _COLUMN_MAP (case-insensitive fallback).
    2. Verify 'close' exists; raise NormalizationError otherwise.
    3. Add missing optional columns (open/high/low/volume) with sensible defaults.
    4. Cast OHLCV columns to float64.
    5. Fill volume NaN with 0.0.
    6. Drop rows where close is NaN.
    7. Set DatetimeIndex, convert to UTC, sort ascending.
    8. Return a new DataFrame — never mutate the input.
    """
    result = df.copy()

    # Flatten MultiIndex columns (yfinance sometimes returns these)
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [" ".join(str(c) for c in col).strip() for col in result.columns]

    # Rename known columns
    rename_map = {}
    for col in list(result.columns):
        canonical = _COLUMN_MAP.get(col) or _COLUMN_MAP.get(col.lower())
        if canonical and col not in rename_map.values():
            rename_map[col] = canonical

    result = result.rename(columns=rename_map)

    # Remove duplicate columns that map to the same canonical name
    # (e.g. both 'Close' and 'Adj Close' → 'close'; keep last)
    result = result.loc[:, ~result.columns.duplicated(keep="last")]

    if "close" not in result.columns:
        raise NormalizationError(
            f"[{ticker}] Cannot find 'close' column in source '{source}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Fill missing optional columns
    for col in ("open", "high", "low"):
        if col not in result.columns:
            logger.warning("[%s] Missing column '%s'; filling from close.", ticker, col)
            result[col] = result["close"]
    if "volume" not in result.columns:
        logger.warning("[%s] Missing 'volume' column; filling with 0.", ticker)
        result["volume"] = 0.0

    # Cast to float64
    for col in _REQUIRED:
        result[col] = result[col].astype("float64")

    result["volume"] = result["volume"].fillna(0.0)
    result = result.dropna(subset=["close"])

    # Normalize index to UTC DatetimeIndex ascending
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")
    result = result.sort_index()

    # Keep only canonical columns in standard order
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in result.columns]
    return result[cols]
