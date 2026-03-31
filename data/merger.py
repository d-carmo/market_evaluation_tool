"""Multi-source OHLCV merger.

When multiple data sources are available for the same ticker, this module
combines them into a single, higher-coverage dataset:

  - Bloomberg values take precedence on dates where both sources have data
    (Bloomberg adjustments are more accurate).
  - yfinance values fill any dates that Bloomberg is missing or returned NaN for.
  - The merged result has a continuous DatetimeIndex spanning both sources.

The source field is set to "bloomberg+yfinance" so downstream components
can distinguish merged data from single-source data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from data.models import MarketData

logger = logging.getLogger(__name__)

_OHLCV = ["open", "high", "low", "close", "volume"]


def merge_sources(primary: MarketData, secondary: MarketData) -> MarketData:
    """Return a new MarketData that combines *primary* and *secondary*.

    *primary* values win on overlapping dates.  *secondary* fills any gaps
    (missing dates or NaN values) in *primary*.

    Args:
        primary:   Higher-quality source (e.g. Bloomberg).
        secondary: Fallback / gap-filling source (e.g. yfinance).

    Returns:
        A new MarketData with source="<primary.source>+<secondary.source>",
        covering the full date range of both inputs.
    """
    # Align columns — both DataFrames are already normalized to OHLCV
    p = primary.df.reindex(columns=_OHLCV)
    s = secondary.df.reindex(columns=_OHLCV)

    # combine_first: keeps primary values where not-NaN, fills from secondary
    merged = p.combine_first(s)

    # Ensure volume is never NaN after merge
    merged["volume"] = merged["volume"].fillna(0.0)

    # Drop any rows where close is still NaN (shouldn't happen, but be safe)
    merged = merged.dropna(subset=["close"]).sort_index()

    primary_rows = len(p)
    secondary_rows = len(s)
    merged_rows = len(merged)
    gap_filled = merged_rows - primary_rows
    if gap_filled > 0:
        logger.info(
            "%s: merged %d bloomberg rows + %d yfinance rows → %d rows "
            "(%d gap-filled from yfinance)",
            primary.ticker, primary_rows, secondary_rows, merged_rows, gap_filled,
        )
    else:
        logger.debug(
            "%s: merged %d bloomberg + %d yfinance → %d rows (no gaps)",
            primary.ticker, primary_rows, secondary_rows, merged_rows,
        )

    source_label = f"{primary.source}+{secondary.source}"
    return MarketData(
        ticker=primary.ticker,
        market_type=primary.market_type,
        fetched_at=datetime.now(timezone.utc),
        source=source_label,
        df=merged,
    )
