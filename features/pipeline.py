from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from data.exceptions import FeatureError
from data.models import EnrichedData, MarketData
from features.statistical import add_autocorrelation, add_return_features, add_rolling_stats
from features.technical import (
    add_momentum_indicators,
    add_trend_indicators,
    add_volatility_indicators,
    add_volume_indicators,
)

logger = logging.getLogger(__name__)

_MIN_ROWS = 60


def build_features(data: MarketData) -> EnrichedData:
    """Run all feature functions in sequence and return EnrichedData.

    Raises FeatureError if data.df has fewer than 60 rows.
    """
    if len(data.df) < _MIN_ROWS:
        raise FeatureError(
            f"{data.ticker}: insufficient data ({len(data.df)} rows, need ≥{_MIN_ROWS})"
        )

    original_cols = set(data.df.columns)
    df = data.df.copy()

    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    df = add_return_features(df)
    df = add_rolling_stats(df, windows=[20])
    df = add_autocorrelation(df, lags=[1, 5])

    feature_names = [c for c in df.columns if c not in original_cols]
    return EnrichedData(
        ticker=data.ticker,
        market_type=data.market_type,
        df=df,
        feature_names=feature_names,
    )


_SENTIMENT_COLS = (
    "sentiment_score",
    "sentiment_velocity",
    "bullish_ratio",
    "engagement_score",
)


def augment_with_sentiment(enriched: EnrichedData, snapshot) -> EnrichedData:
    """Inject sentiment columns into *only the last row* of EnrichedData.df.

    All other rows receive 0.0.  When snapshot is None the columns are still
    added (filled with 0.0) so that XGBoost feature vectors remain consistent.

    Args:
        enriched: Output of build_features().
        snapshot: SentimentSnapshot | None.

    Returns:
        New EnrichedData with 4 extra columns appended.
    """
    df = enriched.df.copy()

    for col in _SENTIMENT_COLS:
        df[col] = 0.0

    if snapshot is not None:
        idx = df.index[-1]
        df.loc[idx, "sentiment_score"]    = float(snapshot.sentiment_score)
        df.loc[idx, "sentiment_velocity"] = float(snapshot.sentiment_velocity)
        df.loc[idx, "bullish_ratio"]      = float(snapshot.bullish_ratio)
        df.loc[idx, "engagement_score"]   = float(snapshot.engagement_score)

    new_feature_names = list(enriched.feature_names) + [
        c for c in _SENTIMENT_COLS if c not in enriched.feature_names
    ]
    return EnrichedData(
        ticker=enriched.ticker,
        market_type=enriched.market_type,
        df=df,
        feature_names=new_feature_names,
    )


def build_features_all(
    market_data: dict[str, MarketData],
    max_workers: int = 4,
) -> dict[str, EnrichedData]:
    """Run build_features for all tickers in parallel (ProcessPoolExecutor).

    Failed tickers are logged and excluded from the result.
    """
    results: dict[str, EnrichedData] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(build_features, data): ticker
            for ticker, data in market_data.items()
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                logger.warning("Feature engineering failed for %s: %s", ticker, exc)
    return results
