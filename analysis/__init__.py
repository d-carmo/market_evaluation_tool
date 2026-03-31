import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from sqlalchemy.orm import Session

from config import SignalWeights
from data.models import CompositeScore, EnrichedData
from analysis.scoring import build_composite
from analysis.signals import compute_signals
from analysis.trend import detect_trend

logger = logging.getLogger(__name__)


def get_ticker_weights(
    ticker: str,
    db: Session,
    global_weights: SignalWeights,
) -> SignalWeights:
    """Return learned per-ticker weights from DB, or global_weights as fallback."""
    from storage.repository import TickerWeightsRepository
    row = TickerWeightsRepository(db).get(ticker)
    if row is None:
        return global_weights
    return SignalWeights(
        rsi=row.rsi, macd=row.macd, trend=row.trend,
        volume=row.volume, bb=row.bb, stoch=row.stoch,
        sentiment=getattr(row, "sentiment", 0.5),
    )


def analyze(data: EnrichedData, weights, snapshot=None) -> CompositeScore:
    """Run detect_trend → compute_signals → build_composite for one ticker.

    Args:
        data:     Enriched ticker data.
        weights:  SignalWeights (global or per-ticker learned).
        snapshot: Optional SentimentSnapshot for the 7th signal.
    """
    trend = detect_trend(data)
    signals = compute_signals(data, weights, snapshot=snapshot)
    return build_composite(data.ticker, signals, trend)


def analyze_all(
    enriched: dict[str, EnrichedData],
    weights,
    max_workers: int = 4,
) -> dict[str, CompositeScore]:
    """Run analyze for all tickers in parallel (ProcessPoolExecutor).

    Failed tickers are logged and excluded from the result.
    """
    results: dict[str, CompositeScore] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze, data, weights): ticker
            for ticker, data in enriched.items()
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                logger.warning("Analysis failed for %s: %s", ticker, exc)
    return results
