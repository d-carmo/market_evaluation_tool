import logging
from datetime import datetime, timezone
from pathlib import Path

from config import AppConfig, load_config
from data import fetch_all
from data.cache import DataCache
from data.models import TickerReport
from features import build_features_all
from analysis import analyze_all
from prediction import get_predictor
from output import render_all
from storage.session import init_db, get_session
from storage.repository import TickerRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def run_pipeline(config: AppConfig) -> list[TickerReport]:
    """Execute the full analysis pipeline for all configured tickers.

    Returns the list of TickerReport objects without rendering output.
    Safe to call from FastAPI or any other context.
    """
    init_db(config.db_path)
    db = next(get_session())
    try:
        tickers = TickerRepository(db).get_all_active()
    finally:
        db.close()

    if not tickers:
        logger.error("No active tickers in database. Add tickers via the API or ticker.cfg.")
        return []

    cache = DataCache(
        cache_dir=Path(".cache"),
        ttl_hours=config.cache_ttl_hours,
    )

    logger.info("Fetching market data for %d ticker(s)...", len(tickers))
    market_data = fetch_all(
        tickers=tickers,
        days=config.historical_days,
        cache=cache,
        max_workers=config.max_fetch_workers,
    )
    if not market_data:
        logger.error("No data fetched. Check your tickers and network connection.")
        return []

    logger.info("Building features for %d ticker(s)...", len(market_data))
    enriched = build_features_all(market_data, max_workers=config.max_compute_workers)
    if not enriched:
        logger.error("Feature engineering produced no results.")
        return []

    logger.info("Running analysis for %d ticker(s)...", len(enriched))
    scores = analyze_all(enriched, config.signal_weights, max_workers=config.max_compute_workers)

    logger.info("Generating forecasts...")
    predictor = get_predictor(config.predictor_method, Path(config.model_dir))
    horizons = [config.forecast_short, config.forecast_long]

    reports: list[TickerReport] = []
    for ticker, score in scores.items():
        if ticker not in enriched:
            continue
        try:
            prediction = predictor.predict(enriched[ticker], score, horizons)
            reports.append(TickerReport(
                ticker=ticker,
                market_type=enriched[ticker].market_type,
                score=score,
                prediction=prediction,
                generated_at=datetime.now(timezone.utc),
            ))
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", ticker, exc)

    if not reports:
        logger.error("No reports generated.")

    return reports


def run(config: AppConfig) -> list[TickerReport]:
    """Run the pipeline and render results to stdout (CLI entry point)."""
    reports = run_pipeline(config)
    if reports:
        render_all(reports)
    return reports


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
