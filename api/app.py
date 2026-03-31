"""FastAPI application factory with APScheduler background job."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import AsyncGenerator

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.routes import analysis, predictions, accuracy, tickers
from config import AppConfig, load_config
from storage.session import init_db, get_session
from storage.repository import TickerRepository

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _get_tickers(config: AppConfig) -> dict[str, str]:
    """Fetch active tickers from the database."""
    db = next(get_session())
    try:
        return TickerRepository(db).get_all_active()
    finally:
        db.close()


def _accuracy_job(config: AppConfig) -> None:
    """Background job: evaluate past-horizon predictions."""
    db = next(get_session())
    try:
        from accuracy import check_and_evaluate
        created = check_and_evaluate(db, config)
        if created:
            logger.info("Background accuracy job: evaluated %d record(s)", len(created))
    except Exception as exc:
        logger.error("Accuracy evaluation job failed: %s", exc)
    finally:
        db.close()


def _analysis_refresh_job(config: AppConfig) -> None:
    """Background job: re-run the full analysis pipeline for all active tickers."""
    from datetime import datetime, timezone
    from data import fetch_all
    from data.cache import DataCache
    from features import build_features_all
    from analysis import analyze_all
    from prediction import get_predictor
    from data.models import TickerReport
    from storage.repository import PredictionRepository

    tickers = _get_tickers(config)
    if not tickers:
        logger.warning("Analysis refresh: no active tickers in database")
        return

    logger.info("Scheduled analysis refresh started (%d tickers)", len(tickers))
    cache = DataCache(Path(".cache"), config.cache_ttl_hours)

    try:
        market_data = fetch_all(tickers, config.historical_days, cache, config.max_fetch_workers)
    except Exception as exc:
        logger.error("Analysis refresh: data fetch failed: %s", exc)
        return

    try:
        enriched = build_features_all(market_data, max_workers=config.max_compute_workers)
    except Exception as exc:
        logger.error("Analysis refresh: feature engineering failed: %s", exc)
        return

    scores = analyze_all(enriched, config.signal_weights, max_workers=config.max_compute_workers)
    horizons = [config.forecast_short, config.forecast_long]

    db = next(get_session())
    stored = 0
    try:
        predictor = get_predictor(config.predictor_method, Path(config.model_dir), db=db)
        pred_repo = PredictionRepository(db)
        for ticker, score in scores.items():
            if ticker not in enriched:
                continue
            try:
                prediction = predictor.predict(enriched[ticker], score, horizons)
                report = TickerReport(
                    ticker=ticker,
                    market_type=enriched[ticker].market_type,
                    score=score,
                    prediction=prediction,
                    generated_at=datetime.now(timezone.utc),
                )
                pred_repo.store(report)
                stored += 1
            except Exception as exc:
                logger.warning("Analysis refresh: prediction failed for %s: %s", ticker, exc)
    except Exception as exc:
        logger.error("Analysis refresh: storage failed: %s", exc)
    finally:
        db.close()

    logger.info("Scheduled analysis refresh complete: stored %d prediction(s)", stored)

    # Evaluate any predictions that have now passed their horizon
    db2 = next(get_session())
    try:
        from accuracy import check_and_evaluate
        created = check_and_evaluate(db2, config)
        if created:
            logger.info("Analysis refresh: evaluated %d accuracy record(s)", len(created))
    except Exception as exc:
        logger.warning("Analysis refresh: accuracy evaluation failed: %s", exc)
    finally:
        db2.close()


def _initial_train_job(config: AppConfig) -> None:
    """Train any missing models on startup (runs once)."""
    from data import fetch_all
    from data.cache import DataCache
    from features import build_features
    from prediction.trainer import XGBoostTrainer
    from prediction.lstm_trainer import LSTMTrainer
    from prediction.rf_trainer import RandomForestTrainer

    tickers = _get_tickers(config)
    if not tickers:
        return

    cache = DataCache(Path(".cache"), config.cache_ttl_hours)
    try:
        market_data = fetch_all(tickers, config.historical_days, cache, config.max_fetch_workers)
    except Exception as exc:
        logger.warning("Startup training: data fetch failed: %s", exc)
        return

    for ticker in tickers:
        if ticker not in market_data:
            continue
        try:
            enriched = build_features(market_data[ticker])
        except Exception as exc:
            logger.warning("Startup training: features failed for %s: %s", ticker, exc)
            continue
        for horizon in (config.forecast_short, config.forecast_long):
            from prediction.trainer import _safe_ticker_slug
            model_path = Path(config.model_dir) / f"{_safe_ticker_slug(ticker)}_{horizon}d.json"
            if not model_path.exists():
                trainer = XGBoostTrainer(
                    Path(config.model_dir),
                    horizon=horizon,
                    label_threshold=config.ml_label_threshold,
                )
                success = trainer.train_and_save(enriched)
                logger.info("Startup train %s h=%dd: %s", ticker, horizon, "ok" if success else "failed")
            rf_model_path = Path(config.model_dir) / f"{_safe_ticker_slug(ticker)}_{horizon}d_rf.joblib"
            if not rf_model_path.exists():
                try:
                    rf_success = RandomForestTrainer(
                        Path(config.model_dir),
                        horizon=horizon,
                        label_threshold=config.ml_label_threshold,
                    ).train_and_save(enriched)
                    logger.info("Startup RF train %s h=%dd: %s", ticker, horizon, "ok" if rf_success else "failed")
                except Exception as exc:
                    logger.warning("Startup RF train failed for %s h=%dd: %s", ticker, horizon, exc)
            lstm_model_path = Path(config.model_dir) / f"{_safe_ticker_slug(ticker)}_{horizon}d_lstm.pt"
            if not lstm_model_path.exists():
                try:
                    lstm_success = LSTMTrainer(
                        Path(config.model_dir),
                        horizon=horizon,
                        label_threshold=config.ml_label_threshold,
                    ).train_and_save(enriched)
                    logger.info("Startup LSTM train %s h=%dd: %s", ticker, horizon, "ok" if lstm_success else "failed")
                except Exception as exc:
                    logger.warning("Startup LSTM train failed for %s h=%dd: %s", ticker, horizon, exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _scheduler

    config = load_config()

    if not os.getenv("API_TOKEN"):
        logger.warning(
            "API_TOKEN is not set — running in open/dev mode. "
            "Set API_TOKEN in .env before exposing this service externally."
        )

    init_db(config.db_path)
    logger.info("Database initialised")

    # Seed tickers from ticker.cfg on first run
    db = next(get_session())
    try:
        repo = TickerRepository(db)
        if repo.count() == 0:
            added = repo.seed_from_file("ticker.cfg")
            logger.info("Seeded %d ticker(s) from ticker.cfg", added)
    finally:
        db.close()

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        partial(_accuracy_job, config),
        trigger="interval",
        hours=config.accuracy_check_interval_hours,
        id="accuracy_eval",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        partial(_analysis_refresh_job, config),
        trigger="interval",
        hours=config.analysis_refresh_hours,
        id="analysis_refresh",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        partial(_initial_train_job, config),
        trigger="date",
        id="initial_train",
    )
    _scheduler.start()
    logger.info(
        "Scheduler started (analysis refresh every %dh, accuracy check every %dh)",
        config.analysis_refresh_hours,
        config.accuracy_check_interval_hours,
    )

    yield

    _scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")

    from data.fetchers.bloomberg import BloombergFetcher
    BloombergFetcher.close_session()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Market Evaluation Tool",
        description="ML-backed market analysis, forecasting, and accuracy tracking",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501")
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    app.include_router(tickers.router)
    app.include_router(analysis.router)
    app.include_router(predictions.router)
    app.include_router(accuracy.router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app
