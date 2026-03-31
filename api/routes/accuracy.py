"""Accuracy and ML retraining routes."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.limiter import limiter
from api.dependencies import get_config, get_db, verify_api_key
from api.schemas import (
    AccuracyRecordSchema,
    AccuracySummarySchema,
    EvaluateResponse,
    RetrainResponse,
    TickerWeightsSchema,
    TrainAllResponse,
    WeightsResetResponse,
    _validate_ticker,
)
from config import AppConfig
from storage.repository import AccuracyRepository, PredictionRepository, TickerWeightsRepository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["accuracy"])


@router.get("/accuracy/{ticker}/summary", response_model=list[AccuracySummarySchema],
            dependencies=[Depends(verify_api_key)])
def accuracy_summary(
    ticker: str,
    db: Session = Depends(get_db),
    config: AppConfig = Depends(get_config),
) -> list[AccuracySummarySchema]:
    """Return direction accuracy and price error for both horizons."""
    ticker = _validate_ticker(ticker)
    acc_repo = AccuracyRepository(db)
    result = []
    for horizon in (config.forecast_short, config.forecast_long):
        summary = acc_repo.summary(ticker, horizon)
        result.append(AccuracySummarySchema(**summary))
    return result


@router.get("/accuracy/{ticker}/records", response_model=list[AccuracyRecordSchema],
            dependencies=[Depends(verify_api_key)])
def accuracy_records(
    ticker: str,
    horizon: int | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[AccuracyRecordSchema]:
    """Raw accuracy records for a ticker."""
    ticker = _validate_ticker(ticker)
    if not 1 <= limit <= 2000:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="limit must be between 1 and 2000")
    acc_repo = AccuracyRepository(db)
    records = acc_repo.get_for_ticker(ticker, horizon=horizon, limit=limit)
    return [AccuracyRecordSchema.model_validate(r) for r in records]


@router.get("/accuracy/{ticker}/weights", response_model=TickerWeightsSchema,
            dependencies=[Depends(verify_api_key)])
def get_ticker_weights_endpoint(
    ticker: str,
    db: Session = Depends(get_db),
) -> TickerWeightsSchema:
    """Return the learned per-ticker signal weights. 404 if no weights have been learned yet."""
    ticker = _validate_ticker(ticker)
    row = TickerWeightsRepository(db).get(ticker)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No learned weights for {ticker!r} yet. Run accuracy evaluation to generate them.",
        )
    return TickerWeightsSchema.model_validate(row)


@router.post("/accuracy/{ticker}/weights/reset", response_model=WeightsResetResponse,
             dependencies=[Depends(verify_api_key)])
def reset_ticker_weights(
    ticker: str,
    db: Session = Depends(get_db),
) -> WeightsResetResponse:
    """Delete learned weights for a ticker, reverting to global config defaults."""
    ticker = _validate_ticker(ticker)
    row = TickerWeightsRepository(db).get(ticker)
    if row is None:
        return WeightsResetResponse(
            ticker=ticker, reset=False,
            message="No learned weights found; already using global defaults.",
        )
    db.delete(row)
    db.commit()
    return WeightsResetResponse(
        ticker=ticker, reset=True,
        message=f"Learned weights for {ticker!r} deleted. Will use global defaults.",
    )


@router.post("/accuracy/evaluate", response_model=EvaluateResponse,
             dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def trigger_evaluation(
    request: Request,
    db: Session = Depends(get_db),
    config: AppConfig = Depends(get_config),
) -> EvaluateResponse:
    """Manually trigger accuracy evaluation for all past-horizon predictions."""
    try:
        from accuracy import check_and_evaluate
        created = check_and_evaluate(db, config)
    except Exception as exc:
        logger.error("Accuracy evaluation failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Accuracy evaluation failed")
    return EvaluateResponse(
        evaluated=len(created),
        records=[AccuracyRecordSchema.model_validate(r) for r in created],
    )


@router.post("/ml/retrain/{ticker}", response_model=RetrainResponse,
             dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def retrain_ticker(
    request: Request,
    ticker: str,
    horizon: int | None = None,
    db: Session = Depends(get_db),
    config: AppConfig = Depends(get_config),
) -> RetrainResponse:
    """Retrain XGBoost for a specific ticker (and optional horizon)."""
    ticker = _validate_ticker(ticker)
    from data import fetch_all
    from data.cache import DataCache
    from features import build_features
    from prediction.trainer import XGBoostTrainer
    from storage.repository import AccuracyRepository
    from data.models import AccuracyRecord

    h = horizon or config.forecast_short
    cache = DataCache(Path(".cache"), config.cache_ttl_hours)
    from storage.repository import TickerRepository
    tickers = TickerRepository(db).get_all_active()
    market_data = fetch_all({ticker: tickers.get(ticker, "stock")},
                            config.historical_days, cache, 1)

    if ticker not in market_data:
        return RetrainResponse(ticker=ticker, horizon=h, success=False,
                               message="No market data available")

    try:
        enriched = build_features(market_data[ticker])
    except Exception as exc:
        logger.error("Feature engineering failed for %s during retrain: %s", ticker, exc)
        return RetrainResponse(ticker=ticker, horizon=h, success=False,
                               message="Feature engineering failed")

    acc_repo = AccuracyRepository(db)
    acc_orm = acc_repo.get_for_ticker(ticker, horizon=h, limit=200)
    acc_domain = [
        AccuracyRecord(
            id=a.id, prediction_id=a.prediction_id, ticker=a.ticker,
            horizon_days=a.horizon_days, evaluated_at=a.evaluated_at,
            actual_price=a.actual_price, predicted_direction=a.predicted_direction,
            actual_direction=a.actual_direction, direction_correct=a.direction_correct,
            predicted_mid=a.predicted_mid, price_error_pct=a.price_error_pct,
        )
        for a in acc_orm
    ]

    trainer = XGBoostTrainer(Path(config.model_dir), horizon=h,
                             label_threshold=config.ml_label_threshold)
    success = trainer.retrain_with_feedback(enriched, acc_domain)

    # Also retrain RF
    try:
        from prediction.rf_trainer import RandomForestTrainer
        RandomForestTrainer(Path(config.model_dir), horizon=h,
                            label_threshold=config.ml_label_threshold).retrain_with_feedback(
            enriched, acc_domain
        )
    except Exception as exc:
        logger.debug("RF retrain skipped for %s h=%dd: %s", ticker, h, exc)

    msg = "Retrained successfully" if success else "Retrain failed (insufficient data)"
    return RetrainResponse(ticker=ticker, horizon=h, success=success, message=msg)


@router.post("/ml/train", response_model=TrainAllResponse,
             dependencies=[Depends(verify_api_key)])
@limiter.limit("2/hour")
def train_all(
    request: Request,
    db: Session = Depends(get_db),
    config: AppConfig = Depends(get_config),
) -> TrainAllResponse:
    """Train XGBoost models for all tickers and configured horizons."""
    from data import fetch_all
    from data.cache import DataCache
    from features import build_features
    from prediction.trainer import XGBoostTrainer

    cache = DataCache(Path(".cache"), config.cache_ttl_hours)
    from storage.repository import TickerRepository
    active_tickers = TickerRepository(db).get_all_active()
    market_data = fetch_all(active_tickers, config.historical_days, cache, config.max_fetch_workers)
    results: list[RetrainResponse] = []

    for ticker in active_tickers:
        if ticker not in market_data:
            for h in (config.forecast_short, config.forecast_long):
                results.append(RetrainResponse(ticker=ticker, horizon=h, success=False,
                                               message="No data"))
            continue
        try:
            enriched = build_features(market_data[ticker])
        except Exception as exc:
            logger.error("Feature engineering failed for %s during train_all: %s", ticker, exc)
            for h in (config.forecast_short, config.forecast_long):
                results.append(RetrainResponse(ticker=ticker, horizon=h, success=False,
                                               message="Feature engineering failed"))
            continue
        for h in (config.forecast_short, config.forecast_long):
            trainer = XGBoostTrainer(Path(config.model_dir), horizon=h,
                                     label_threshold=config.ml_label_threshold)
            success = trainer.train_and_save(enriched)
            # Also train RF
            try:
                from prediction.rf_trainer import RandomForestTrainer
                RandomForestTrainer(Path(config.model_dir), horizon=h,
                                    label_threshold=config.ml_label_threshold).train_and_save(enriched)
            except Exception as exc:
                logger.debug("RF train_all skipped for %s h=%dd: %s", ticker, h, exc)
            results.append(RetrainResponse(
                ticker=ticker, horizon=h, success=success,
                message="Trained" if success else "Failed (insufficient data)",
            ))

    return TrainAllResponse(results=results)
