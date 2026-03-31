"""Analysis routes: run pipeline for a ticker, get latest report, chart data."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.limiter import limiter
from api.dependencies import get_config, get_db, verify_api_key
from api.schemas import (
    AnalyzeRequest,
    ChartDataSchema,
    OHLCVPoint,
    StoredPredictionSchema,
    TickerReportSchema,
    ForecastSchema,
    ScoreSchema,
    PredictionSchema,
    _validate_ticker,
)
from config import AppConfig
from data import fetch_all
from data.cache import DataCache
from features import build_features
from analysis import analyze, get_ticker_weights
from prediction import get_predictor
from data.models import TickerReport
from storage.repository import (
    PredictionRepository, SentimentRepository, TickerRepository, _detect_market_type,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analysis"])


def _fetch_sentiment(ticker: str, market_type: str, config: AppConfig, db: Session | None):
    """Fetch a SentimentSnapshot if Reddit is configured; persist it; return it or None."""
    if not config.sentiment_enabled:
        return None
    try:
        from pathlib import Path as _Path
        from sentiment import get_sentiment_snapshot

        prev_score: float | None = None
        if db is not None:
            prev = SentimentRepository(db).get_latest(ticker)
            if prev is not None:
                prev_score = prev.sentiment_score

        snap = get_sentiment_snapshot(
            ticker=ticker,
            market_type=market_type,
            reddit_client_id=config.reddit_client_id,
            reddit_client_secret=config.reddit_client_secret,
            user_agent=config.reddit_user_agent,
            cache_dir=_Path(".cache/reddit"),
            ttl_hours=config.sentiment_ttl_hours,
            prev_score=prev_score,
        )
        if snap is not None and db is not None:
            SentimentRepository(db).store(snap)
        return snap
    except Exception as exc:
        logger.warning("Sentiment fetch skipped for %s: %s", ticker, exc)
        return None


def _run_single(ticker: str, config: AppConfig, db: Session | None = None) -> TickerReport:
    """Run the full pipeline for one ticker and return a TickerReport."""
    # Resolve market_type: prefer DB record, fall back to symbol convention
    market_type = _detect_market_type(ticker)
    if db is not None:
        rows = TickerRepository(db).get_all_active()
        if ticker in rows:
            market_type = rows[ticker]
    cache = DataCache(Path(".cache"), config.cache_ttl_hours)
    market_data = fetch_all({ticker: market_type}, config.historical_days, cache, 1)
    if ticker not in market_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No market data available for {ticker!r}")

    try:
        enriched = build_features(market_data[ticker])
    except Exception as exc:
        logger.error("Feature engineering failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Feature engineering failed")

    # Phase 2: inject sentiment features into the current inference row
    snapshot = _fetch_sentiment(ticker, market_type, config, db)
    from features.pipeline import augment_with_sentiment
    enriched = augment_with_sentiment(enriched, snapshot)

    weights = get_ticker_weights(ticker, db, config.signal_weights) if db is not None else config.signal_weights
    score = analyze(enriched, weights, snapshot=snapshot)
    predictor = get_predictor(config.predictor_method, Path(config.model_dir), db=db)
    horizons = [config.forecast_short, config.forecast_long]

    try:
        prediction = predictor.predict(enriched, score, horizons)
    except Exception as exc:
        logger.error("Prediction failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Prediction failed")

    return TickerReport(
        ticker=ticker,
        market_type=enriched.market_type,
        score=score,
        prediction=prediction,
        generated_at=datetime.now(timezone.utc),
    )


def _report_to_schema(report: TickerReport, prediction_id: int | None = None) -> TickerReportSchema:
    sc = report.score
    pred = report.prediction
    return TickerReportSchema(
        ticker=report.ticker,
        market_type=report.market_type,
        generated_at=report.generated_at,
        score=ScoreSchema(
            score=sc.score,
            action=sc.action,
            confidence=sc.confidence,
            signal_details={s.name: {"value": s.value, "raw": s.raw} for s in sc.signals},
        ),
        prediction=PredictionSchema(
            model=pred.model,
            current_price=pred.current_price,
            short_forecast=ForecastSchema(
                horizon_days=pred.short_forecast.horizon_days,
                low=pred.short_forecast.low,
                mid=pred.short_forecast.mid,
                high=pred.short_forecast.high,
                direction=pred.short_forecast.direction,
                confidence=pred.short_forecast.confidence,
            ),
            long_forecast=ForecastSchema(
                horizon_days=pred.long_forecast.horizon_days,
                low=pred.long_forecast.low,
                mid=pred.long_forecast.mid,
                high=pred.long_forecast.high,
                direction=pred.long_forecast.direction,
                confidence=pred.long_forecast.confidence,
            ),
        ),
        prediction_id=prediction_id,
    )


@router.post("/", response_model=TickerReportSchema, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def analyze_ticker_endpoint(
    request: Request,
    req: AnalyzeRequest,
    db: Session = Depends(get_db),
    config: AppConfig = Depends(get_config),
) -> TickerReportSchema:
    """Run full analysis pipeline for a ticker and persist the prediction."""
    import json
    report = _run_single(req.ticker, config, db)
    signals_json = json.dumps({s.name: s.value for s in report.score.signals})
    pred_repo = PredictionRepository(db)
    pred_id = pred_repo.store(report, signals_json=signals_json)
    return _report_to_schema(report, pred_id)


@router.get("/{ticker}/latest", response_model=StoredPredictionSchema,
            dependencies=[Depends(verify_api_key)])
def get_latest_prediction(
    ticker: str,
    db: Session = Depends(get_db),
) -> StoredPredictionSchema:
    """Return the most recent stored prediction for a ticker."""
    ticker = _validate_ticker(ticker)
    pred_repo = PredictionRepository(db)
    rec = pred_repo.get_latest(ticker)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No prediction found for {ticker!r}")
    return StoredPredictionSchema.model_validate(rec)


@router.get("/{ticker}/chart", response_model=ChartDataSchema,
            dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
def get_chart_data(
    request: Request,
    ticker: str,
    config: AppConfig = Depends(get_config),
) -> ChartDataSchema:
    """Return OHLCV + indicator columns for charting (last 120 rows)."""
    ticker = _validate_ticker(ticker)
    market_type = _detect_market_type(ticker)
    cache = DataCache(Path(".cache"), config.cache_ttl_hours)
    market_data = fetch_all({ticker: market_type}, config.historical_days, cache, 1)
    if ticker not in market_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No market data available for {ticker!r}")

    try:
        enriched = build_features(market_data[ticker])
    except Exception as exc:
        logger.error("Feature engineering failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Feature engineering failed")

    df = enriched.df.tail(120).copy()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index

    points: list[OHLCVPoint] = []
    for ts, row in df.iterrows():
        def _get(col: str) -> float | None:
            if col in row.index:
                v = row[col]
                return None if (v != v) else float(v)
            return None

        points.append(OHLCVPoint(
            date=str(ts)[:10],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0)),
            rsi_14=_get("rsi_14"),
            macd_hist=_get("macd_hist"),
            bb_upper=_get("bb_upper"),
            bb_lower=_get("bb_lower"),
        ))

    return ChartDataSchema(ticker=ticker, points=points)
