"""Accuracy evaluator: compares stored predictions against actual prices."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf
from sqlalchemy.orm import Session

from config import AppConfig
from data.cache import DataCache
from storage.orm_models import AccuracyORM, PredictionRecord
from storage.repository import AccuracyRepository, PredictionRepository

logger = logging.getLogger(__name__)


def _fetch_actual_price(ticker: str, target_date: datetime) -> float | None:
    """Fetch closing price closest to target_date using yfinance."""
    try:
        start = target_date - timedelta(days=5)
        end = target_date + timedelta(days=2)
        raw = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        if raw is None or raw.empty:
            return None
        # Pick the row closest to target_date
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
        diffs = abs(raw.index - target_date.replace(tzinfo=None))
        closest = raw.iloc[diffs.argmin()]
        col = "Close" if "Close" in closest.index else "close"
        return float(closest[col])
    except Exception as exc:
        logger.warning("Failed to fetch actual price for %s at %s: %s", ticker, target_date, exc)
        return None


def _compute_direction(actual: float, entry: float, threshold: float = 0.005) -> str:
    if entry == 0:
        logger.warning("Cannot compute direction: entry price is zero")
        return "FLAT"
    fwd = (actual - entry) / entry
    if fwd > threshold:
        return "UP"
    if fwd < -threshold:
        return "DOWN"
    return "FLAT"


def check_and_evaluate(db: Session, config: AppConfig) -> list[AccuracyORM]:
    """Find predictions past their horizon, fetch actuals, store AccuracyORM rows."""
    pred_repo = PredictionRepository(db)
    acc_repo = AccuracyRepository(db)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    created = []

    candidates = pred_repo.get_unevaluated_past_horizon()
    if not candidates:
        logger.info("Accuracy evaluation: nothing to evaluate.")
        return []

    logger.info("Evaluating %d prediction(s)...", len(candidates))

    for rec in candidates:
        gen = rec.generated_at
        evaluated_horizons = {a.horizon_days for a in rec.accuracy_records}

        for horizon, pred_mid, pred_dir in [
            (rec.short_horizon, rec.short_mid, rec.short_direction),
            (rec.long_horizon, rec.long_mid, rec.long_direction),
        ]:
            elapsed = (now - gen).days
            if elapsed < horizon or horizon in evaluated_horizons:
                continue

            target_date = gen + timedelta(days=horizon)
            actual_price = _fetch_actual_price(rec.ticker, target_date)
            if actual_price is None:
                continue

            actual_dir = _compute_direction(actual_price, rec.current_price, config.ml_label_threshold)
            price_err = (actual_price - pred_mid) / pred_mid * 100 if pred_mid else 0.0

            acc = AccuracyORM(
                prediction_id=rec.id,
                ticker=rec.ticker,
                horizon_days=horizon,
                evaluated_at=now,
                actual_price=actual_price,
                predicted_direction=pred_dir,
                actual_direction=actual_dir,
                direction_correct=(pred_dir == actual_dir),
                predicted_mid=pred_mid,
                price_error_pct=price_err,
                model_used=rec.model_used,
            )
            acc_repo.store(acc)
            created.append(acc)
            logger.info(
                "Evaluated %s h=%dd: predicted=%s actual=%s correct=%s err=%.1f%%",
                rec.ticker, horizon, pred_dir, actual_dir, pred_dir == actual_dir, price_err,
            )

    # Trigger retraining if threshold met
    _maybe_retrain(db, config, created)

    # Adjust per-ticker signal weights based on accumulated accuracy data
    for ticker in {rec.ticker for rec in created}:
        try:
            from accuracy.weight_optimizer import optimize_weights
            optimize_weights(ticker, db, config.signal_weights)
        except Exception as exc:
            logger.warning("Weight optimization failed for %s: %s", ticker, exc)

    return created


def _maybe_retrain(db: Session, config: AppConfig, new_records: list[AccuracyORM]) -> None:
    """Trigger XGBoost retraining when enough new accuracy data exists."""
    from datetime import timezone
    from pathlib import Path
    from storage.repository import AccuracyRepository

    acc_repo = AccuracyRepository(db)
    seen: set[tuple[str, int]] = set()

    for rec in new_records:
        key = (rec.ticker, rec.horizon_days)
        if key in seen:
            continue
        seen.add(key)

        model_path = Path(config.model_dir) / f"{rec.ticker.replace('=','_')}_{rec.horizon_days}d.json"
        since = datetime.fromtimestamp(model_path.stat().st_mtime) if model_path.exists() else datetime.min
        count = acc_repo.count_since_mtime(rec.ticker, rec.horizon_days, since)

        if count >= config.retrain_min_records or not model_path.exists():
            logger.info("Triggering retrain for %s h=%dd (%d new records)", rec.ticker, rec.horizon_days, count)
            _retrain(rec.ticker, rec.horizon_days, db, config)


def _retrain(ticker: str, horizon: int, db: Session, config: AppConfig) -> None:
    """Retrain XGBoost, LSTM, and RF for one (ticker, horizon) pair."""
    from data.cache import DataCache
    from data import fetch_all
    from features import build_features
    from prediction.trainer import XGBoostTrainer
    from storage.repository import AccuracyRepository

    cache = DataCache(Path(".cache"), config.cache_ttl_hours)
    from storage.repository import TickerRepository
    active_tickers = TickerRepository(db).get_all_active()
    market_data = fetch_all({ticker: active_tickers.get(ticker, "stock")},
                            config.historical_days, cache, 1)

    if ticker not in market_data:
        logger.warning("Cannot retrain %s: no data in cache", ticker)
        return

    try:
        enriched = build_features(market_data[ticker])
    except Exception as exc:
        logger.warning("Feature engineering failed during retrain for %s: %s", ticker, exc)
        return

    acc_repo = AccuracyRepository(db)
    acc_records_orm = acc_repo.get_for_ticker(ticker, horizon=horizon, limit=200)

    from data.models import AccuracyRecord
    acc_domain = [
        AccuracyRecord(
            id=a.id, prediction_id=a.prediction_id, ticker=a.ticker,
            horizon_days=a.horizon_days, evaluated_at=a.evaluated_at,
            actual_price=a.actual_price, predicted_direction=a.predicted_direction,
            actual_direction=a.actual_direction, direction_correct=a.direction_correct,
            predicted_mid=a.predicted_mid, price_error_pct=a.price_error_pct,
        )
        for a in acc_records_orm
    ]

    # XGBoost retrain
    trainer = XGBoostTrainer(Path(config.model_dir), horizon=horizon, label_threshold=config.ml_label_threshold)
    trainer.retrain_with_feedback(enriched, acc_domain)

    # LSTM retrain (silently skips if torch is not installed)
    try:
        from prediction.lstm_trainer import LSTMTrainer
        LSTMTrainer(Path(config.model_dir), horizon=horizon,
                    label_threshold=config.ml_label_threshold).train_and_save(enriched)
    except Exception as exc:
        logger.debug("LSTM retrain skipped for %s h=%dd: %s", ticker, horizon, exc)

    # RF retrain
    try:
        from prediction.rf_trainer import RandomForestTrainer
        RandomForestTrainer(Path(config.model_dir), horizon=horizon,
                            label_threshold=config.ml_label_threshold).retrain_with_feedback(
            enriched, acc_domain
        )
    except Exception as exc:
        logger.debug("RF retrain skipped for %s h=%dd: %s", ticker, horizon, exc)

    # Adjust ensemble blend weights based on per-model accuracy
    try:
        _update_blend_weights(ticker, horizon, db)
    except Exception as exc:
        logger.debug("Blend weight update failed for %s: %s", ticker, exc)


def _update_blend_weights(ticker: str, horizon: int, db) -> None:
    """Compute recent accuracy for XGB / LSTM / RF and adjust 3-way blend weights."""
    from storage.orm_models import AccuracyORM, PredictionRecord
    from accuracy.weight_optimizer import optimize_blend_weights, optimize_3way_blend_weights

    def _model_acc(model_label: str) -> float | None:
        rows = (
            db.query(AccuracyORM)
            .join(PredictionRecord, AccuracyORM.prediction_id == PredictionRecord.id)
            .filter(
                AccuracyORM.ticker == ticker,
                AccuracyORM.horizon_days == horizon,
                PredictionRecord.model_used == model_label,
            )
            .order_by(AccuracyORM.evaluated_at.desc())
            .limit(50)
            .all()
        )
        return (sum(1 for r in rows if r.direction_correct) / len(rows)) if len(rows) >= 10 else None

    xgb_acc  = _model_acc("ml")
    lstm_acc = _model_acc("lstm")
    rf_acc   = _model_acc("rf")

    # Use 3-way optimizer when RF data is available, else fall back to 2-way
    if rf_acc is not None:
        optimize_3way_blend_weights(ticker, db, xgb_acc, lstm_acc, rf_acc)
    else:
        optimize_blend_weights(ticker, db, xgb_acc, lstm_acc)
