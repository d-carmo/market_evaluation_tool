"""Database repository — all SQLAlchemy operations isolated here."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session

from data.models import AccuracyRecord, StoredPrediction, TickerReport
from storage.orm_models import (
    AccuracyORM, PredictionRecord, SentimentSnapshotORM, TickerRecord, TickerWeightsORM,
)

logger = logging.getLogger(__name__)


class PredictionRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def store(self, report: TickerReport, signals_json: str | None = None) -> int:
        """Persist a TickerReport and return the new record id."""
        sc = report.score
        pred = report.prediction
        rec = PredictionRecord(
            ticker=report.ticker,
            market_type=report.market_type,
            generated_at=report.generated_at,
            model_used=pred.model,
            current_price=pred.current_price,
            short_horizon=pred.short_forecast.horizon_days,
            short_low=pred.short_forecast.low,
            short_mid=pred.short_forecast.mid,
            short_high=pred.short_forecast.high,
            short_direction=pred.short_forecast.direction,
            short_confidence=pred.short_forecast.confidence,
            long_horizon=pred.long_forecast.horizon_days,
            long_low=pred.long_forecast.low,
            long_mid=pred.long_forecast.mid,
            long_high=pred.long_forecast.high,
            long_direction=pred.long_forecast.direction,
            long_confidence=pred.long_forecast.confidence,
            composite_score=sc.score,
            action=sc.action,
            score_confidence=sc.confidence,
            signals_json=signals_json,
        )
        self._db.add(rec)
        self._db.commit()
        self._db.refresh(rec)
        return rec.id

    def get_latest(self, ticker: str) -> PredictionRecord | None:
        return (
            self._db.query(PredictionRecord)
            .filter(PredictionRecord.ticker == ticker)
            .order_by(PredictionRecord.generated_at.desc())
            .first()
        )

    def get_history(self, ticker: str, limit: int = 50) -> list[PredictionRecord]:
        return (
            self._db.query(PredictionRecord)
            .filter(PredictionRecord.ticker == ticker)
            .order_by(PredictionRecord.generated_at.desc())
            .limit(limit)
            .all()
        )

    def get_all_latest(self) -> list[PredictionRecord]:
        """One row per ticker — the most recent prediction for each."""
        from sqlalchemy import func
        subq = (
            self._db.query(
                PredictionRecord.ticker,
                func.max(PredictionRecord.generated_at).label("max_gen"),
            )
            .group_by(PredictionRecord.ticker)
            .subquery()
        )
        return (
            self._db.query(PredictionRecord)
            .join(subq, (PredictionRecord.ticker == subq.c.ticker) &
                        (PredictionRecord.generated_at == subq.c.max_gen))
            .all()
        )

    def get_unevaluated_past_horizon(self) -> list[PredictionRecord]:
        """Records whose short or long horizon has elapsed with no accuracy evaluation."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        # Find predictions where long horizon has passed
        candidates = (
            self._db.query(PredictionRecord)
            .all()
        )
        result = []
        for rec in candidates:
            gen = rec.generated_at
            elapsed_days = (now - gen).days
            existing_horizons = {a.horizon_days for a in rec.accuracy_records}
            for horizon in (rec.short_horizon, rec.long_horizon):
                if elapsed_days >= horizon and horizon not in existing_horizons:
                    result.append(rec)
                    break
        return result

    def tickers(self) -> list[str]:
        rows = self._db.query(PredictionRecord.ticker).distinct().all()
        return [r[0] for r in rows]


class AccuracyRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def store(self, rec: AccuracyORM) -> None:
        self._db.add(rec)
        self._db.commit()

    def get_for_ticker(self, ticker: str, horizon: int | None = None, limit: int = 200) -> list[AccuracyORM]:
        q = self._db.query(AccuracyORM).filter(AccuracyORM.ticker == ticker)
        if horizon is not None:
            q = q.filter(AccuracyORM.horizon_days == horizon)
        return q.order_by(AccuracyORM.evaluated_at.desc()).limit(limit).all()

    def summary(self, ticker: str, horizon: int) -> dict:
        records = self.get_for_ticker(ticker, horizon=horizon)
        if not records:
            return {"ticker": ticker, "horizon_days": horizon, "n_evaluated": 0,
                    "direction_accuracy_pct": 0.0, "mean_price_error_pct": 0.0}
        n = len(records)
        correct = sum(1 for r in records if r.direction_correct)
        errors = [abs(r.price_error_pct) for r in records]
        return {
            "ticker": ticker,
            "horizon_days": horizon,
            "n_evaluated": n,
            "n_correct": correct,
            "direction_accuracy_pct": round(correct / n * 100, 1),
            "mean_price_error_pct": round(sum(errors) / n, 2),
        }

    def count_since_mtime(self, ticker: str, horizon: int, since: datetime) -> int:
        return (
            self._db.query(AccuracyORM)
            .filter(
                AccuracyORM.ticker == ticker,
                AccuracyORM.horizon_days == horizon,
                AccuracyORM.evaluated_at >= since,
            )
            .count()
        )


def _detect_market_type(symbol: str) -> str:
    """Infer market type from ticker symbol conventions."""
    if symbol.endswith("-USD"):
        return "crypto"
    if symbol.endswith("=F"):
        return "commodity"
    return "stock"


class TickerRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_all_active(self) -> dict[str, str]:
        """Return {symbol: market_type} for all active tickers."""
        rows = self._db.query(TickerRecord).filter(TickerRecord.active == True).all()
        return {r.symbol: r.market_type for r in rows}

    def add(self, symbol: str, market_type: str | None = None) -> TickerRecord:
        """Add a ticker; infers market_type if not provided. Returns existing row if duplicate."""
        symbol = symbol.upper()
        existing = self._db.query(TickerRecord).filter(TickerRecord.symbol == symbol).first()
        if existing:
            if not existing.active:
                existing.active = True
                self._db.commit()
                self._db.refresh(existing)
            return existing
        rec = TickerRecord(
            symbol=symbol,
            market_type=market_type or _detect_market_type(symbol),
            active=True,
        )
        self._db.add(rec)
        self._db.commit()
        self._db.refresh(rec)
        return rec

    def remove(self, symbol: str) -> bool:
        """Deactivate a ticker (soft delete). Returns True if found."""
        rec = self._db.query(TickerRecord).filter(
            TickerRecord.symbol == symbol.upper()
        ).first()
        if rec is None:
            return False
        rec.active = False
        self._db.commit()
        return True

    def count(self) -> int:
        return self._db.query(TickerRecord).filter(TickerRecord.active == True).count()

    def seed_from_file(self, cfg_path: str = "ticker.cfg") -> int:
        """Populate tickers table from ticker.cfg. Returns number of tickers added."""
        from pathlib import Path
        path = Path(cfg_path)
        if not path.exists():
            return 0

        section_to_type = {"[stocks]": "stock", "[crypto]": "crypto", "[commodities]": "commodity"}
        current_type: str | None = None
        added = 0

        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower() in section_to_type:
                current_type = section_to_type[line.lower()]
            elif current_type:
                symbol = line.upper()
                existing = self._db.query(TickerRecord).filter(
                    TickerRecord.symbol == symbol
                ).first()
                if not existing:
                    self._db.add(TickerRecord(symbol=symbol, market_type=current_type, active=True))
                    added += 1

        if added:
            self._db.commit()
        return added


class TickerWeightsRepository:
    """CRUD for per-ticker learned signal weights."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, ticker: str) -> TickerWeightsORM | None:
        return (
            self._db.query(TickerWeightsORM)
            .filter(TickerWeightsORM.ticker == ticker)
            .first()
        )

    def upsert(self, ticker: str, weights_dict: dict[str, float]) -> TickerWeightsORM:
        """Create or update learned weights for a ticker. Returns the ORM row."""
        _WEIGHT_FIELDS = ("rsi", "macd", "trend", "volume", "bb", "stoch",
                          "sentiment", "xgb_blend", "lstm_blend", "rf_blend")
        _DEFAULTS: dict[str, float] = {
            "rsi": 1.0, "macd": 1.5, "trend": 2.0, "volume": 0.8,
            "bb": 1.0, "stoch": 0.8, "sentiment": 0.5,
            "xgb_blend": 0.50, "lstm_blend": 0.25, "rf_blend": 0.25,
        }
        row = self.get(ticker)
        if row is None:
            row = TickerWeightsORM(
                ticker=ticker,
                **{k: weights_dict.get(k, _DEFAULTS[k]) for k in _WEIGHT_FIELDS},
                update_count=1,
                updated_at=datetime.utcnow(),
            )
            self._db.add(row)
        else:
            for key in _WEIGHT_FIELDS:
                if key in weights_dict:
                    setattr(row, key, weights_dict[key])
            row.update_count += 1
            row.updated_at = datetime.utcnow()
        self._db.commit()
        self._db.refresh(row)
        return row

    def get_all(self) -> list[TickerWeightsORM]:
        return self._db.query(TickerWeightsORM).order_by(TickerWeightsORM.ticker).all()


class SentimentRepository:
    """CRUD for persisted sentiment snapshots."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def store(self, snap) -> None:
        """Persist a SentimentSnapshot domain object."""
        from data.models import SentimentSnapshot
        rec = SentimentSnapshotORM(
            ticker=snap.ticker,
            fetched_at=snap.fetched_at.replace(tzinfo=None),
            window_hours=snap.window_hours,
            mention_count=snap.mention_count,
            sentiment_score=snap.sentiment_score,
            sentiment_velocity=snap.sentiment_velocity,
            bullish_ratio=snap.bullish_ratio,
            engagement_score=snap.engagement_score,
        )
        self._db.add(rec)
        self._db.commit()

    def get_latest(self, ticker: str) -> SentimentSnapshotORM | None:
        return (
            self._db.query(SentimentSnapshotORM)
            .filter(SentimentSnapshotORM.ticker == ticker)
            .order_by(SentimentSnapshotORM.fetched_at.desc())
            .first()
        )

    def get_previous(self, ticker: str, before_id: int) -> SentimentSnapshotORM | None:
        """Return the snapshot immediately before the given id."""
        return (
            self._db.query(SentimentSnapshotORM)
            .filter(SentimentSnapshotORM.ticker == ticker,
                    SentimentSnapshotORM.id < before_id)
            .order_by(SentimentSnapshotORM.fetched_at.desc())
            .first()
        )
