from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship

# Explicit lengths on all String columns so the schema is compatible with
# both SQLite (ignores lengths) and MySQL (requires them for indexed columns).


class Base(DeclarativeBase):
    pass


class TickerRecord(Base):
    __tablename__ = "tickers"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(20), unique=True, nullable=False, index=True)
    market_type = Column(String(20), nullable=False)   # stock | crypto | commodity
    active      = Column(Boolean, nullable=False, default=True)
    created_at  = Column(DateTime, nullable=False, default=datetime.utcnow)


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(20), nullable=False, index=True)
    market_type      = Column(String(20), nullable=False)
    generated_at     = Column(DateTime, nullable=False, default=datetime.utcnow)
    model_used       = Column(String(20), nullable=False)         # "statistical" | "ml"
    current_price    = Column(Float, nullable=False)
    # Short forecast
    short_horizon    = Column(Integer, nullable=False)
    short_low        = Column(Float, nullable=False)
    short_mid        = Column(Float, nullable=False)
    short_high       = Column(Float, nullable=False)
    short_direction  = Column(String(10), nullable=False)
    short_confidence = Column(Float, nullable=False)
    # Long forecast
    long_horizon     = Column(Integer, nullable=False)
    long_low         = Column(Float, nullable=False)
    long_mid         = Column(Float, nullable=False)
    long_high        = Column(Float, nullable=False)
    long_direction   = Column(String(10), nullable=False)
    long_confidence  = Column(Float, nullable=False)
    # Score snapshot
    composite_score  = Column(Float, nullable=False)
    action           = Column(String(20), nullable=False)
    score_confidence = Column(Float, nullable=False)
    # JSON snapshot of signal votes at prediction time: {"rsi": 1, "macd_cross": -1, ...}
    signals_json     = Column(Text, nullable=True)

    accuracy_records = relationship("AccuracyORM", back_populates="prediction", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_pred_ticker_generated", "ticker", "generated_at"),
    )


class AccuracyORM(Base):
    __tablename__ = "accuracy_records"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id       = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    ticker              = Column(String(20), nullable=False, index=True)
    horizon_days        = Column(Integer, nullable=False)
    evaluated_at        = Column(DateTime, nullable=False)
    actual_price        = Column(Float, nullable=False)
    predicted_direction = Column(String(10), nullable=False)
    actual_direction    = Column(String(10), nullable=False)
    direction_correct   = Column(Boolean, nullable=False)
    predicted_mid       = Column(Float, nullable=False)
    price_error_pct     = Column(Float, nullable=False)
    # Model that generated this prediction (populated for all new records)
    model_used          = Column(String(20), nullable=True)
    # Per-component directions from simulation (NULL for live predictions)
    xgb_direction       = Column(String(10), nullable=True)
    lstm_direction      = Column(String(10), nullable=True)
    rf_direction        = Column(String(10), nullable=True)

    prediction = relationship("PredictionRecord", back_populates="accuracy_records")

    __table_args__ = (
        Index("ix_acc_ticker_horizon", "ticker", "horizon_days"),
    )


class TickerWeightsORM(Base):
    """Per-ticker learned signal weights, updated by the accuracy feedback loop."""
    __tablename__ = "ticker_weights"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    ticker       = Column(String(20), unique=True, nullable=False, index=True)
    rsi          = Column(Float, nullable=False, default=1.0)
    macd         = Column(Float, nullable=False, default=1.5)
    trend        = Column(Float, nullable=False, default=2.0)
    volume       = Column(Float, nullable=False, default=0.8)
    bb           = Column(Float, nullable=False, default=1.0)
    stoch        = Column(Float, nullable=False, default=0.8)
    sentiment    = Column(Float, nullable=False, default=0.5)
    # Blend weights for ensemble predictor (XGBoost / LSTM / Random Forest)
    xgb_blend    = Column(Float, nullable=False, default=0.50)
    lstm_blend   = Column(Float, nullable=False, default=0.25)
    rf_blend     = Column(Float, nullable=False, default=0.25)
    update_count = Column(Integer, nullable=False, default=0)
    updated_at   = Column(DateTime, nullable=False, default=datetime.utcnow)


class SentimentSnapshotORM(Base):
    """Cached sentiment snapshot persisted after each analysis call."""
    __tablename__ = "sentiment_snapshots"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(20), nullable=False, index=True)
    fetched_at       = Column(DateTime, nullable=False, index=True)
    window_hours     = Column(Integer, nullable=False, default=24)
    mention_count    = Column(Integer, nullable=False, default=0)
    sentiment_score  = Column(Float,   nullable=False, default=0.0)
    sentiment_velocity = Column(Float, nullable=False, default=0.0)
    bullish_ratio    = Column(Float,   nullable=False, default=0.5)
    engagement_score = Column(Float,   nullable=False, default=0.0)

    __table_args__ = (
        Index("ix_sentiment_ticker_fetched", "ticker", "fetched_at"),
    )
