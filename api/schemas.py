"""Pydantic response/request schemas for the FastAPI service."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Ticker symbols: 1–15 uppercase alphanumeric chars, hyphens, equals sign
_TICKER_RE = re.compile(r"^[A-Z0-9\-=]{1,15}$")


def _validate_ticker(v: str) -> str:
    v = v.strip().upper()
    if not _TICKER_RE.match(v):
        raise ValueError(
            f"Invalid ticker {v!r}. Must be 1–15 uppercase alphanumeric characters, hyphens, or '='."
        )
    return v


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class ForecastSchema(BaseModel):
    horizon_days: int
    low: float
    mid: float
    high: float
    direction: str
    confidence: float


class ScoreSchema(BaseModel):
    score: float
    action: str
    confidence: float
    signal_details: dict


class PredictionSchema(BaseModel):
    model: str
    current_price: float
    short_forecast: ForecastSchema
    long_forecast: ForecastSchema


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=15)
    market_type: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        return _validate_ticker(v)


class TickerReportSchema(BaseModel):
    ticker: str
    market_type: str
    generated_at: datetime
    score: ScoreSchema
    prediction: PredictionSchema
    prediction_id: Optional[int] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------------

class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi_14: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None


class ChartDataSchema(BaseModel):
    ticker: str
    points: list[OHLCVPoint]


# ---------------------------------------------------------------------------
# Predictions (stored)
# ---------------------------------------------------------------------------

class StoredPredictionSchema(BaseModel):
    id: int
    ticker: str
    market_type: str
    generated_at: datetime
    model_used: str
    current_price: float
    short_horizon: int
    short_mid: float
    short_direction: str
    short_confidence: float
    long_horizon: int
    long_mid: float
    long_direction: str
    long_confidence: float
    composite_score: float
    action: str

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

class AccuracyRecordSchema(BaseModel):
    id: int
    prediction_id: int
    ticker: str
    horizon_days: int
    evaluated_at: datetime
    actual_price: float
    predicted_direction: str
    actual_direction: str
    direction_correct: bool
    predicted_mid: float
    price_error_pct: float
    model_used: Optional[str] = None
    xgb_direction: Optional[str] = None
    lstm_direction: Optional[str] = None
    rf_direction: Optional[str] = None

    model_config = {"from_attributes": True}


class AccuracySummarySchema(BaseModel):
    ticker: str
    horizon_days: int
    n_evaluated: int
    n_correct: Optional[int] = None
    direction_accuracy_pct: float
    mean_price_error_pct: float


class EvaluateResponse(BaseModel):
    evaluated: int
    records: list[AccuracyRecordSchema]


# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------

class RetrainResponse(BaseModel):
    ticker: str
    horizon: int
    success: bool
    message: str


class TrainAllResponse(BaseModel):
    results: list[RetrainResponse]


# ---------------------------------------------------------------------------
# Per-ticker learned weights
# ---------------------------------------------------------------------------

class TickerWeightsSchema(BaseModel):
    ticker: str
    rsi: float
    macd: float
    trend: float
    volume: float
    bb: float
    stoch: float
    xgb_blend: Optional[float] = None
    lstm_blend: Optional[float] = None
    rf_blend: Optional[float] = None
    update_count: int
    updated_at: datetime

    model_config = {"from_attributes": True}


class WeightsResetResponse(BaseModel):
    ticker: str
    reset: bool
    message: str
