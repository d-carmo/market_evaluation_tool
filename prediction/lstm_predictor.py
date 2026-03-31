"""LSTM-based directional price predictor.

Falls back transparently to StatisticalPredictor when:
- PyTorch is not installed.
- No saved LSTM model exists for the ticker/horizon.
- Inference fails for any reason.

PredictionResult.model is "lstm" when LSTM is used, "statistical" on fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path

from data.exceptions import PredictionError
from data.models import CompositeScore, EnrichedData, PriceForecast, PredictionResult
from prediction.base import AbstractPredictor
from prediction.lstm_trainer import LSTMTrainer

logger = logging.getLogger(__name__)

_DIRECTION_MAP = {0: "DOWN", 1: "FLAT", 2: "UP"}


class LSTMPredictor(AbstractPredictor):
    """Per-ticker LSTM direction classifier with statistical price bands.

    Architecture mirrors MLPredictor: LSTM provides direction probability,
    StatisticalPredictor supplies ATR-based price bands, then mid is clamped
    to agree with the LSTM direction (same consistency fix as MLPredictor).
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        seq_len: int = 60,
        hidden: int = 128,
    ) -> None:
        self._model_dir = model_dir or Path(".models")
        self._seq_len = seq_len
        self._hidden  = hidden

    def predict(
        self,
        data: EnrichedData,
        score: CompositeScore,
        horizons: list[int],
    ) -> PredictionResult:
        from prediction.statistical import StatisticalPredictor

        if len(data.df) < self._seq_len:
            raise PredictionError(f"{data.ticker}: need ≥{self._seq_len} rows for LSTM")

        forecasts: list[PriceForecast] = []
        used_lstm = False

        for h in horizons:
            trainer = LSTMTrainer(
                model_dir=self._model_dir,
                horizon=h,
                seq_len=self._seq_len,
                hidden=self._hidden,
            )
            proba = trainer.predict_proba(data)

            if proba is None:
                break   # no model → fall through to statistical

            label_idx  = int(proba.argmax())
            direction  = _DIRECTION_MAP[label_idx]
            confidence = float(proba[label_idx])
            used_lstm  = True

            # Price bands from statistical predictor
            stat = StatisticalPredictor()
            try:
                stat_result = stat.predict(data, score, [h])
                stat_fc     = stat_result.short_forecast
            except PredictionError:
                break

            # Clamp mid to be consistent with LSTM direction
            current   = float(data.df["close"].dropna().iloc[-1])
            threshold = 0.005
            mid = stat_fc.mid
            if direction == "UP"   and mid < current * (1 + threshold):
                mid = current * (1 + threshold)
            elif direction == "DOWN" and mid > current * (1 - threshold):
                mid = current * (1 - threshold)

            half_band = (stat_fc.high - stat_fc.low) / 2
            low  = max(mid - half_band, current * 0.01)
            high = mid + half_band

            forecasts.append(PriceForecast(
                horizon_days=h,
                low=round(low,  4),
                mid=round(mid,  4),
                high=round(high, 4),
                direction=direction,      # type: ignore[arg-type]
                confidence=confidence,
            ))

        if len(forecasts) == len(horizons) and used_lstm:
            return PredictionResult(
                ticker=data.ticker,
                current_price=round(float(data.df["close"].dropna().iloc[-1]), 4),
                short_forecast=forecasts[0],
                long_forecast=forecasts[1] if len(forecasts) > 1 else forecasts[0],
                model="lstm",
            )

        logger.info("%s: no LSTM model — using statistical predictor", data.ticker)
        return StatisticalPredictor().predict(data, score, horizons)
