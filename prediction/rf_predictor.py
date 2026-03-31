"""Random Forest predictor — wraps RandomForestTrainer for inference.

Fallback hierarchy:
  RF model available  → RF classification
  RF model missing    → StatisticalPredictor

PredictionResult.model is "rf" when the RF model contributes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from data.exceptions import PredictionError
from data.models import CompositeScore, EnrichedData, PriceForecast, PredictionResult
from prediction.base import AbstractPredictor
from prediction.rf_trainer import RandomForestTrainer

logger = logging.getLogger(__name__)

_DIRECTION_MAP = {0: "DOWN", 1: "FLAT", 2: "UP"}


class RFPredictor(AbstractPredictor):
    """Predict directional price movement using a trained Random Forest model."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or Path(".models")

    def predict(
        self,
        data: EnrichedData,
        score: CompositeScore,
        horizons: list[int],
    ) -> PredictionResult:
        from prediction.statistical import StatisticalPredictor

        forecasts: list[PriceForecast] = []

        for h in horizons:
            trainer = RandomForestTrainer(self._model_dir, horizon=h)
            model = trainer.load(data.ticker)
            if model is None:
                break

            feats = trainer.extract_features(data)
            if feats is None:
                break

            proba = model.predict_proba(feats)[0].astype(np.float32)
            label_idx = int(proba.argmax())
            direction = _DIRECTION_MAP[label_idx]
            confidence = float(proba[label_idx])

            stat = StatisticalPredictor()
            try:
                stat_result = stat.predict(data, score, [h])
                stat_fc = stat_result.short_forecast
            except PredictionError:
                break

            current = float(data.df["close"].dropna().iloc[-1])
            threshold = 0.005
            mid = stat_fc.mid
            if direction == "UP" and mid < current * (1 + threshold):
                mid = current * (1 + threshold)
            elif direction == "DOWN" and mid > current * (1 - threshold):
                mid = current * (1 - threshold)

            half_band = (stat_fc.high - stat_fc.low) / 2
            low = max(mid - half_band, current * 0.01)
            high = mid + half_band

            forecasts.append(PriceForecast(
                horizon_days=h,
                low=round(low, 4),
                mid=round(mid, 4),
                high=round(high, 4),
                direction=direction,      # type: ignore[arg-type]
                confidence=confidence,
            ))

        if len(forecasts) == len(horizons):
            current = float(data.df["close"].dropna().iloc[-1])
            return PredictionResult(
                ticker=data.ticker,
                current_price=round(current, 4),
                short_forecast=forecasts[0],
                long_forecast=forecasts[1] if len(forecasts) > 1 else forecasts[0],
                model="rf",
            )

        logger.info("%s: RF fallback to statistical predictor", data.ticker)
        return StatisticalPredictor().predict(data, score, horizons)
