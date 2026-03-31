"""XGBoost-based directional price predictor with statistical fallback."""
from __future__ import annotations

import logging
from pathlib import Path

from data.exceptions import PredictionError
from data.models import CompositeScore, EnrichedData, PriceForecast, PredictionResult
from prediction.base import AbstractPredictor
from prediction.trainer import XGBoostTrainer, _LABEL_NAMES

logger = logging.getLogger(__name__)

_DIRECTION_MAP = {0: "DOWN", 1: "FLAT", 2: "UP"}


class MLPredictor(AbstractPredictor):
    """XGBoost direction classifier with StatisticalPredictor price bands.

    Falls back to StatisticalPredictor transparently when no trained model
    exists for the ticker. PredictionResult.model reflects which was used.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or Path(".models")

    def predict(
        self,
        data: EnrichedData,
        score: CompositeScore,
        horizons: list[int],
    ) -> PredictionResult:
        from prediction.statistical import StatisticalPredictor

        if len(data.df) < 30:
            raise PredictionError(f"{data.ticker}: need ≥30 rows")

        # Try to get ML forecasts for each horizon
        forecasts: list[PriceForecast] = []
        used_ml = False

        for h in horizons:
            trainer = XGBoostTrainer(self._model_dir, horizon=h)
            model = trainer.load(data.ticker)

            if model is None:
                # No trained model — fall through to statistical
                break

            features = trainer.extract_features(data)
            if features is None:
                break

            try:
                proba = model.predict_proba(features)[0]   # shape (3,)
                label_idx = int(proba.argmax())
                direction = _DIRECTION_MAP[label_idx]
                ml_confidence = float(proba[label_idx])
                used_ml = True
            except Exception as exc:
                logger.warning("ML inference failed for %s h=%d: %s", data.ticker, h, exc)
                break

            # Price range from statistical predictor (keep ATR bands)
            stat = StatisticalPredictor()
            try:
                stat_result = stat.predict(data, score, [h])
                stat_fc = stat_result.short_forecast
            except PredictionError:
                break

            # Ensure mid is consistent with the ML direction.
            # The stat predictor projects a regression mid independently; when it
            # disagrees with the ML classifier the displayed price contradicts the
            # direction label.  Clamp mid so it always aligns with the direction.
            current = float(data.df["close"].dropna().iloc[-1])
            threshold = 0.005  # 0.5 % — mirrors StatisticalPredictor
            mid = stat_fc.mid
            if direction == "UP" and mid < current * (1 + threshold):
                mid = current * (1 + threshold)
            elif direction == "DOWN" and mid > current * (1 - threshold):
                mid = current * (1 - threshold)

            # Rebuild bands around the adjusted mid, preserving original width
            half_band = (stat_fc.high - stat_fc.low) / 2
            low  = max(mid - half_band, current * 0.01)
            high = mid + half_band

            forecasts.append(PriceForecast(
                horizon_days=h,
                low=round(low, 4),
                mid=round(mid, 4),
                high=round(high, 4),
                direction=direction,      # type: ignore[arg-type]
                confidence=ml_confidence,
            ))

        if len(forecasts) == len(horizons) and used_ml:
            return PredictionResult(
                ticker=data.ticker,
                current_price=round(float(data.df["close"].dropna().iloc[-1]), 4),
                short_forecast=forecasts[0],
                long_forecast=forecasts[1] if len(forecasts) > 1 else forecasts[0],
                model="ml",
            )

        # Fallback: statistical
        logger.info("%s: no ML model — using statistical predictor", data.ticker)
        return StatisticalPredictor().predict(data, score, horizons)
