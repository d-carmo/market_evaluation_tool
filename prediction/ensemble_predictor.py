"""Ensemble predictor: blends XGBoost, LSTM, and (optionally) Random Forest
probability distributions with per-ticker blend weights.

Blend weights are read from the ticker_weights DB table:
  xgb_blend, lstm_blend, rf_blend
Defaults: 0.50 / 0.25 / 0.25  (rf_ensemble — 3-way)
          0.60 / 0.40 / 0.00  (ensemble    — XGB+LSTM only, backward compatible)

Fallback hierarchy (applied per model type):
  All available       → weighted blend (2- or 3-way, weights re-normalised)
  Only one available  → that model alone (model label = "ml"/"lstm"/"rf")
  None available      → StatisticalPredictor (model label = "statistical")

PredictionResult.model is "rf_ensemble" (3-way) or "ensemble" (2-way XGB+LSTM).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from data.exceptions import PredictionError
from data.models import CompositeScore, EnrichedData, PriceForecast, PredictionResult
from prediction.base import AbstractPredictor
from prediction.trainer import XGBoostTrainer, _LABEL_NAMES
from prediction.lstm_trainer import LSTMTrainer

logger = logging.getLogger(__name__)

_DIRECTION_MAP = {0: "DOWN", 1: "FLAT", 2: "UP"}

# Defaults when no DB row exists — rf_ensemble (3-way)
_DEFAULT_XGB_BLEND  = 0.50
_DEFAULT_LSTM_BLEND = 0.25
_DEFAULT_RF_BLEND   = 0.25

# Defaults for the legacy 2-way ensemble (no RF)
_DEFAULT_XGB_BLEND_2WAY  = 0.60
_DEFAULT_LSTM_BLEND_2WAY = 0.40


class EnsemblePredictor(AbstractPredictor):
    """Blends XGBoost + LSTM (+ optionally RF) probability vectors.

    Pass ``include_rf=True`` (or use ``get_predictor("rf_ensemble")``) to
    activate the Random Forest component.  Blend weights are loaded from the
    DB when a session is injected via ``set_db_session()``.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        seq_len: int = 60,
        lstm_hidden: int = 128,
        include_rf: bool = False,
    ) -> None:
        self._model_dir  = model_dir or Path(".models")
        self._seq_len    = seq_len
        self._hidden     = lstm_hidden
        self._include_rf = include_rf
        self._db         = None   # optional; set via set_db_session()

    def set_db_session(self, db) -> None:
        """Inject a SQLAlchemy Session so blend weights can be loaded from DB."""
        self._db = db

    def predict(
        self,
        data: EnrichedData,
        score: CompositeScore,
        horizons: list[int],
    ) -> PredictionResult:
        from prediction.statistical import StatisticalPredictor

        xgb_blend, lstm_blend, rf_blend = self._get_blends(data.ticker)

        forecasts: list[PriceForecast] = []
        # Track which model types contributed across all horizons.
        used_xgb  = False
        used_lstm = False
        used_rf   = False

        for h in horizons:
            xgb_proba  = self._xgb_proba(data, h)
            lstm_proba = self._lstm_proba(data, h)
            rf_proba   = self._rf_proba(data, h) if self._include_rf else None

            # Collect available (proba, weight) pairs
            available: list[tuple[np.ndarray, float]] = []
            if xgb_proba is not None:
                available.append((xgb_proba, xgb_blend))
                used_xgb = True
            if lstm_proba is not None:
                available.append((lstm_proba, lstm_blend))
                used_lstm = True
            if rf_proba is not None:
                available.append((rf_proba, rf_blend))
                used_rf = True

            if not available:
                break

            # Normalise surviving weights and blend
            total_w = sum(w for _, w in available)
            blended = sum(p * (w / total_w) for p, w in available).astype(np.float32)

            label_idx  = int(blended.argmax())
            direction  = _DIRECTION_MAP[label_idx]
            confidence = float(blended[label_idx])

            # Price bands from statistical predictor
            stat = StatisticalPredictor()
            try:
                stat_result = stat.predict(data, score, [h])
                stat_fc     = stat_result.short_forecast
            except PredictionError:
                break

            current   = float(data.df["close"].dropna().iloc[-1])
            threshold = 0.005
            mid = stat_fc.mid
            if direction == "UP" and mid < current * (1 + threshold):
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

        if len(forecasts) == len(horizons):
            n_models = sum([used_xgb, used_lstm, used_rf])
            if n_models >= 2:
                model_label = "rf_ensemble" if self._include_rf else "ensemble"
            elif used_xgb:
                model_label = "ml"
            elif used_lstm:
                model_label = "lstm"
            else:
                model_label = "rf"
            return PredictionResult(
                ticker=data.ticker,
                current_price=round(float(data.df["close"].dropna().iloc[-1]), 4),
                short_forecast=forecasts[0],
                long_forecast=forecasts[1] if len(forecasts) > 1 else forecasts[0],
                model=model_label,
            )

        logger.info("%s: ensemble fallback to statistical predictor", data.ticker)
        return StatisticalPredictor().predict(data, score, horizons)

    # ── private ───────────────────────────────────────────────────────────────

    def _get_blends(self, ticker: str) -> tuple[float, float, float]:
        """Return (xgb_blend, lstm_blend, rf_blend) for this ticker."""
        if self._db is not None:
            try:
                from storage.repository import TickerWeightsRepository
                row = TickerWeightsRepository(self._db).get(ticker)
                if row is not None:
                    xgb  = float(getattr(row, "xgb_blend",  _DEFAULT_XGB_BLEND))
                    lstm = float(getattr(row, "lstm_blend", _DEFAULT_LSTM_BLEND))
                    rf   = float(getattr(row, "rf_blend",   _DEFAULT_RF_BLEND))
                    return xgb, lstm, rf
            except Exception as exc:
                logger.debug("Blend weight lookup failed for %s: %s", ticker, exc)

        if self._include_rf:
            return _DEFAULT_XGB_BLEND, _DEFAULT_LSTM_BLEND, _DEFAULT_RF_BLEND
        return _DEFAULT_XGB_BLEND_2WAY, _DEFAULT_LSTM_BLEND_2WAY, 0.0

    def _xgb_proba(self, data: EnrichedData, horizon: int) -> np.ndarray | None:
        try:
            trainer = XGBoostTrainer(self._model_dir, horizon=horizon)
            model   = trainer.load(data.ticker)
            if model is None:
                return None
            feats = trainer.extract_features(data)
            if feats is None:
                return None
            return model.predict_proba(feats)[0].astype(np.float32)
        except Exception as exc:
            logger.debug("XGB proba failed for %s h=%d: %s", data.ticker, horizon, exc)
            return None

    def _lstm_proba(self, data: EnrichedData, horizon: int) -> np.ndarray | None:
        try:
            trainer = LSTMTrainer(self._model_dir, horizon=horizon,
                                  seq_len=self._seq_len, hidden=self._hidden)
            return trainer.predict_proba(data)
        except Exception as exc:
            logger.debug("LSTM proba failed for %s h=%d: %s", data.ticker, horizon, exc)
            return None

    def _rf_proba(self, data: EnrichedData, horizon: int) -> np.ndarray | None:
        try:
            from prediction.rf_trainer import RandomForestTrainer
            trainer = RandomForestTrainer(self._model_dir, horizon=horizon)
            model   = trainer.load(data.ticker)
            if model is None:
                return None
            feats = trainer.extract_features(data)
            if feats is None:
                return None
            return model.predict_proba(feats)[0].astype(np.float32)
        except Exception as exc:
            logger.debug("RF proba failed for %s h=%d: %s", data.ticker, horizon, exc)
            return None
