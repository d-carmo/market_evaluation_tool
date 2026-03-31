"""Tests for prediction/ensemble_predictor.py — EnsemblePredictor."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.test_analysis.conftest import make_enriched


class TestEnsemblePredictorBlends:

    def test_defaults_when_no_db(self, tmp_path):
        from prediction.ensemble_predictor import (
            EnsemblePredictor,
            _DEFAULT_XGB_BLEND,
            _DEFAULT_LSTM_BLEND,
        )
        pred = EnsemblePredictor(model_dir=tmp_path)
        xgb, lstm = pred._get_blends("AAPL")
        assert xgb == _DEFAULT_XGB_BLEND
        assert lstm == _DEFAULT_LSTM_BLEND

    def test_reads_blends_from_db(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor

        fake_row = MagicMock()
        fake_row.xgb_blend = 0.7
        fake_row.lstm_blend = 0.3

        fake_repo = MagicMock()
        fake_repo.get.return_value = fake_row

        pred = EnsemblePredictor(model_dir=tmp_path)
        pred._db = MagicMock()

        with patch("storage.repository.TickerWeightsRepository",
                   return_value=fake_repo):
            xgb, lstm = pred._get_blends("AAPL")

        assert xgb == pytest.approx(0.7)
        assert lstm == pytest.approx(0.3)

    def test_falls_back_to_defaults_when_db_row_missing(self, tmp_path):
        from prediction.ensemble_predictor import (
            EnsemblePredictor,
            _DEFAULT_XGB_BLEND,
            _DEFAULT_LSTM_BLEND,
        )
        fake_repo = MagicMock()
        fake_repo.get.return_value = None

        pred = EnsemblePredictor(model_dir=tmp_path)
        pred._db = MagicMock()

        with patch("storage.repository.TickerWeightsRepository",
                   return_value=fake_repo):
            xgb, lstm = pred._get_blends("AAPL")

        assert xgb == _DEFAULT_XGB_BLEND
        assert lstm == _DEFAULT_LSTM_BLEND

    def test_falls_back_to_defaults_on_db_exception(self, tmp_path):
        from prediction.ensemble_predictor import (
            EnsemblePredictor,
            _DEFAULT_XGB_BLEND,
            _DEFAULT_LSTM_BLEND,
        )
        pred = EnsemblePredictor(model_dir=tmp_path)
        pred._db = MagicMock()

        with patch("storage.repository.TickerWeightsRepository",
                   side_effect=Exception("db error")):
            xgb, lstm = pred._get_blends("AAPL")

        assert xgb == _DEFAULT_XGB_BLEND
        assert lstm == _DEFAULT_LSTM_BLEND

    def test_set_db_session_stores_db(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        pred = EnsemblePredictor(model_dir=tmp_path)
        fake_db = object()
        pred.set_db_session(fake_db)
        assert pred._db is fake_db


class TestEnsemblePredictorProba:

    def test_xgb_proba_returns_none_when_no_model(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        pred = EnsemblePredictor(model_dir=tmp_path)
        data = make_enriched(n=100)
        result = pred._xgb_proba(data, horizon=3)
        assert result is None

    def test_lstm_proba_returns_none_when_no_model(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        pred = EnsemblePredictor(model_dir=tmp_path)
        data = make_enriched(n=100)
        result = pred._lstm_proba(data, horizon=3)
        assert result is None

    def test_xgb_proba_handles_exception_gracefully(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        pred = EnsemblePredictor(model_dir=tmp_path)
        data = make_enriched(n=100)
        with patch("prediction.ensemble_predictor.XGBoostTrainer",
                   side_effect=Exception("boom")):
            result = pred._xgb_proba(data, horizon=3)
        assert result is None

    def test_lstm_proba_handles_exception_gracefully(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        pred = EnsemblePredictor(model_dir=tmp_path)
        data = make_enriched(n=100)
        with patch("prediction.ensemble_predictor.LSTMTrainer",
                   side_effect=Exception("boom")):
            result = pred._lstm_proba(data, horizon=3)
        assert result is None


class TestEnsemblePredictorFallbackChain:

    def _setup(self, tmp_path, data):
        from analysis.signals import compute_signals
        from analysis.scoring import build_composite
        from analysis.trend import detect_trend
        from config import SignalWeights
        trend = detect_trend(data)
        signals = compute_signals(data, SignalWeights())
        score = build_composite(data.ticker, signals, trend)
        return score

    def test_falls_back_to_statistical_when_both_none(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        data = make_enriched(n=100, close_trend="up")
        score = self._setup(tmp_path, data)
        pred = EnsemblePredictor(model_dir=tmp_path)
        result = pred.predict(data, score, horizons=[3, 7])
        assert result.model == "statistical"

    def test_uses_xgb_only_when_lstm_none(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        data = make_enriched(n=100, close_trend="up")
        score = self._setup(tmp_path, data)

        fake_proba = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        pred = EnsemblePredictor(model_dir=tmp_path)

        with patch.object(pred, "_xgb_proba", return_value=fake_proba), \
             patch.object(pred, "_lstm_proba", return_value=None):
            result = pred.predict(data, score, horizons=[3, 7])

        assert result.model == "ml"

    def test_uses_lstm_only_when_xgb_none(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        data = make_enriched(n=100, close_trend="up")
        score = self._setup(tmp_path, data)

        fake_proba = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        pred = EnsemblePredictor(model_dir=tmp_path)

        with patch.object(pred, "_xgb_proba", return_value=None), \
             patch.object(pred, "_lstm_proba", return_value=fake_proba):
            result = pred.predict(data, score, horizons=[3, 7])

        assert result.model == "lstm"

    def test_uses_ensemble_when_both_available(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        data = make_enriched(n=100, close_trend="up")
        score = self._setup(tmp_path, data)

        xgb_proba  = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        lstm_proba = np.array([0.15, 0.25, 0.6], dtype=np.float32)
        pred = EnsemblePredictor(model_dir=tmp_path)

        with patch.object(pred, "_xgb_proba", return_value=xgb_proba), \
             patch.object(pred, "_lstm_proba", return_value=lstm_proba):
            result = pred.predict(data, score, horizons=[3, 7])

        assert result.model == "ensemble"

    def test_ensemble_blend_direction_up(self, tmp_path):
        """When blended probas strongly suggest UP, direction should be UP."""
        from prediction.ensemble_predictor import EnsemblePredictor
        data = make_enriched(n=100, close_trend="up")
        score = self._setup(tmp_path, data)

        # Index 2 = UP dominates
        xgb_proba  = np.array([0.05, 0.05, 0.90], dtype=np.float32)
        lstm_proba = np.array([0.05, 0.05, 0.90], dtype=np.float32)
        pred = EnsemblePredictor(model_dir=tmp_path)

        with patch.object(pred, "_xgb_proba", return_value=xgb_proba), \
             patch.object(pred, "_lstm_proba", return_value=lstm_proba):
            result = pred.predict(data, score, horizons=[3, 7])

        assert result.short_forecast.direction == "UP"
        assert result.short_forecast.mid > result.current_price * 0.99

    def test_ensemble_blend_direction_down(self, tmp_path):
        """When blended probas strongly suggest DOWN, direction should be DOWN."""
        from prediction.ensemble_predictor import EnsemblePredictor
        data = make_enriched(n=100, close_trend="up")
        score = self._setup(tmp_path, data)

        # Index 0 = DOWN dominates
        xgb_proba  = np.array([0.90, 0.05, 0.05], dtype=np.float32)
        lstm_proba = np.array([0.90, 0.05, 0.05], dtype=np.float32)
        pred = EnsemblePredictor(model_dir=tmp_path)

        with patch.object(pred, "_xgb_proba", return_value=xgb_proba), \
             patch.object(pred, "_lstm_proba", return_value=lstm_proba):
            result = pred.predict(data, score, horizons=[3, 7])

        assert result.short_forecast.direction == "DOWN"
        assert result.short_forecast.mid < result.current_price * 1.01

    def test_result_has_correct_horizons(self, tmp_path):
        from prediction.ensemble_predictor import EnsemblePredictor
        data = make_enriched(n=100, close_trend="up")
        score = self._setup(tmp_path, data)

        fake_proba = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        pred = EnsemblePredictor(model_dir=tmp_path)

        with patch.object(pred, "_xgb_proba", return_value=fake_proba), \
             patch.object(pred, "_lstm_proba", return_value=fake_proba):
            result = pred.predict(data, score, horizons=[3, 7])

        assert result.short_forecast.horizon_days == 3
        assert result.long_forecast.horizon_days == 7
