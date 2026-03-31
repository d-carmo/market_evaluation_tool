"""Tests for prediction/lstm_trainer.py and prediction/lstm_predictor.py."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_analysis.conftest import make_enriched


torch = pytest.importorskip("torch", reason="torch not installed")


def _make_mixed(n: int = 300) -> object:
    """Make EnrichedData with genuinely mixed UP/DOWN/FLAT labels."""
    import pandas as pd
    rng = np.random.default_rng(42)
    # Oscillating + noise → produces diverse labels over 3-day horizon
    t = np.linspace(0, 10 * np.pi, n)
    close = 100.0 + 5 * np.sin(t) + rng.normal(0, 0.5, n)
    enriched = make_enriched(n=n, close_trend="flat")
    # Overwrite close with oscillating series
    df = enriched.df.copy()
    df["close"] = close
    df["open"]  = close * 0.999
    df["high"]  = close * 1.01
    df["low"]   = close * 0.99
    from data.models import EnrichedData
    return EnrichedData(ticker=enriched.ticker, market_type=enriched.market_type,
                        df=df, feature_names=enriched.feature_names)


class TestLSTMTrainer:

    def test_train_and_save_returns_true_with_enough_data(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        data = _make_mixed()
        trainer = LSTMTrainer(
            model_dir=tmp_path, horizon=3,
            seq_len=30, epochs=2, batch_size=16,
        )
        success = trainer.train_and_save(data)
        assert success is True

    def test_model_file_created(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        data = _make_mixed()
        trainer = LSTMTrainer(
            model_dir=tmp_path, horizon=3, seq_len=30, epochs=2,
        )
        trainer.train_and_save(data)
        model_files = list(tmp_path.glob("*_lstm.pt"))
        assert len(model_files) == 1

    def test_sha256_sidecar_created(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        data = _make_mixed()
        trainer = LSTMTrainer(model_dir=tmp_path, horizon=3, seq_len=30, epochs=2)
        trainer.train_and_save(data)
        sidecars = list(tmp_path.glob("*.sha256"))
        assert len(sidecars) >= 1

    def test_load_returns_model_and_scaler(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        data = _make_mixed()
        trainer = LSTMTrainer(model_dir=tmp_path, horizon=3, seq_len=30, epochs=2)
        trainer.train_and_save(data)
        result = trainer.load(data.ticker)
        assert result is not None
        model, scaler = result
        assert "mean" in scaler
        assert "std" in scaler

    def test_load_returns_none_when_no_model(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        trainer = LSTMTrainer(model_dir=tmp_path, horizon=3)
        assert trainer.load("NONEXISTENT") is None

    def test_predict_proba_returns_shape_3(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        data = _make_mixed()
        trainer = LSTMTrainer(model_dir=tmp_path, horizon=3, seq_len=30, epochs=2)
        trainer.train_and_save(data)
        proba = trainer.predict_proba(data)
        assert proba is not None
        assert proba.shape == (3,)

    def test_predict_proba_sums_to_one(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        data = _make_mixed()
        trainer = LSTMTrainer(model_dir=tmp_path, horizon=3, seq_len=30, epochs=2)
        trainer.train_and_save(data)
        proba = trainer.predict_proba(data)
        assert proba is not None
        assert abs(proba.sum() - 1.0) < 1e-5

    def test_predict_proba_returns_none_when_no_model(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        data = make_enriched(n=200)
        trainer = LSTMTrainer(model_dir=tmp_path, horizon=3)
        result = trainer.predict_proba(data)
        assert result is None

    def test_train_returns_false_with_insufficient_data(self, tmp_path):
        from prediction.lstm_trainer import LSTMTrainer
        data = make_enriched(n=40)   # fewer than seq_len + labels
        trainer = LSTMTrainer(model_dir=tmp_path, horizon=3, seq_len=60, epochs=2)
        success = trainer.train_and_save(data)
        assert success is False


class TestLSTMPredictor:

    def test_falls_back_to_statistical_when_no_model(self, tmp_path):
        from prediction.lstm_predictor import LSTMPredictor
        data = make_enriched(n=100, close_trend="up")
        from analysis.signals import compute_signals
        from analysis.scoring import build_composite
        from analysis.trend import detect_trend
        from config import SignalWeights
        trend = detect_trend(data)
        signals = compute_signals(data, SignalWeights())
        score = build_composite(data.ticker, signals, trend)
        predictor = LSTMPredictor(model_dir=tmp_path)
        result = predictor.predict(data, score, horizons=[3, 7])
        assert result.model == "statistical"

    def test_uses_lstm_when_model_exists(self, tmp_path):
        from prediction.lstm_predictor import LSTMPredictor
        from prediction.lstm_trainer import LSTMTrainer
        from analysis.signals import compute_signals
        from analysis.scoring import build_composite
        from analysis.trend import detect_trend
        from config import SignalWeights
        data = _make_mixed()
        # Train models for both horizons
        for h in [3, 7]:
            t = LSTMTrainer(model_dir=tmp_path, horizon=h, seq_len=30, epochs=2)
            t.train_and_save(data)
        trend = detect_trend(data)
        signals = compute_signals(data, SignalWeights())
        score = build_composite(data.ticker, signals, trend)
        predictor = LSTMPredictor(model_dir=tmp_path, seq_len=30)
        result = predictor.predict(data, score, horizons=[3, 7])
        assert result.model == "lstm"
        assert result.short_forecast.horizon_days == 3
        assert result.long_forecast.horizon_days == 7

    def test_mid_consistent_with_direction(self, tmp_path):
        """mid price must move in the direction the LSTM predicted."""
        from prediction.lstm_predictor import LSTMPredictor
        from prediction.lstm_trainer import LSTMTrainer
        from analysis.signals import compute_signals
        from analysis.scoring import build_composite
        from analysis.trend import detect_trend
        from config import SignalWeights
        data = _make_mixed()
        for h in [3, 7]:
            t = LSTMTrainer(model_dir=tmp_path, horizon=h, seq_len=30, epochs=2)
            t.train_and_save(data)
        trend = detect_trend(data)
        signals = compute_signals(data, SignalWeights())
        score = build_composite(data.ticker, signals, trend)
        predictor = LSTMPredictor(model_dir=tmp_path, seq_len=30)
        result = predictor.predict(data, score, horizons=[3, 7])
        current = result.current_price
        for fc in [result.short_forecast, result.long_forecast]:
            if fc.direction == "UP":
                assert fc.mid > current * 0.99, \
                    f"UP direction but mid={fc.mid} <= current={current}"
            elif fc.direction == "DOWN":
                assert fc.mid < current * 1.01, \
                    f"DOWN direction but mid={fc.mid} >= current={current}"
