"""Tests for prediction/__init__.py — get_predictor() factory."""
from __future__ import annotations

import pytest


class TestGetPredictor:

    def test_statistical_returns_statistical_predictor(self):
        from prediction import get_predictor
        from prediction.statistical import StatisticalPredictor
        p = get_predictor("statistical")
        assert isinstance(p, StatisticalPredictor)

    def test_ml_returns_ml_predictor(self):
        from prediction import get_predictor
        from prediction.ml import MLPredictor
        p = get_predictor("ml")
        assert isinstance(p, MLPredictor)

    def test_lstm_returns_lstm_predictor(self):
        from prediction import get_predictor
        from prediction.lstm_predictor import LSTMPredictor
        p = get_predictor("lstm")
        assert isinstance(p, LSTMPredictor)

    def test_ensemble_returns_ensemble_predictor(self):
        from prediction import get_predictor
        from prediction.ensemble_predictor import EnsemblePredictor
        p = get_predictor("ensemble")
        assert isinstance(p, EnsemblePredictor)

    def test_unknown_method_raises_value_error(self):
        from prediction import get_predictor
        with pytest.raises(ValueError, match="Unknown predictor method"):
            get_predictor("magic_beans")

    def test_default_method_is_ensemble(self):
        from prediction import get_predictor
        from prediction.ensemble_predictor import EnsemblePredictor
        p = get_predictor()
        assert isinstance(p, EnsemblePredictor)

    def test_db_injected_into_ensemble(self):
        from prediction import get_predictor
        from prediction.ensemble_predictor import EnsemblePredictor
        fake_db = object()
        p = get_predictor("ensemble", db=fake_db)
        assert isinstance(p, EnsemblePredictor)
        assert p._db is fake_db

    def test_model_dir_passed_to_ml(self, tmp_path):
        from prediction import get_predictor
        from prediction.ml import MLPredictor
        p = get_predictor("ml", model_dir=tmp_path)
        assert isinstance(p, MLPredictor)
        assert p._model_dir == tmp_path
