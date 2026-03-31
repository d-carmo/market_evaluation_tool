from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data.exceptions import PredictionError
from data.models import CompositeScore, EnrichedData, PredictionResult, Signal, TrendResult
from prediction.statistical import StatisticalPredictor
from tests.test_analysis.conftest import make_enriched


def _make_score(ticker: str = "TEST") -> CompositeScore:
    trend = TrendResult(ticker=ticker, direction="BULLISH", strength=0.6,
                        ma_aligned=True, slope=0.002, adx=30.0)
    return CompositeScore(
        ticker=ticker, score=0.5, action="BUY", confidence=0.8,
        signals=[], trend=trend,
    )


def test_predict_returns_prediction_result():
    data = make_enriched(n=100, close_trend="up")
    score = _make_score()
    result = StatisticalPredictor().predict(data, score, horizons=[3, 7])
    assert isinstance(result, PredictionResult)


def test_model_field():
    data = make_enriched(n=100, close_trend="up")
    result = StatisticalPredictor().predict(data, _make_score(), horizons=[3, 7])
    assert result.model == "statistical"


def test_low_lt_mid_lt_high_short():
    data = make_enriched(n=100, close_trend="up")
    result = StatisticalPredictor().predict(data, _make_score(), horizons=[3, 7])
    fc = result.short_forecast
    assert fc.low < fc.mid < fc.high


def test_low_lt_mid_lt_high_long():
    data = make_enriched(n=100, close_trend="up")
    result = StatisticalPredictor().predict(data, _make_score(), horizons=[3, 7])
    fc = result.long_forecast
    assert fc.low < fc.mid < fc.high


def test_uptrend_direction_is_up():
    """A clearly rising series should project UP."""
    data = make_enriched(n=100, close_trend="up")
    result = StatisticalPredictor().predict(data, _make_score(), horizons=[3, 7])
    assert result.short_forecast.direction == "UP"


def test_raises_on_short_df():
    data = make_enriched(n=20, close_trend="up")
    with pytest.raises(PredictionError):
        StatisticalPredictor().predict(data, _make_score(), horizons=[3, 7])


def test_current_price_matches_last_close():
    data = make_enriched(n=100, close_trend="up")
    result = StatisticalPredictor().predict(data, _make_score(), horizons=[3, 7])
    last_close = round(float(data.df["close"].dropna().iloc[-1]), 4)
    assert result.current_price == last_close
