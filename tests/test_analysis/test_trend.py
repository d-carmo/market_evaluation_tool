import pandas as pd
import pytest

from analysis.trend import detect_trend
from data.exceptions import AnalysisError
from data.models import EnrichedData
from tests.test_analysis.conftest import make_enriched


def test_detect_trend_bullish(bullish_data):
    result = detect_trend(bullish_data)
    assert result.direction == "BULLISH"


def test_detect_trend_bearish(bearish_data):
    result = detect_trend(bearish_data)
    assert result.direction == "BEARISH"


def test_strength_in_range(bullish_data):
    result = detect_trend(bullish_data)
    assert 0.0 <= result.strength <= 1.0


def test_sideways_low_adx(sideways_data):
    result = detect_trend(sideways_data)
    # Low ADX → SIDEWAYS regardless of small slope
    assert result.direction == "SIDEWAYS"


def test_ticker_preserved(bullish_data):
    result = detect_trend(bullish_data)
    assert result.ticker == bullish_data.ticker


def test_missing_column_raises_analysis_error():
    data = make_enriched()
    df = data.df.drop(columns=["adx_14"])
    bad_data = EnrichedData(
        ticker=data.ticker, market_type=data.market_type,
        df=df, feature_names=data.feature_names,
    )
    with pytest.raises(AnalysisError):
        detect_trend(bad_data)
