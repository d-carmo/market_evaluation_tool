import pandas as pd
import pytest

from data.exceptions import NormalizationError
from data.normalizer import normalize


def _make_raw(columns: dict) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame(columns, index=idx)


def test_normalize_renames_capitalized_columns():
    df = _make_raw({"Open": [1.0]*5, "High": [2.0]*5, "Low": [0.5]*5,
                    "Close": [1.5]*5, "Volume": [1000.0]*5})
    out = normalize(df, source="test", ticker="X")
    assert set(out.columns) == {"open", "high", "low", "close", "volume"}


def test_normalize_fills_volume_nan():
    df = _make_raw({"open": [1.0]*5, "high": [2.0]*5, "low": [0.5]*5,
                    "close": [1.5]*5, "volume": [None, 100.0, None, 200.0, None]})
    out = normalize(df, source="test", ticker="X")
    assert out["volume"].isna().sum() == 0
    assert out["volume"].iloc[0] == 0.0


def test_normalize_drops_close_nan():
    df = _make_raw({"open": [1.0]*5, "high": [2.0]*5, "low": [0.5]*5,
                    "close": [None, 1.5, None, 1.5, 1.5], "volume": [100.0]*5})
    out = normalize(df, source="test", ticker="X")
    assert len(out) == 3
    assert out["close"].isna().sum() == 0


def test_normalize_raises_on_missing_close():
    df = _make_raw({"open": [1.0]*5, "high": [2.0]*5, "low": [0.5]*5, "volume": [100.0]*5})
    with pytest.raises(NormalizationError):
        normalize(df, source="test", ticker="X")


def test_normalize_does_not_mutate_input():
    df = _make_raw({"Open": [1.0]*5, "High": [2.0]*5, "Low": [0.5]*5,
                    "Close": [1.5]*5, "Volume": [100.0]*5})
    original_cols = list(df.columns)
    normalize(df, source="test", ticker="X")
    assert list(df.columns) == original_cols


def test_normalize_utc_index():
    df = _make_raw({"close": [1.0]*5, "open": [1.0]*5, "high": [1.0]*5,
                    "low": [1.0]*5, "volume": [0.0]*5})
    out = normalize(df, source="test", ticker="X")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"


def test_normalize_sorted_ascending():
    idx = pd.to_datetime(["2023-01-05", "2023-01-03", "2023-01-04"])
    df = pd.DataFrame({"close": [3.0, 1.0, 2.0], "open": [3.0, 1.0, 2.0],
                       "high": [3.0, 1.0, 2.0], "low": [3.0, 1.0, 2.0],
                       "volume": [0.0, 0.0, 0.0]}, index=idx)
    out = normalize(df, source="test", ticker="X")
    assert list(out["close"]) == [1.0, 2.0, 3.0]
