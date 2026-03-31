import numpy as np
import pandas as pd
import pytest

from features.statistical import add_autocorrelation, add_return_features, add_rolling_stats


def test_return_features_columns(sample_df):
    out = add_return_features(sample_df)
    assert {"returns_1d", "returns_5d", "log_returns"}.issubset(set(out.columns))


def test_returns_1d_values(sample_df):
    out = add_return_features(sample_df)
    expected = sample_df["close"].pct_change(1)
    pd.testing.assert_series_equal(out["returns_1d"], expected, check_names=False)


def test_rolling_stats_columns(sample_df):
    df_with_returns = add_return_features(sample_df)
    out = add_rolling_stats(df_with_returns, windows=[20])
    expected = {"rolling_mean_20", "rolling_std_20", "rolling_skew_20",
                "rolling_kurt_20", "z_score_20", "realized_vol_20"}
    assert expected.issubset(set(out.columns))


def test_z_score_formula(sample_df):
    df_r = add_return_features(sample_df)
    out = add_rolling_stats(df_r, windows=[20])
    valid = out[["close", "rolling_mean_20", "rolling_std_20", "z_score_20"]].dropna()
    expected_z = (valid["close"] - valid["rolling_mean_20"]) / valid["rolling_std_20"]
    pd.testing.assert_series_equal(valid["z_score_20"], expected_z, check_names=False)


def test_autocorr_columns(sample_df):
    df_r = add_return_features(sample_df)
    out = add_autocorrelation(df_r, lags=[1, 5])
    assert {"autocorr_lag1", "autocorr_lag5"}.issubset(set(out.columns))


def test_autocorr_range(sample_df):
    df_r = add_return_features(sample_df)
    out = add_autocorrelation(df_r, lags=[1, 5])
    for col in ("autocorr_lag1", "autocorr_lag5"):
        valid = out[col].dropna()
        assert ((valid >= -1.0) & (valid <= 1.0)).all()
