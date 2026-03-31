import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return-based features: daily/5-day returns and log returns."""
    result = df.copy()
    result["returns_1d"] = result["close"].pct_change(1)
    result["returns_5d"] = result["close"].pct_change(5)
    ratio = result["close"] / result["close"].shift(1)
    result["log_returns"] = np.log(ratio.clip(lower=1e-10))
    return result


def add_rolling_stats(df: pd.DataFrame, windows: list[int] = [20]) -> pd.DataFrame:
    """Add rolling statistics for each window.

    Requires returns_1d and log_returns columns to exist in df.
    For each window w, adds:
        rolling_mean_w, rolling_std_w, rolling_skew_w, rolling_kurt_w,
        z_score_w, realized_vol_w
    """
    result = df.copy()
    for w in windows:
        rolling_close = result["close"].rolling(w)
        rolling_ret = result["returns_1d"].rolling(w)
        rolling_log = result["log_returns"].rolling(w)

        mean = rolling_close.mean()
        std = rolling_close.std()

        result[f"rolling_mean_{w}"] = mean
        result[f"rolling_std_{w}"] = std
        result[f"rolling_skew_{w}"] = rolling_ret.skew()
        result[f"rolling_kurt_{w}"] = rolling_ret.kurt()
        result[f"z_score_{w}"] = (result["close"] - mean) / std
        result[f"realized_vol_{w}"] = rolling_log.std() * np.sqrt(252)

    return result


def add_autocorrelation(df: pd.DataFrame, lags: list[int] = [1, 5]) -> pd.DataFrame:
    """Add rolling autocorrelation of log returns for each lag.

    Requires log_returns column to exist in df.
    For each lag n, adds: autocorr_lag{n}
    """
    result = df.copy()
    for lag in lags:
        result[f"autocorr_lag{lag}"] = (
            result["log_returns"]
            .rolling(30)
            .apply(lambda x: x.autocorr(lag=lag), raw=False)
        )
    return result
