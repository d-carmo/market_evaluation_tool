import logging

import numpy as np

from data.exceptions import AnalysisError
from data.models import Direction, EnrichedData, TrendResult

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"close", "ema_12", "sma_50", "sma_200", "adx_14"}
_SLOPE_WINDOW = 20


def detect_trend(data: EnrichedData) -> TrendResult:
    """Classify current market regime as BULLISH, BEARISH, or SIDEWAYS.

    Uses the last row for indicator values and last 20 closes for slope.
    Raises AnalysisError if any required column is missing or all-NaN.
    """
    df = data.df
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise AnalysisError(
            f"{data.ticker}: missing required columns for trend detection: {missing}"
        )

    last = df.iloc[-1]

    # Last non-NaN values for indicators
    def _last_valid(col: str) -> float:
        s = df[col].dropna()
        if s.empty:
            raise AnalysisError(f"{data.ticker}: column '{col}' is all-NaN")
        return float(s.iloc[-1])

    close = _last_valid("close")
    ema_12 = _last_valid("ema_12")
    sma_50 = _last_valid("sma_50")
    sma_200 = _last_valid("sma_200")
    adx = _last_valid("adx_14")

    # Linear regression slope over last N closes, normalized by mean price
    closes = df["close"].dropna().values[-_SLOPE_WINDOW:]
    if len(closes) < 2:
        raise AnalysisError(f"{data.ticker}: insufficient non-NaN close values for slope")
    x = np.arange(len(closes))
    coeffs = np.polyfit(x, closes, 1)
    slope = float(coeffs[0]) / float(np.mean(closes))

    # MA stack alignment
    ma_aligned = (close > ema_12 > sma_50 > sma_200) or (close < ema_12 < sma_50 < sma_200)

    # Direction classification
    direction: Direction
    if slope > 0.001 and adx > 20:
        direction = "BULLISH"
    elif slope < -0.001 and adx > 20:
        direction = "BEARISH"
    else:
        direction = "SIDEWAYS"

    strength = min(adx / 50.0, 1.0)

    return TrendResult(
        ticker=data.ticker,
        direction=direction,
        strength=strength,
        ma_aligned=ma_aligned,
        slope=slope,
        adx=adx,
    )
