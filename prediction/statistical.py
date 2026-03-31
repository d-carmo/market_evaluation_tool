import logging

import numpy as np

from data.exceptions import PredictionError
from data.models import (
    CompositeScore,
    EnrichedData,
    PriceForecast,
    PriceDir,
    PredictionResult,
)
from prediction.base import AbstractPredictor

logger = logging.getLogger(__name__)

_MIN_ROWS = 30
_REQUIRED_COLS = {"close", "atr_14", "rsi_14"}


class StatisticalPredictor(AbstractPredictor):
    """v1 prediction strategy using linear regression + ATR bands + momentum adjustment.

    Algorithm per horizon h:
    1. Fit linear regression on last 30 closes.
    2. Project mid price: intercept + slope * (30 + h).
    3. Band width = last ATR * multiplier (1.5 for short, 2.5 for long).
    4. Momentum adjustment: (rsi - 50) / 50 * atr * 0.3 shifts mid.
    5. Direction: UP / DOWN / FLAT based on mid vs current price (±0.5% threshold).
    6. Confidence: 1 - normalized residual std from regression fit.
    """

    def predict(
        self,
        data: EnrichedData,
        score: CompositeScore,
        horizons: list[int],
    ) -> PredictionResult:
        df = data.df
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise PredictionError(
                f"{data.ticker}: missing columns for prediction: {missing}"
            )

        close_series = df["close"].dropna()
        if len(close_series) < _MIN_ROWS:
            raise PredictionError(
                f"{data.ticker}: need ≥{_MIN_ROWS} rows, got {len(close_series)}"
            )

        closes = close_series.values[-_MIN_ROWS:]
        current_price = float(closes[-1])
        x = np.arange(_MIN_ROWS)
        coeffs = np.polyfit(x, closes, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

        # Residual std for confidence
        predicted = intercept + slope * x
        residuals_std = float(np.std(closes - predicted))
        confidence = max(0.0, 1.0 - min(residuals_std / current_price, 1.0))

        # Indicator values for adjustment
        atr = float(df["atr_14"].dropna().iloc[-1]) if not df["atr_14"].dropna().empty else 0.0
        rsi = float(df["rsi_14"].dropna().iloc[-1]) if not df["rsi_14"].dropna().empty else 50.0
        momentum_adj = (rsi - 50) / 50.0 * atr * 0.3

        forecasts: list[PriceForecast] = []
        for h in horizons:
            multiplier = 1.5 if h <= 3 else 2.5
            mid_raw = intercept + slope * (_MIN_ROWS + h)
            mid = mid_raw + momentum_adj
            band = atr * multiplier
            low = mid - band
            high = mid + band

            # Clamp low to a small positive value
            low = max(low, current_price * 0.01)

            if mid > current_price * 1.005:
                direction: PriceDir = "UP"
            elif mid < current_price * 0.995:
                direction = "DOWN"
            else:
                direction = "FLAT"

            forecasts.append(
                PriceForecast(
                    horizon_days=h,
                    low=round(low, 4),
                    mid=round(mid, 4),
                    high=round(high, 4),
                    direction=direction,
                    confidence=round(confidence, 4),
                )
            )

        # Ensure exactly two forecasts (short, long)
        if len(forecasts) < 2:
            forecasts = forecasts + [forecasts[-1]] * (2 - len(forecasts))

        return PredictionResult(
            ticker=data.ticker,
            current_price=round(current_price, 4),
            short_forecast=forecasts[0],
            long_forecast=forecasts[1],
            model="statistical",
        )
