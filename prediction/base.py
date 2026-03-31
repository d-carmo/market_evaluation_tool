from abc import ABC, abstractmethod

from data.models import CompositeScore, EnrichedData, PredictionResult


class AbstractPredictor(ABC):
    """Contract every predictor implementation must satisfy."""

    @abstractmethod
    def predict(
        self,
        data: EnrichedData,
        score: CompositeScore,
        horizons: list[int],
    ) -> PredictionResult:
        """Generate price forecasts for each horizon (days into the future).

        Args:
            data:     Enriched ticker data with all feature columns.
            score:    Composite signal score providing directional bias.
            horizons: List of forecast horizons in days, e.g. [3, 7].

        Returns:
            PredictionResult with short_forecast (horizons[0]) and
            long_forecast (horizons[1]).

        Raises:
            PredictionError on insufficient data or missing columns.
        """
