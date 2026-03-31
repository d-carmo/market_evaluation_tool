class MarketToolError(Exception):
    """Base exception for all tool errors."""


class ConfigError(MarketToolError):
    """Missing or invalid configuration."""


class FetchError(MarketToolError):
    """Network or API failure during data fetch."""


class NormalizationError(MarketToolError):
    """Cannot map source columns to canonical schema."""


class FeatureError(MarketToolError):
    """Feature engineering failure (e.g. insufficient rows)."""


class AnalysisError(MarketToolError):
    """Indicator computation or column lookup failure."""


class PredictionError(MarketToolError):
    """Insufficient data or missing columns for forecast."""


class MLError(MarketToolError):
    """XGBoost training or inference failure."""


class DatabaseError(MarketToolError):
    """SQLAlchemy operation failure."""
