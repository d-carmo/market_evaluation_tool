import os
from dataclasses import dataclass

from dotenv import load_dotenv

from data.exceptions import ConfigError


@dataclass(frozen=True)
class SignalWeights:
    rsi: float = 1.0
    macd: float = 1.5
    trend: float = 2.0
    volume: float = 0.8
    bb: float = 1.0
    stoch: float = 0.8
    sentiment: float = 0.5   # 7th signal; 0 when Reddit disabled


@dataclass(frozen=True)
class AppConfig:
    historical_days: int
    cache_ttl_hours: int
    signal_weights: SignalWeights
    forecast_short: int
    forecast_long: int
    max_fetch_workers: int
    max_compute_workers: int
    db_path: str = "market_eval.db"
    model_dir: str = ".models"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    predictor_method: str = "ml"
    ml_label_threshold: float = 0.005
    retrain_min_records: int = 30
    accuracy_check_interval_hours: int = 1
    analysis_refresh_hours: int = 3
    bloomberg_enabled: bool = False
    bloomberg_host: str = "localhost"
    bloomberg_port: int = 8194
    # Reddit sentiment (Phase 1)
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "market_eval/2.0"
    sentiment_enabled: bool = False       # auto-enabled when credentials are present
    sentiment_ttl_hours: int = 1          # how long to cache Reddit post lists
    # LSTM / Ensemble (Phase 3)
    lstm_sequence_len: int = 60           # look-back bars for LSTM input
    lstm_hidden_units: int = 128
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    # Database backend — "sqlite" (default) or "mysql"
    db_type: str = "sqlite"
    database_url: str = ""          # full override; takes priority over all other db_* fields
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "market_eval"
    mysql_password: str = ""
    mysql_database: str = "market_eval"

    @property
    def db_url(self) -> str:
        """Effective SQLAlchemy connection URL for this configuration."""
        if self.database_url:
            return self.database_url
        if self.db_type == "mysql":
            return (
                f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
                f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
                "?charset=utf8mb4"
            )
        return f"sqlite:///{self.db_path}"


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError as e:
        raise ConfigError(f"{key} must be a float, got: {val!r}") from e


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as e:
        raise ConfigError(f"{key} must be an integer, got: {val!r}") from e


def load_config() -> AppConfig:
    """Load AppConfig from environment / .env. Tickers are stored in the database."""
    load_dotenv()

    weights = SignalWeights(
        rsi=_get_float("SIGNAL_WEIGHTS_RSI", 1.0),
        macd=_get_float("SIGNAL_WEIGHTS_MACD", 1.5),
        trend=_get_float("SIGNAL_WEIGHTS_TREND", 2.0),
        volume=_get_float("SIGNAL_WEIGHTS_VOLUME", 0.8),
        bb=_get_float("SIGNAL_WEIGHTS_BB", 1.0),
        stoch=_get_float("SIGNAL_WEIGHTS_STOCH", 0.8),
        sentiment=_get_float("SIGNAL_WEIGHTS_SENTIMENT", 0.5),
    )

    reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    sentiment_enabled = bool(reddit_client_id and reddit_client_secret)

    return AppConfig(
        historical_days=_get_int("HISTORICAL_DAYS", 365),
        cache_ttl_hours=_get_int("CACHE_TTL_HOURS", 3),
        signal_weights=weights,
        forecast_short=_get_int("FORECAST_SHORT", 3),
        forecast_long=_get_int("FORECAST_LONG", 7),
        max_fetch_workers=_get_int("MAX_FETCH_WORKERS", 8),
        max_compute_workers=_get_int("MAX_COMPUTE_WORKERS", 4),
        db_path=os.getenv("DB_PATH", "market_eval.db"),
        model_dir=os.getenv("MODEL_DIR", ".models"),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=_get_int("API_PORT", 8000),
        predictor_method=os.getenv("PREDICTOR_METHOD", "ensemble"),
        ml_label_threshold=_get_float("ML_LABEL_THRESHOLD", 0.005),
        retrain_min_records=_get_int("RETRAIN_MIN_RECORDS", 30),
        accuracy_check_interval_hours=_get_int("ACCURACY_CHECK_INTERVAL_HOURS", 1),
        analysis_refresh_hours=_get_int("ANALYSIS_REFRESH_HOURS", 3),
        bloomberg_enabled=os.getenv("BLOOMBERG_ENABLED", "false").strip().lower()
            in ("1", "true", "yes"),
        bloomberg_host=os.getenv("BLOOMBERG_HOST", "localhost"),
        bloomberg_port=_get_int("BLOOMBERG_PORT", 8194),
        db_type=os.getenv("DB_TYPE", "sqlite").strip().lower(),
        database_url=os.getenv("DATABASE_URL", "").strip(),
        mysql_host=os.getenv("MYSQL_HOST", "localhost"),
        mysql_port=_get_int("MYSQL_PORT", 3306),
        mysql_user=os.getenv("MYSQL_USER", "market_eval"),
        mysql_password=os.getenv("MYSQL_PASSWORD", ""),
        mysql_database=os.getenv("MYSQL_DATABASE", "market_eval"),
        # Reddit / sentiment
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "market_eval/2.0"),
        sentiment_enabled=sentiment_enabled,
        sentiment_ttl_hours=_get_int("SENTIMENT_TTL_HOURS", 1),
        # LSTM
        lstm_sequence_len=_get_int("LSTM_SEQUENCE_LEN", 60),
        lstm_hidden_units=_get_int("LSTM_HIDDEN_UNITS", 128),
        lstm_epochs=_get_int("LSTM_EPOCHS", 50),
        lstm_batch_size=_get_int("LSTM_BATCH_SIZE", 32),
    )
