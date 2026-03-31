# Market Evaluation Tool — Internal API Specification

All inter-module contracts are defined here. Only import from a module's public surface
(`__init__.py` or the definitions below). No module should import another module's internals.

---

## Core Data Models (`data/models.py`)

All shared data structures are **frozen dataclasses**.

```python
MarketType = Literal["stock", "crypto", "commodity"]
Direction  = Literal["BULLISH", "BEARISH", "SIDEWAYS"]
Action     = Literal["BUY", "HOLD", "SELL"]
SignalDir  = Literal[-1, 0, 1]    # -1=sell, 0=hold, 1=buy
PriceDir   = Literal["UP", "DOWN", "FLAT"]


@dataclass(frozen=True)
class MarketData:
    ticker: str; market_type: MarketType; fetched_at: datetime; source: str
    df: pd.DataFrame
    # df contract: DatetimeIndex UTC ascending, columns open/high/low/close/volume float64
    # volume = 0.0 when unavailable; never NaN


@dataclass(frozen=True)
class EnrichedData:
    ticker: str; market_type: MarketType
    df: pd.DataFrame   # MarketData columns + feature columns + optional sentiment columns
    feature_names: list[str]
    # Guaranteed feature columns (NaN only in warm-up rows, never in last 30):
    #   sma_20/50/200, ema_12/26, macd, macd_signal, macd_hist
    #   rsi_14, stoch_k/d, williams_r, bb_upper/mid/lower, atr_14, adx_14
    #   obv, vwap, mfi_14, returns_1d/5d, log_returns
    #   rolling_mean/std/skew/kurt_20, z_score_20, realized_vol_20
    #   autocorr_lag1/5
    # Optional (added by augment_with_sentiment if snapshot provided):
    #   sentiment_score, sentiment_velocity, bullish_ratio, engagement_score


@dataclass(frozen=True)
class SentimentSnapshot:
    """Social-media sentiment for one ticker over a look-back window."""
    ticker: str; window_hours: int; mention_count: int
    sentiment_score: float     # weighted avg sentiment [-1, 1]
    sentiment_velocity: float  # change vs previous window
    bullish_ratio: float       # fraction of clearly bullish texts [0, 1]
    engagement_score: float    # normalised engagement proxy
    fetched_at: datetime


@dataclass(frozen=True)
class Signal:
    name: str; value: SignalDir; raw: float; weight: float


@dataclass(frozen=True)
class CompositeScore:
    ticker: str; score: float  # [-1, 1]
    action: Action; confidence: float; signals: list[Signal]; trend: TrendResult


@dataclass(frozen=True)
class PriceForecast:
    horizon_days: int; low: float; mid: float; high: float
    direction: PriceDir; confidence: float


@dataclass(frozen=True)
class PredictionResult:
    ticker: str; current_price: float
    short_forecast: PriceForecast; long_forecast: PriceForecast
    model: str   # "statistical" | "ml" | "lstm" | "ensemble"


@dataclass(frozen=True)
class TickerReport:
    ticker: str; market_type: MarketType; score: CompositeScore
    prediction: PredictionResult; generated_at: datetime
```

---

## Config API (`config.py`)

```python
@dataclass(frozen=True)
class SignalWeights:
    rsi: float = 1.0; macd: float = 1.5; trend: float = 2.0
    volume: float = 0.8; bb: float = 1.0; stoch: float = 0.8
    sentiment: float = 0.5      # Phase 1; 0 when Reddit disabled

@dataclass(frozen=True)
class AppConfig:
    # Data
    historical_days: int; cache_ttl_hours: int
    max_fetch_workers: int; max_compute_workers: int
    # Prediction
    signal_weights: SignalWeights; forecast_short: int; forecast_long: int
    predictor_method: str        # "ensemble" by default
    ml_label_threshold: float; retrain_min_records: int
    # Paths
    db_path: str; model_dir: str
    # API
    api_host: str; api_port: int
    # Reddit sentiment (Phase 1)
    reddit_client_id: str; reddit_client_secret: str
    reddit_user_agent: str; sentiment_enabled: bool; sentiment_ttl_hours: int
    # LSTM (Phase 3)
    lstm_sequence_len: int; lstm_hidden_units: int
    lstm_epochs: int; lstm_batch_size: int
    # Database
    db_type: str; database_url: str
    mysql_host: str; mysql_port: int; mysql_user: str
    mysql_password: str; mysql_database: str
    @property
    def db_url(self) -> str: ...   # effective SQLAlchemy URL

def load_config() -> AppConfig:
    """Load from environment / .env. Raises ConfigError on bad values."""
```

---

## Data Layer

### `data/cache.py`

```python
class DataCache:
    def __init__(self, cache_dir: Path, ttl_hours: int): ...
    def get(self, ticker: str) -> MarketData | None: ...
    def set(self, data: MarketData) -> None: ...
    def is_valid(self, ticker: str) -> bool: ...
    def invalidate(self, ticker: str | None = None) -> None: ...
```

### `data/fetchers/base.py`

```python
class AbstractFetcher(ABC):
    @abstractmethod
    def fetch(self, ticker: str, days: int) -> MarketData: ...
    @abstractmethod
    def supports(self, ticker: str) -> bool: ...
```

### `data/fetchers/reddit.py`

```python
class RedditFetcher:
    def __init__(self, client_id, client_secret, user_agent,
                 cache_dir: Path | None = None, ttl_hours: int = 1): ...

    def fetch(self, ticker: str, market_type: str = "stock",
              limit: int = 100) -> list[tuple[str, float]]:
        """Return [(text, engagement_weight)] for recent posts mentioning ticker.
        Returns [] on missing credentials, network errors, or praw ImportError.
        engagement_weight = sqrt(upvote_ratio * max(num_comments, 1))
        """
```

### `data/__init__.py`

```python
def fetch_all(tickers: dict[str, MarketType], days: int,
              cache: DataCache, max_workers: int = 8) -> dict[str, MarketData]:
    """ThreadPoolExecutor fetch. Returns only successful tickers."""
```

---

## Sentiment Layer

### `sentiment/__init__.py`

```python
def get_sentiment_snapshot(
    ticker: str, market_type: str,
    reddit_client_id: str, reddit_client_secret: str,
    user_agent: str = "market_eval/2.0",
    cache_dir: Path | None = None, ttl_hours: int = 1,
    window_hours: int = 24, prev_score: float | None = None,
) -> SentimentSnapshot | None:
    """Fetch Reddit posts and score sentiment.
    Returns None when credentials empty or all steps fail.
    mention_count may be 0 when no posts found (still returns a snapshot).
    """
```

### `sentiment/scorer.py`

```python
def score_texts(texts: list[tuple[str, float]]) -> tuple[float, float, float]:
    """Score [(text, engagement_weight)] pairs.
    Returns (sentiment_score [-1,1], bullish_ratio [0,1], mean_engagement).
    Tries FinBERT first, falls back to VADER.
    Returns (0.0, 0.5, 0.0) when no backend is available.
    """
```

---

## Feature Layer

### `features/pipeline.py`

```python
def build_features(data: MarketData) -> EnrichedData:
    """Run technical + statistical features. Raises FeatureError if < 60 rows."""

def augment_with_sentiment(
    enriched: EnrichedData, snapshot: SentimentSnapshot | None
) -> EnrichedData:
    """Inject 4 sentiment columns into the last row of EnrichedData.df.
    All other rows get 0.0. Columns: sentiment_score, sentiment_velocity,
    bullish_ratio, engagement_score.
    Always adds the columns (0.0) even when snapshot is None, so feature
    vectors remain consistent between training and inference.
    """

def build_features_all(market_data: dict[str, MarketData],
                       max_workers: int = 4) -> dict[str, EnrichedData]:
    """ProcessPoolExecutor. Failed tickers logged and excluded."""
```

---

## Analysis Layer

### `analysis/signals.py`

```python
def rsi_signal(data: EnrichedData, weight: float) -> Signal: ...
def macd_cross_signal(data: EnrichedData, weight: float) -> Signal: ...
def bb_position_signal(data: EnrichedData, weight: float) -> Signal: ...
def ma_cross_signal(data: EnrichedData, weight: float) -> Signal: ...
def volume_trend_signal(data: EnrichedData, weight: float) -> Signal: ...
def stoch_signal(data: EnrichedData, weight: float) -> Signal: ...

def sentiment_signal(snapshot: SentimentSnapshot | None, weight: float) -> Signal:
    """score > 0.3 → BUY, < -0.3 → SELL, else HOLD.
    Returns HOLD when snapshot is None or mention_count < 5."""

def compute_signals(data: EnrichedData, weights: SignalWeights,
                    snapshot: SentimentSnapshot | None = None) -> list[Signal]:
    """Returns all 7 signals. sentiment_signal uses snapshot (may be None)."""
```

### `analysis/__init__.py`

```python
def get_ticker_weights(ticker: str, db: Session,
                       global_weights: SignalWeights) -> SignalWeights:
    """Load per-ticker learned weights from DB; fall back to global_weights."""

def analyze(data: EnrichedData, weights: SignalWeights,
            snapshot: SentimentSnapshot | None = None) -> CompositeScore:
    """detect_trend → compute_signals → build_composite."""

def analyze_all(enriched: dict[str, EnrichedData], weights: SignalWeights,
                max_workers: int = 4) -> dict[str, CompositeScore]:
    """ProcessPoolExecutor. snapshot=None for all (no live Reddit in parallel)."""
```

---

## Prediction Layer

### `prediction/base.py`

```python
class AbstractPredictor(ABC):
    @abstractmethod
    def predict(self, data: EnrichedData, score: CompositeScore,
                horizons: list[int]) -> PredictionResult: ...
```

### `prediction/__init__.py`

```python
def get_predictor(method: str = "ensemble",
                  model_dir: Path | None = None,
                  db=None) -> AbstractPredictor:
    """Factory for "statistical" | "ml" | "lstm" | "ensemble".
    db is passed to EnsemblePredictor.set_db_session() for blend weight loading.
    Raises ValueError for unknown method names.
    """
```

### `prediction/trainer.py`

```python
TRAINING_FEATURES: list[str]   # 19 columns (15 technical + 4 sentiment)

class XGBoostTrainer:
    def __init__(self, model_dir: Path, horizon: int,
                 label_threshold: float = 0.005): ...
    def train(self, data: EnrichedData, ...) -> XGBClassifier: ...
    def save(self, model, ticker: str) -> Path: ...
    def load(self, ticker: str) -> XGBClassifier | None: ...
    def train_and_save(self, data: EnrichedData) -> bool: ...
    def retrain_with_feedback(self, data, accuracy_records, ...) -> bool: ...
    def extract_features(self, data: EnrichedData) -> np.ndarray | None: ...
```

### `prediction/lstm_trainer.py`

```python
class LSTMTrainer:
    def __init__(self, model_dir: Path, horizon: int, seq_len: int = 60,
                 hidden: int = 128, label_threshold: float = 0.005,
                 epochs: int = 50, batch_size: int = 32): ...
    def train_and_save(self, data: EnrichedData) -> bool:
        """No-op (returns False) when torch is not installed."""
    def load(self, ticker: str) -> tuple[model, scaler_dict] | None:
        """Returns None when torch absent or no saved model."""
    def predict_proba(self, data: EnrichedData) -> np.ndarray | None:
        """Returns shape-(3,) probability array [DOWN,FLAT,UP] or None."""
```

### `prediction/ensemble_predictor.py`

```python
class EnsemblePredictor(AbstractPredictor):
    def set_db_session(self, db) -> None:
        """Inject Session for per-ticker blend weight loading."""
    def predict(self, data, score, horizons) -> PredictionResult:
        """Blend XGB + LSTM probabilities. Falls back: XGB-only → LSTM-only → statistical."""
```

---

## Accuracy Layer

### `accuracy/__init__.py`

```python
def check_and_evaluate(db: Session, config: AppConfig) -> list[AccuracyORM]:
    """Evaluate past-horizon predictions, trigger XGB + LSTM retrain,
    update signal weights and blend weights."""

def optimize_weights(ticker: str, db: Session, global_weights: SignalWeights,
                     min_records: int = 20) -> SignalWeights | None:
    """Adjust all 7 signal weights from accuracy records. Returns None if no change."""

def optimize_blend_weights(ticker: str, db, xgb_accuracy: float | None,
                            lstm_accuracy: float | None) -> None:
    """Nudge xgb_blend / lstm_blend toward the more accurate model."""
```

---

## Storage Layer

### Tables

| Table | Key columns |
|-------|-------------|
| `predictions` | id, ticker, generated_at, model_used, current_price, short_*/long_*, composite_score, action, signals_json |
| `accuracy_records` | id, prediction_id, ticker, horizon_days, direction_correct, price_error_pct |
| `tickers` | id, symbol, market_type, active |
| `ticker_weights` | id, ticker, rsi, macd, trend, volume, bb, stoch, sentiment, xgb_blend, lstm_blend, update_count |
| `sentiment_snapshots` | id, ticker, fetched_at, sentiment_score, bullish_ratio, mention_count, … |

### `storage/session.py`

```python
def init_db(db_path: str = "market_eval.db") -> None:
    """Create tables + run additive migrations. Safe to call on every startup."""

def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a session, close on exit."""
```

### `storage/repository.py`

```python
class PredictionRepository:
    def store(self, report: TickerReport, signals_json: str | None = None) -> int: ...
    def get_latest(self, ticker: str) -> PredictionRecord | None: ...
    def get_history(self, ticker: str, limit: int = 50) -> list[PredictionRecord]: ...
    def get_all_latest(self) -> list[PredictionRecord]: ...
    def get_unevaluated_past_horizon(self) -> list[PredictionRecord]: ...

class AccuracyRepository:
    def store(self, rec: AccuracyORM) -> None: ...
    def get_for_ticker(self, ticker, horizon=None, limit=200) -> list[AccuracyORM]: ...
    def summary(self, ticker, horizon) -> dict: ...

class TickerRepository:
    def get_all_active(self) -> dict[str, str]: ...
    def add(self, symbol: str, market_type: str | None = None) -> TickerRecord: ...
    def remove(self, symbol: str) -> bool: ...
    def seed_from_file(self, cfg_path: str = "ticker.cfg") -> int: ...

class TickerWeightsRepository:
    def get(self, ticker: str) -> TickerWeightsORM | None: ...
    def upsert(self, ticker: str, weights_dict: dict[str, float]) -> TickerWeightsORM: ...
    def get_all(self) -> list[TickerWeightsORM]: ...

class SentimentRepository:
    def store(self, snap: SentimentSnapshot) -> None: ...
    def get_latest(self, ticker: str) -> SentimentSnapshotORM | None: ...
    def get_previous(self, ticker, before_id) -> SentimentSnapshotORM | None: ...
```

---

## Error Hierarchy

```python
class MarketToolError(Exception): ...      # base

# data
class ConfigError(MarketToolError): ...    # bad .env values
class FetchError(MarketToolError): ...     # network / API failure
class NormalizationError(MarketToolError): ...

# features
class FeatureError(MarketToolError): ...   # too few rows / missing columns

# analysis
class AnalysisError(MarketToolError): ...

# prediction
class PredictionError(MarketToolError): ... # insufficient data

# ml
class MLError(MarketToolError): ...        # xgboost / torch failures
```

---

## Module Dependency Rules

```
config               ← no internal imports
data/models          ← config only
data/exceptions      ← no internal imports
data/cache           ← data/models
data/fetchers/*      ← data/models, data/exceptions
data/__init__        ← data/fetchers/*, data/cache
sentiment/           ← data/models, data/fetchers/reddit
features/*           ← data/models
features/pipeline    ← features/*, data/models, sentiment/
analysis/*           ← data/models, config
analysis/__init__    ← analysis/*, storage/repository
prediction/base      ← data/models
prediction/trainer   ← data/models, data/exceptions
prediction/statistical ← data/models, prediction/base
prediction/ml        ← data/models, prediction/base, prediction/trainer
prediction/lstm_*    ← data/models, prediction/base, prediction/trainer
prediction/__init__  ← prediction/*
accuracy/*           ← data/models, prediction/*, storage/*
api/*                ← everything above
ui/                  ← api/ via HTTP only (no direct Python imports)
```

No circular imports. Leaf modules never import from orchestrators.
