# Market Evaluation Tool — Architecture

## Overview

A modular, Python-based market analysis and prediction tool targeting stocks, crypto, and
commodities. Combines rule-based technical signals, XGBoost ML direction classification,
an LSTM deep-learning predictor, and Reddit social-media sentiment into a self-adapting
ensemble. Prediction accuracy is tracked per-ticker; signal weights and model blend ratios
adjust automatically as feedback accumulates.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│  Streamlit UI  (ui/)                                                │
│  ui/app.py  ──►  ui/pages/{overview, forecasts, accuracy}.py       │
│                  ui/components/{candlestick, forecast_chart,        │
│                                 signal_table}.py                    │
│                  ui/api_client.py  (httpx, sync)                    │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ HTTP  X-API-Key
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI Service  (api/, serve.py)                                  │
│                                                                     │
│  POST /analyze/               ──► full pipeline for one ticker      │
│  GET  /analyze/{t}/latest     ──► latest stored prediction          │
│  GET  /analyze/{t}/chart      ──► OHLCV + indicators (120 rows)     │
│  GET  /predictions/           ──► one latest per ticker             │
│  GET  /predictions/{t}/history                                      │
│  GET  /accuracy/{t}/summary   ──► direction accuracy + price error  │
│  GET  /accuracy/{t}/records                                         │
│  POST /accuracy/evaluate      ──► manual trigger (also runs hourly) │
│  POST /ml/retrain/{t}         ──► retrain XGBoost + LSTM            │
│  POST /ml/train               ──► train all tickers                 │
│  GET  /health                                                       │
│                                                                     │
│  Background (APScheduler):                                          │
│    accuracy_eval — every ACCURACY_CHECK_INTERVAL_HOURS              │
│    initial_train — once at startup (trains missing models)          │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ├── data/           fetch OHLCV + Reddit posts
         │   ├── fetchers/   StockFetcher / CryptoFetcher / CommodityFetcher
         │   │               RedditFetcher (Phase 1; requires REDDIT_CLIENT_ID)
         │   └── cache/      Parquet + JSON sidecar, TTL-based
         │
         ├── sentiment/      Reddit NLP pipeline (Phase 1)
         │   ├── scorer.py   FinBERT (preferred) → VADER fallback
         │   └── __init__    get_sentiment_snapshot() → SentimentSnapshot
         │
         ├── features/       build_features_all + augment_with_sentiment (Phase 2)
         │   ├── technical/  RSI, MACD, BB, ATR, ADX, OBV, VWAP, MFI, Stoch, %R
         │   ├── statistical/ returns, z-score, vol, autocorrelation
         │   └── pipeline/   orchestration + sentiment feature injection
         │
         ├── analysis/       detect_trend → compute_signals → build_composite
         │   ├── trend/      SMA/EMA crossovers, ADX regime detection
         │   ├── signals/    7 weighted signals → SignalDir {-1, 0, +1}
         │   │               (rsi, macd_cross, bb_position, ma_cross,
         │   │                volume_trend, stoch, sentiment)
         │   └── scoring/    CompositeScore [-1.0, 1.0] → BUY/HOLD/SELL
         │
         ├── prediction/     AbstractPredictor
         │   ├── statistical/ linear regression + ATR bands (always works)
         │   ├── ml/         XGBoost classifier (falls back to statistical)
         │   ├── lstm/       LSTM deep-learning classifier (Phase 3)
         │   ├── ensemble/   blends XGBoost + LSTM per-ticker weights (Phase 4)
         │   └── trainer/    XGBoostTrainer + LSTMTrainer
         │                   SHA-256 integrity sidecars for all model files
         │
         ├── accuracy/       check_and_evaluate → retrain XGB + LSTM → optimize weights
         │   └── weight_optimizer  per-ticker signal weights + XGB/LSTM blend weights
         │
         └── storage/        SQLAlchemy ORM, SQLite (WAL mode) or MySQL
             ├── orm_models/ predictions, accuracy_records,
             │               ticker_weights, sentiment_snapshots
             └── repository/ PredictionRepository, AccuracyRepository,
                             TickerWeightsRepository, SentimentRepository
```

---

## Target Markets & Data Sources

| Market      | Tickers         | Primary Source        | Fallback         |
|-------------|-----------------|----------------------|------------------|
| Stocks      | User-configured | `yfinance`           | —                |
| Crypto      | User-configured | `yfinance` + `ccxt`  | —                |
| Commodities | User-configured | `yfinance` (futures) | —                |
| Sentiment   | Same tickers    | Reddit (PRAW)        | Disabled gracefully |

---

## Project Structure

```
market_evaluation_tool/
├── .env                        # Runtime config (not committed)
├── config.py                   # Central config + SignalWeights dataclass
├── requirements.txt
├── main.py                     # CLI entry point
├── serve.py                    # FastAPI + APScheduler launcher
├── training.py                 # Historical training simulation (--save-to-db)
├── ticker.cfg                  # Initial ticker list (seeds DB on first run)
│
├── data/
│   ├── models.py               # Shared frozen dataclasses (MarketData, EnrichedData,
│   │                           #   SentimentSnapshot, CompositeScore, PredictionResult …)
│   ├── exceptions.py           # Error hierarchy
│   ├── cache.py                # Parquet + JSON sidecar TTL cache
│   └── fetchers/
│       ├── base.py             # AbstractFetcher
│       ├── stocks.py           # yfinance StockFetcher
│       ├── crypto.py           # yfinance + ccxt CryptoFetcher
│       ├── commodities.py      # yfinance CommodityFetcher
│       └── reddit.py           # PRAW RedditFetcher (Phase 1)
│
├── sentiment/                  # Phase 1 — NLP scoring pipeline
│   ├── __init__.py             # get_sentiment_snapshot() entry point
│   └── scorer.py               # FinBERT → VADER fallback; score_texts()
│
├── features/
│   ├── technical.py            # TA indicators (RSI, MACD, BB, ATR, ADX, Stoch, %R …)
│   ├── statistical.py          # Returns, rolling stats, Z-score, autocorrelation
│   └── pipeline.py             # build_features() + augment_with_sentiment() (Phase 2)
│
├── analysis/
│   ├── trend.py                # detect_trend() → TrendResult
│   ├── signals.py              # 7 signal functions → Signal{-1,0,+1}
│   │                           #   rsi, macd_cross, bb_position, ma_cross,
│   │                           #   volume_trend, stoch, sentiment (Phase 1)
│   ├── scoring.py              # build_composite() → CompositeScore [-1,1]
│   └── __init__.py             # analyze(), get_ticker_weights()
│
├── prediction/
│   ├── base.py                 # AbstractPredictor (predict → PredictionResult)
│   ├── statistical.py          # Linear regression + ATR bands (always available)
│   ├── ml.py                   # XGBoost classifier + statistical price bands
│   ├── lstm_trainer.py         # PyTorch LSTM trainer (Phase 3)
│   ├── lstm_predictor.py       # LSTMPredictor (Phase 3)
│   ├── ensemble_predictor.py   # EnsemblePredictor — blends XGB + LSTM (Phase 4)
│   ├── trainer.py              # XGBoostTrainer; TRAINING_FEATURES list (19 cols)
│   └── __init__.py             # get_predictor(method) factory
│
├── accuracy/
│   ├── evaluator.py            # check_and_evaluate(); triggers XGB + LSTM retrain
│   ├── weight_optimizer.py     # compute_adjusted_weights(), optimize_blend_weights()
│   └── __init__.py
│
├── storage/
│   ├── orm_models.py           # SQLAlchemy ORM:
│   │                           #   PredictionRecord, AccuracyORM,
│   │                           #   TickerRecord, TickerWeightsORM,
│   │                           #   SentimentSnapshotORM
│   ├── repository.py           # PredictionRepository, AccuracyRepository,
│   │                           #   TickerRepository, TickerWeightsRepository,
│   │                           #   SentimentRepository
│   └── session.py              # init_db(), get_session(), _migrate()
│
├── api/
│   ├── main.py                 # FastAPI app + CORS + auth dependency
│   ├── dependencies.py
│   ├── schemas.py              # Pydantic request/response models
│   ├── limiter.py              # slowapi rate limiter
│   └── routes/
│       ├── analysis.py         # POST /analyze/, GET /analyze/{t}/latest|chart
│       ├── predictions.py      # GET /predictions/
│       ├── accuracy.py         # /accuracy/* + /ml/*
│       └── tickers.py          # GET|POST|DELETE /tickers/
│
├── ui/
│   ├── app.py                  # Streamlit navigation + session setup
│   ├── api_client.py           # httpx sync client wrapper
│   ├── pages/
│   │   ├── overview.py         # All-tickers dashboard
│   │   ├── forecasts.py        # Per-ticker deep-dive
│   │   └── accuracy.py         # Accuracy tracking + ML retraining controls
│   └── components/
│       ├── candlestick.py
│       ├── forecast_chart.py
│       └── signal_table.py
│
└── tests/
    ├── test_data/
    ├── test_features/
    ├── test_analysis/
    ├── test_prediction/
    ├── test_sentiment/
    └── test_accuracy/
```

---

## Data Flow

```
OHLCV fetch ──► DataCache ──► build_features() ──► augment_with_sentiment()
                                                         ▲
                              Reddit API ─► scorer ──────┘

                                          analyze(data, weights, snapshot)
                                          ├── detect_trend()
                                          ├── compute_signals()   ← 7 signals
                                          └── build_composite()   → CompositeScore

                                          EnsemblePredictor.predict()
                                          ├── XGBoost proba  × xgb_blend
                                          └── LSTM proba     × lstm_blend
                                              → blended direction + stat bands

                                          PredictionRepository.store()

                                      ◄── accuracy evaluation (hourly) ──►
                                          _maybe_retrain (XGB + LSTM)
                                          optimize_weights (signal weights)
                                          optimize_blend_weights (XGB/LSTM blend)
```

---

## Signal Weights — Adaptation Loop

Each of the 7 signals has a per-ticker weight stored in `ticker_weights`.
After each accuracy evaluation batch the optimizer adjusts weights proportionally to
each signal's historical direction accuracy (learning rate 0.25, clamped 0.05–5.0).
The XGBoost / LSTM ensemble blend weights are adjusted similarly (learning rate 0.10,
clamped 0.1–0.9), favouring whichever model has been more accurate recently.

---

## Module Responsibilities

### `data/`
- **fetchers/** — Normalize all sources to `{date: index, open, high, low, close, volume}`
- **cache.py** — Parquet TTL cache; invalidated on age; never mutates fetched data
- **reddit.py** — Fetches Reddit posts mentioning a ticker; returns `[(text, engagement)]`;
  result cached in `.cache/reddit/` with configurable TTL

### `sentiment/`
- **scorer.py** — NLP inference. Tries FinBERT (`ProsusAI/finbert` via `transformers`);
  falls back to VADER if torch/transformers unavailable. Returns `(score, bullish_ratio, engagement)`.
- **`__init__.py`** — `get_sentiment_snapshot()` orchestrates fetch + score; returns `None`
  when Reddit credentials absent — all callers handle `None` gracefully.

### `features/`
- **pipeline.py** — `build_features()`: technical + statistical chain.
  `augment_with_sentiment()`: injects 4 sentiment columns into the last row of `EnrichedData.df`.
  Historical rows receive 0.0; the most recent bar receives live sentiment values.

### `analysis/`
- **signals.py** — 7 independent signal functions. `compute_signals(data, weights, snapshot=None)`.
  `sentiment_signal()` returns HOLD when snapshot is None or mention_count < 5.
- **scoring.py** — Weighted sum → [-1, 1]; thresholds ±0.25 for BUY/SELL.
- **`__init__.py`** — `get_ticker_weights()` loads per-ticker DB weights (fallback: global config).
  `analyze(data, weights, snapshot=None)` is the single analysis entry point.

### `prediction/`
- **statistical.py** — Linear regression + momentum + ATR bands. No model files required.
- **ml.py** — XGBoost direction → mid clamped to direction → statistical ATR bands.
- **lstm_trainer.py** — Per-(ticker, horizon) LSTM; saves `.pt` checkpoint with SHA-256 sidecar;
  graceful no-op when `torch` is absent.
- **lstm_predictor.py** — Mirrors MLPredictor design (direction + statistical bands + mid-clamp).
- **ensemble_predictor.py** — Reads `xgb_blend` / `lstm_blend` from DB; falls back through
  XGB-only → LSTM-only → statistical as models become unavailable.
- **trainer.py** — `TRAINING_FEATURES` (19 columns: 15 technical + 4 sentiment).
  `retrain_with_feedback()` up-weights historically wrong rows.
- **`__init__.py`** — `get_predictor(method, model_dir, db)` factory; supports
  `"statistical"`, `"ml"`, `"lstm"`, `"ensemble"`.

### `accuracy/`
- **evaluator.py** — `check_and_evaluate()`: fetches actual prices for past-horizon predictions,
  stores `AccuracyORM`, triggers XGB + LSTM retrain, updates weights.
- **weight_optimizer.py** — Pure functions for weight adaptation:
  - `compute_adjusted_weights()` — signal accuracy → weight delta (LR 0.25, bounds 0.05–5.0)
  - `optimize_blend_weights()` — XGB vs LSTM accuracy → blend delta (LR 0.10, bounds 0.1–0.9)

### `storage/`
- **orm_models.py** — Five tables: `predictions`, `accuracy_records`, `tickers`,
  `ticker_weights`, `sentiment_snapshots`.
- **session.py** — `init_db()` creates tables + runs `_migrate()` (additive column additions,
  safe to call on every startup).

---

## Key Design Principles

- **Graceful degradation** — Reddit credentials absent → neutral sentiment signal.
  `torch` absent → LSTM skipped, ensemble falls back to XGB; XGB absent → statistical.
- **No global state** — config passed explicitly; no module-level side effects.
- **Abstract interfaces** — `AbstractFetcher`, `AbstractPredictor`; swap implementations
  without touching analysis logic.
- **Immutable DataFrames** — every transformation returns a new DataFrame.
- **SHA-256 integrity** — all model files have sidecar checksums; corrupted models are
  refused and trigger a warning.
- **Additive migrations** — `_migrate()` only adds columns; never drops or alters existing
  schema; safe to call on startup.
- **Adaptation loop** — weights and blend ratios converge toward optimal values as
  accuracy feedback accumulates; no manual tuning required after initial setup.

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

---

## Parallelization

| Layer | Strategy | Library |
|-------|----------|---------|
| OHLCV fetch | Thread pool (I/O-bound) | `ThreadPoolExecutor` |
| Feature engineering | Process pool (CPU-bound) | `ProcessPoolExecutor` |
| Analysis + prediction | Process pool per ticker | `ProcessPoolExecutor` |
| Reddit fetch | Synchronous (cached; < 1 s) | — |
| LSTM training | Synchronous per ticker | PyTorch CPU/GPU |

---

## Internal API Contracts

For detailed internal Python API specifications (data model dataclasses, layer function
signatures, storage schemas, and error hierarchy) see [API.md](API.md).
