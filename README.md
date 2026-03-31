# Market Evaluation Tool

A Python-based market analysis and price-direction forecasting service for stocks,
cryptocurrencies, and commodity futures. Combines seven weighted signals (including Reddit
social-media sentiment), an XGBoost classifier, an LSTM deep-learning model, and an
accuracy-driven ensemble that self-adapts per ticker over time.

> **Disclaimer:** This project is a personal study and research exercise. It is not intended
> for use as serious investment advice or as the basis for any financial decisions. Forecasts
> produced by this tool are experimental and may be inaccurate. Always consult a qualified
> financial professional before making investment decisions.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Reference](#configuration-reference)
3. [User Guide](#user-guide)
4. [API Reference](#api-reference)
5. [Running in Production](#running-in-production)
6. [Architecture](ARCHITECTURE.md)

---

## Quick Start

### Prerequisites

- Python 3.11+
- `python3-venv` (`sudo apt install python3.12-venv` on Ubuntu)
- ~3 GB disk space (PyTorch model download on first use)

### Setup

```bash
# 1. Enter the project
cd market_evaluation_tool

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies  (~2–5 min for PyTorch)
pip install -r requirements.txt

# 4. Copy and edit configuration
cp .env.example .env   # or create .env directly
```

### Run the API server

```bash
source .venv/bin/activate
python serve.py
# API at http://localhost:8000   Swagger at http://localhost:8000/docs
```

### Run the Streamlit UI

```bash
# Second terminal
source .venv/bin/activate
streamlit run ui/app.py
# UI at http://localhost:8501
```

### CLI mode (no API needed)

```bash
source .venv/bin/activate
python main.py
```

### Populate training data

```bash
# Simulate historical predictions + accuracy (recommended before first use)
python training.py --save-to-db
```

---

## Configuration Reference

All configuration lives in `.env`. Copy `.env.example` and remove sensitive values before
committing.

### Tickers

Tickers are managed via the Streamlit UI or directly in the database.
`ticker.cfg` is used to seed the database on first run.

### Data

| Variable | Default | Description |
|----------|---------|-------------|
| `HISTORICAL_DAYS` | `365` | Calendar days of OHLCV history to fetch |
| `CACHE_TTL_HOURS` | `3` | How long fetched data is reused before re-fetching |

### Signal Weights

| Variable | Default | Indicator |
|----------|---------|-----------|
| `SIGNAL_WEIGHTS_RSI` | `1.0` | RSI overbought/oversold |
| `SIGNAL_WEIGHTS_MACD` | `1.5` | MACD histogram crossover |
| `SIGNAL_WEIGHTS_TREND` | `2.0` | SMA/EMA trend direction |
| `SIGNAL_WEIGHTS_VOLUME` | `0.8` | OBV volume flow |
| `SIGNAL_WEIGHTS_BB` | `1.0` | Bollinger Band position |
| `SIGNAL_WEIGHTS_STOCH` | `0.8` | Stochastic oscillator |
| `SIGNAL_WEIGHTS_SENTIMENT` | `0.5` | Reddit social sentiment (Phase 1) |

Per-ticker weights are learned automatically and stored in `ticker_weights`.
These values are the global defaults used before enough accuracy data accumulates.

### Reddit Sentiment (Phase 1)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDDIT_CLIENT_ID` | *(empty)* | PRAW application client ID — sentiment disabled when empty |
| `REDDIT_CLIENT_SECRET` | *(empty)* | PRAW application client secret |
| `REDDIT_USER_AGENT` | `market_eval/2.0` | HTTP user-agent string |
| `SENTIMENT_TTL_HOURS` | `1` | How long Reddit post lists are cached |

Obtain credentials at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) —
create a "script" type application.

### Prediction

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_SHORT` | `3` | Short-horizon forecast in calendar days |
| `FORECAST_LONG` | `7` | Long-horizon forecast in calendar days |
| `PREDICTOR_METHOD` | `ensemble` | `statistical` \| `ml` \| `lstm` \| `ensemble` |
| `ML_LABEL_THRESHOLD` | `0.005` | Min move (0.5%) to classify as UP/DOWN; below = FLAT |
| `RETRAIN_MIN_RECORDS` | `30` | New accuracy records needed before auto-retraining |
| `ACCURACY_CHECK_INTERVAL_HOURS` | `1` | Background accuracy evaluation frequency |

### LSTM (Phase 3)

| Variable | Default | Description |
|----------|---------|-------------|
| `LSTM_SEQUENCE_LEN` | `60` | Look-back bars per training sequence |
| `LSTM_HIDDEN_UNITS` | `128` | LSTM hidden layer size |
| `LSTM_EPOCHS` | `50` | Training epochs per (ticker, horizon) pair |
| `LSTM_BATCH_SIZE` | `32` | Mini-batch size |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `market_eval.db` | SQLite file path (relative, no `..`) |
| `MODEL_DIR` | `.models` | Directory for XGBoost `.json` and LSTM `.pt` model files |

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` only behind a proxy |
| `API_PORT` | `8000` | TCP port |
| `API_TOKEN` | *(empty)* | `X-API-Key` bearer token. Empty = open/dev mode |

### Database Backend (MySQL)

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_TYPE` | `sqlite` | `sqlite` or `mysql` |
| `DATABASE_URL` | *(empty)* | Full SQLAlchemy URL — overrides all other `db_*` settings |
| `MYSQL_HOST` | `localhost` | MySQL host |
| `MYSQL_PORT` | `3306` | MySQL port |
| `MYSQL_USER` | `market_eval` | MySQL user |
| `MYSQL_PASSWORD` | *(empty)* | MySQL password |
| `MYSQL_DATABASE` | `market_eval` | MySQL database name |

### Concurrency

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FETCH_WORKERS` | `8` | Thread pool size for parallel yfinance fetching |
| `MAX_COMPUTE_WORKERS` | `4` | Process pool size for feature engineering |

---

## User Guide

### Overview Page

Displays the most recent prediction for every configured ticker. The **Action** column
shows BUY / HOLD / SELL colour-coded green / amber / red. The **3d Acc %** and **7d Acc %**
columns show historical direction accuracy for that ticker once enough evaluations
have accumulated.

Use **Manage Tickers** to add or remove symbols. Use **Analyze All** to refresh every
ticker at once.

### Forecasts Page

Deep-dive into a single ticker:

- **Chart** tab — candlestick with Bollinger Bands and RSI sub-panel
- **Forecast** tab — current price vs short/long midpoint with low/high range
- **Signals** tab — all 7 individual signal directions and raw values
- **History** tab — last 30 stored predictions

### Accuracy Page

- **Run Evaluation** — compares past-horizon predictions to actual closing prices.
  Also runs automatically every hour.
- Summary cards show **direction accuracy** and **mean price error %** per horizon.
- **Retrain** / **Train All** — triggers XGBoost + LSTM retraining with accuracy feedback,
  then adjusts per-ticker signal weights and ensemble blend weights.

### Models

Model files are stored in `.models/`:

```
.models/<TICKER>_<horizon>d.json          # XGBoost weights
.models/<TICKER>_<horizon>d.json.sha256   # XGBoost integrity sidecar
.models/<TICKER>_<horizon>d_lstm.pt       # LSTM checkpoint (state_dict + scaler)
.models/<TICKER>_<horizon>d_lstm.pt.sha256
```

XGBoost models are trained at startup for any ticker without a model file.
LSTM models are trained when the accuracy evaluator triggers retraining.

### API Authentication

When `API_TOKEN` is set, every request requires:

```
X-API-Key: <your-token>
```

Generate a token: `python -c "import secrets; print(secrets.token_hex(32))"`

---

## API Reference

Full interactive docs at `http://localhost:8000/docs`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze/` | Run full pipeline for a ticker; persist and return result |
| `GET`  | `/analyze/{ticker}/latest` | Most recent stored prediction |
| `GET`  | `/analyze/{ticker}/chart` | OHLCV + indicators for charting |
| `GET`  | `/predictions/` | One latest prediction per ticker |
| `GET`  | `/predictions/{ticker}/history` | Prediction history (default 50, max 500) |
| `GET`  | `/accuracy/{ticker}/summary` | Direction accuracy + mean price error |
| `GET`  | `/accuracy/{ticker}/records` | Raw accuracy evaluation records |
| `GET`  | `/accuracy/{ticker}/weights` | Per-ticker learned signal weights |
| `POST` | `/accuracy/{ticker}/weights/reset` | Reset per-ticker weights to global defaults |
| `POST` | `/accuracy/evaluate` | Manually trigger accuracy evaluation |
| `POST` | `/ml/retrain/{ticker}` | Retrain XGBoost + LSTM for one ticker |
| `POST` | `/ml/train` | Train all tickers |
| `GET`  | `/health` | Health check (no auth required) |

All endpoints except `/health` require `X-API-Key` when `API_TOKEN` is configured.

---

## Running in Production

1. **Set a strong API token:**
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. **Set Reddit credentials** (optional — enables sentiment signal):
   ```
   REDDIT_CLIENT_ID=your_id
   REDDIT_CLIENT_SECRET=your_secret
   ```

3. **Restrict CORS** to your Streamlit domain:
   ```
   CORS_ORIGINS=https://yourdomain.com
   ```

4. **Use a reverse proxy** (nginx, Caddy) and bind uvicorn to `127.0.0.1`:
   ```
   API_HOST=127.0.0.1
   ```

5. **Pre-populate training data:**
   ```bash
   python training.py --save-to-db
   ```

6. **Run with a process manager** (systemd, supervisor):
   ```bash
   /path/to/.venv/bin/python serve.py
   /path/to/.venv/bin/streamlit run ui/app.py --server.port 8501
   ```

7. **SQLite vs MySQL:** SQLite with WAL mode is suitable for single-server deployments.
   For high concurrency set `DB_TYPE=mysql` (or `DATABASE_URL=mysql+pymysql://...`).

---

## Architecture

For a detailed breakdown of the system layers, module structure, data flow, design
principles, and dependency rules, see [ARCHITECTURE.md](ARCHITECTURE.md).
