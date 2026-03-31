"""XGBoost trainer for directional price prediction."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from data.exceptions import MLError
from data.models import AccuracyRecord, EnrichedData

logger = logging.getLogger(__name__)

# Canonical feature columns used for training AND inference (order is fixed).
TRAINING_FEATURES: list[str] = [
    "rsi_14",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "adx_14",
    "macd_hist",
    "returns_1d",
    "returns_5d",
    "z_score_20",
    "realized_vol_20",
    "autocorr_lag1",
    "autocorr_lag5",
    # Derived ratios — computed inside _add_derived()
    "bb_pct",
    "atr_pct",
    "macd_norm",
    # Sentiment features (Phase 2) — 0.0 for historical rows, populated at inference
    "sentiment_score",
    "sentiment_velocity",
    "bullish_ratio",
    "engagement_score",
]

# Label encoding
_LABEL_MAP = {-1: 0, 0: 1, 1: 2}   # DOWN=0, FLAT=1, UP=2
_LABEL_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}

# Per-ticker write lock prevents concurrent read-during-write of model files
_MODEL_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

# Only allow alphanumeric, hyphens and underscores in ticker-derived filenames
_SAFE_TICKER_RE = re.compile(r"^[A-Z0-9\-_]{1,20}$")


def _get_lock(ticker: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if ticker not in _MODEL_LOCKS:
            _MODEL_LOCKS[ticker] = threading.Lock()
        return _MODEL_LOCKS[ticker]


def _safe_ticker_slug(ticker: str) -> str:
    """Return a filesystem-safe slug from a ticker symbol."""
    slug = re.sub(r"[^A-Z0-9\-]", "_", ticker.upper())
    if not _SAFE_TICKER_RE.match(slug):
        raise ValueError(f"Cannot derive safe filename slug from ticker {ticker!r}")
    return slug


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio features from existing columns. Returns new DataFrame."""
    out = df.copy()
    # BB position: 0 = at lower band, 1 = at upper band
    denom = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["bb_pct"] = (out["close"] - out["bb_lower"]) / denom
    # ATR as fraction of price
    out["atr_pct"] = out["atr_14"] / out["close"].replace(0, np.nan)
    # MACD hist normalized by price
    out["macd_norm"] = out["macd_hist"] / out["close"].replace(0, np.nan)
    return out


def _build_labels(close: pd.Series, horizon: int, threshold: float) -> pd.Series:
    """Forward-looking direction labels. Last `horizon` rows are NaN."""
    fwd_return = close.pct_change(horizon).shift(-horizon)
    labels = pd.Series(np.where(
        fwd_return > threshold, 1,
        np.where(fwd_return < -threshold, -1, 0)
    ), index=close.index, dtype=int)
    labels[fwd_return.isna()] = np.nan
    return labels


class XGBoostTrainer:
    """Trains and persists per-(ticker, horizon) XGBoost classifiers."""

    def __init__(self, model_dir: Path, horizon: int, label_threshold: float = 0.005) -> None:
        self.model_dir = model_dir.resolve()
        self.horizon = horizon
        self.threshold = label_threshold
        self.model_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # ── public ────────────────────────────────────────────────────────────────

    def train(self, data: EnrichedData, sample_weights: np.ndarray | None = None):
        """Fit an XGBClassifier and return it (does not save to disk)."""
        try:
            import xgboost as xgb
        except ImportError as e:
            raise MLError("xgboost is not installed") from e

        X, y = self._build_dataset(data)
        if X is None:
            raise MLError(f"{data.ticker}: insufficient training data for horizon {self.horizon}d")

        if sample_weights is not None and len(sample_weights) == len(y):
            sw = sample_weights
        else:
            sw = None

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y, sample_weight=sw)
        return model

    def save(self, model, ticker: str) -> Path:
        """Save model to {model_dir}/{ticker}_{horizon}d.json under a write lock.
        Writes a SHA-256 sidecar for integrity verification on load.
        """
        path = self._model_path(ticker)
        hash_path = path.with_suffix(".json.sha256")
        with _get_lock(ticker):
            model.save_model(str(path))
            hash_path.write_text(_sha256(path))
        logger.info("Saved ML model: %s", path.name)
        return path

    def load(self, ticker: str):
        """Load saved model; returns None if not found or integrity check fails."""
        try:
            import xgboost as xgb
        except ImportError:
            return None

        path = self._model_path(ticker)
        hash_path = path.with_suffix(".json.sha256")

        if not path.exists():
            return None

        with _get_lock(ticker):
            # Integrity check before loading
            if hash_path.exists():
                expected = hash_path.read_text().strip()
                actual = _sha256(path)
                if actual != expected:
                    logger.error(
                        "Model integrity check FAILED for %s (expected=%s actual=%s). "
                        "Refusing to load.",
                        ticker, expected[:12], actual[:12],
                    )
                    return None
            else:
                logger.warning(
                    "No SHA-256 sidecar for %s model — integrity unverified. "
                    "Re-train to generate a verified model.",
                    ticker,
                )
            model = xgb.XGBClassifier()
            model.load_model(str(path))

        return model

    def train_and_save(self, data: EnrichedData) -> bool:
        """Convenience: train from scratch and save. Returns True on success."""
        try:
            model = self.train(data)
            self.save(model, data.ticker)
            return True
        except MLError as exc:
            logger.warning("Training failed for %s h=%d: %s", data.ticker, self.horizon, exc)
            return False

    def retrain_with_feedback(
        self,
        data: EnrichedData,
        accuracy_records: list[AccuracyRecord],
        wrong_weight: float = 1.5,
    ) -> bool:
        """Retrain with accuracy feedback: up-weight historically wrong predictions.

        Rejects feedback if the overall correct rate is suspiciously low (< 20%),
        which may indicate poisoned accuracy data.
        """
        X, y = self._build_dataset(data)
        if X is None:
            return False

        df = self._prepare_df(data)
        if df is None:
            return False

        sw = np.ones(len(y), dtype=float)

        if accuracy_records:
            # Sanity check: reject suspiciously low accuracy (possible data poisoning)
            correct_rate = sum(1 for r in accuracy_records if r.direction_correct) / len(accuracy_records)
            if correct_rate < 0.20:
                logger.warning(
                    "Accuracy feedback for %s has only %.1f%% correct — "
                    "skipping retrain to avoid poisoning.",
                    data.ticker, correct_rate * 100,
                )
                return False

            dates = df.index.normalize()
            for rec in accuracy_records:
                if not rec.direction_correct:
                    target = pd.Timestamp(rec.evaluated_at).normalize()
                    # Align tz-awareness: df index is always UTC-aware; target may be naive
                    if target.tzinfo is None and dates.tzinfo is not None:
                        target = target.tz_localize("UTC")
                    elif target.tzinfo is not None and dates.tzinfo is None:
                        target = target.tz_localize(None)
                    mask = (dates >= target - pd.Timedelta(days=3)) & (dates <= target)
                    sw[mask[:len(sw)]] = wrong_weight

        try:
            model = self.train(data, sample_weights=sw)
            self.save(model, data.ticker)
            return True
        except MLError as exc:
            logger.warning("Feedback retrain failed: %s", exc)
            return False

    def extract_features(self, data: EnrichedData) -> np.ndarray | None:
        """Extract the feature vector for the most recent row."""
        df = _add_derived(data.df)
        available = [c for c in TRAINING_FEATURES if c in df.columns]
        if len(available) < len(TRAINING_FEATURES) * 0.8:
            return None
        row = df[available].dropna(how="all").iloc[-1:]
        if row.empty:
            return None
        return row.reindex(columns=TRAINING_FEATURES, fill_value=0.0).replace([np.inf, -np.inf], 0.0).values

    # ── private ───────────────────────────────────────────────────────────────

    def _model_path(self, ticker: str) -> Path:
        slug = _safe_ticker_slug(ticker)
        candidate = self.model_dir / f"{slug}_{self.horizon}d.json"
        # Resolve and confirm the path stays within model_dir (path traversal guard)
        resolved = candidate.resolve()
        if not str(resolved).startswith(str(self.model_dir)):
            raise ValueError(
                f"Model path {resolved} escapes model_dir {self.model_dir}"
            )
        return resolved

    def _prepare_df(self, data: EnrichedData) -> pd.DataFrame | None:
        required = {"close", "bb_upper", "bb_lower", "atr_14", "macd_hist"}
        if not required.issubset(data.df.columns):
            return None
        return _add_derived(data.df)

    def _build_dataset(self, data: EnrichedData):
        df = self._prepare_df(data)
        if df is None:
            return None, None

        labels = _build_labels(df["close"], self.horizon, self.threshold)
        valid = labels.dropna().index
        df_valid = df.loc[valid]
        y_raw = labels.loc[valid].astype(int)

        if len(df_valid) < 60:
            return None, None

        X_df = df_valid.reindex(columns=TRAINING_FEATURES, fill_value=0.0).fillna(0.0)
        X = X_df.replace([np.inf, -np.inf], 0.0).values
        y = np.array([_LABEL_MAP[v] for v in y_raw])

        if len(np.unique(y)) < 2:
            return None, None

        return X, y
