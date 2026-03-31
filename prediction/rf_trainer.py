"""Random Forest trainer for directional price prediction.

Mirrors XGBoostTrainer exactly:
  - Same TRAINING_FEATURES feature set
  - Same label encoding (DOWN=0, FLAT=1, UP=2)
  - Same file-path convention: {ticker}_{horizon}d_rf.joblib
  - Same SHA-256 integrity sidecar

Requires scikit-learn (already in requirements.txt).
joblib ships with scikit-learn; no additional dependency.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from data.exceptions import MLError
from data.models import AccuracyRecord, EnrichedData
from prediction.trainer import (
    TRAINING_FEATURES,
    _LABEL_MAP,
    _add_derived,
    _build_labels,
    _get_lock,
    _safe_ticker_slug,
)

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RandomForestTrainer:
    """Trains and persists per-(ticker, horizon) Random Forest classifiers."""

    def __init__(self, model_dir: Path, horizon: int, label_threshold: float = 0.005) -> None:
        self.model_dir = model_dir.resolve()
        self.horizon = horizon
        self.threshold = label_threshold
        self.model_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # ── public ────────────────────────────────────────────────────────────────

    def train(self, data: EnrichedData, sample_weights: np.ndarray | None = None):
        """Fit a RandomForestClassifier and return it (does not save to disk)."""
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as e:
            raise MLError("scikit-learn is not installed") from e

        X, y = self._build_dataset(data)
        if X is None:
            raise MLError(f"{data.ticker}: insufficient training data for horizon {self.horizon}d")

        sw = sample_weights if (sample_weights is not None and len(sample_weights) == len(y)) else None

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X, y, sample_weight=sw)
        return model

    def save(self, model, ticker: str) -> Path:
        """Save model via joblib; write SHA-256 sidecar for integrity."""
        try:
            import joblib
        except ImportError as e:
            raise MLError("joblib is not installed (ships with scikit-learn)") from e

        path = self._model_path(ticker)
        hash_path = path.with_suffix(".joblib.sha256")
        with _get_lock(ticker):
            joblib.dump(model, str(path))
            hash_path.write_text(_sha256(path))
        logger.info("Saved RF model: %s", path.name)
        return path

    def load(self, ticker: str):
        """Load saved model; returns None if not found or integrity check fails."""
        try:
            import joblib
        except ImportError:
            return None

        path = self._model_path(ticker)
        hash_path = path.with_suffix(".joblib.sha256")

        if not path.exists():
            return None

        with _get_lock(ticker):
            if hash_path.exists():
                expected = hash_path.read_text().strip()
                actual = _sha256(path)
                if actual != expected:
                    logger.error(
                        "RF model integrity check FAILED for %s (expected=%s actual=%s). "
                        "Refusing to load.",
                        ticker, expected[:12], actual[:12],
                    )
                    return None
            else:
                logger.warning(
                    "No SHA-256 sidecar for %s RF model — integrity unverified. "
                    "Re-train to generate a verified model.",
                    ticker,
                )
            model = joblib.load(str(path))

        return model

    def train_and_save(self, data: EnrichedData) -> bool:
        """Convenience: train from scratch and save. Returns True on success."""
        try:
            model = self.train(data)
            self.save(model, data.ticker)
            return True
        except MLError as exc:
            logger.warning("RF training failed for %s h=%d: %s", data.ticker, self.horizon, exc)
            return False

    def retrain_with_feedback(
        self,
        data: EnrichedData,
        accuracy_records: list[AccuracyRecord],
        wrong_weight: float = 1.5,
    ) -> bool:
        """Retrain with accuracy feedback: up-weight historically wrong predictions.

        Rejects feedback if overall correct rate is suspiciously low (< 20%).
        """
        import pandas as pd

        X, y = self._build_dataset(data)
        if X is None:
            return False

        df = self._prepare_df(data)
        if df is None:
            return False

        sw = np.ones(len(y), dtype=float)

        if accuracy_records:
            correct_rate = sum(1 for r in accuracy_records if r.direction_correct) / len(accuracy_records)
            if correct_rate < 0.20:
                logger.warning(
                    "RF accuracy feedback for %s has only %.1f%% correct — "
                    "skipping retrain to avoid poisoning.",
                    data.ticker, correct_rate * 100,
                )
                return False

            dates = df.index.normalize()
            for rec in accuracy_records:
                if not rec.direction_correct:
                    target = pd.Timestamp(rec.evaluated_at).normalize()
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
            logger.warning("RF feedback retrain failed: %s", exc)
            return False

    def extract_features(self, data: EnrichedData) -> np.ndarray | None:
        """Extract feature vector for the most recent row."""
        df = _add_derived(data.df)
        available = [c for c in TRAINING_FEATURES if c in df.columns]
        if len(available) < len(TRAINING_FEATURES) * 0.8:
            return None
        row = df[available].dropna(how="all").iloc[-1:]
        if row.empty:
            return None
        return (
            row.reindex(columns=TRAINING_FEATURES, fill_value=0.0)
            .replace([np.inf, -np.inf], 0.0)
            .values
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _model_path(self, ticker: str) -> Path:
        slug = _safe_ticker_slug(ticker)
        candidate = self.model_dir / f"{slug}_{self.horizon}d_rf.joblib"
        resolved = candidate.resolve()
        if not str(resolved).startswith(str(self.model_dir)):
            raise ValueError(f"Model path {resolved} escapes model_dir {self.model_dir}")
        return resolved

    def _prepare_df(self, data: EnrichedData):
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
