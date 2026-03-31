"""PyTorch LSTM trainer for per-ticker directional price prediction.

Architecture
------------
Input:  (batch, seq_len, n_features)  — sequences of TRAINING_FEATURES
LSTM:   hidden=128, num_layers=2, dropout=0.2
Linear: 128 → 64  +  ReLU
Linear: 64 → 3    +  Softmax
Output: probabilities over [DOWN=0, FLAT=1, UP=2]

Training mirrors XGBoostTrainer:
- Same forward-looking direction labels.
- Min-max normalisation per-feature over the training set.
- Saves {ticker}_{horizon}d_lstm.pt (state-dict + scaler params) alongside the
  SHA-256 sidecar used by XGBoostTrainer for integrity verification.

Fallback
--------
All public methods are no-ops (returning False / None) when PyTorch is not
installed, so the rest of the codebase never needs to guard the import.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from data.exceptions import MLError
from data.models import EnrichedData
from prediction.trainer import (
    TRAINING_FEATURES,
    XGBoostTrainer,
    _add_derived,
    _build_labels,
    _get_lock,
    _safe_ticker_slug,
)

logger = logging.getLogger(__name__)

_SEQ_LEN    = 60    # look-back bars
_HIDDEN     = 128
_NUM_LAYERS = 2
_DROPOUT    = 0.2
_LR         = 1e-3
_LABEL_MAP  = {-1: 0, 0: 1, 1: 2}
_LABEL_NAMES= {0: "DOWN", 1: "FLAT", 2: "UP"}


# ── optional torch / ipex imports ────────────────────────────────────────────

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception as exc:
        logger.debug("PyTorch unavailable: %s", exc)
        return False


def _get_device():
    """Return the best available torch.device: XPU > CUDA > CPU.

    PyTorch 2.4+ exposes torch.xpu natively — no third-party extension needed.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _build_model(n_features: int, hidden: int = _HIDDEN, num_layers: int = _NUM_LAYERS):
    """Construct the LSTM model.  Caller must ensure torch is available."""
    import torch
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=_DROPOUT if num_layers > 1 else 0.0,
            )
            self.fc1 = nn.Linear(hidden, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 3)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]          # last time-step
            out = self.relu(self.fc1(out))
            return self.fc2(out)         # logits

    return _LSTMNet()


class LSTMTrainer:
    """Train and persist a per-(ticker, horizon) LSTM classifier.

    All public methods silently degrade to no-ops when PyTorch is absent.
    """

    def __init__(
        self,
        model_dir: Path,
        horizon: int,
        seq_len: int = _SEQ_LEN,
        hidden: int = _HIDDEN,
        label_threshold: float = 0.005,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        self.model_dir = model_dir.resolve()
        self.horizon = horizon
        self.seq_len = seq_len
        self.hidden = hidden
        self.threshold = label_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # ── public API ────────────────────────────────────────────────────────────

    def train_and_save(self, data: EnrichedData) -> bool:
        """Train from scratch and save.  Returns True on success."""
        if not _torch_available():
            logger.info("PyTorch not installed — LSTM training skipped for %s", data.ticker)
            return False
        try:
            model, scaler_params = self._train(data)
            if model is None:
                return False
            self._save(model, scaler_params, data.ticker)
            return True
        except Exception as exc:
            logger.warning("LSTM training failed for %s h=%d: %s", data.ticker, self.horizon, exc)
            return False

    def load(self, ticker: str) -> tuple | None:
        """Load saved model and scaler params.  Returns (model, scaler_params) or None."""
        if not _torch_available():
            return None
        import torch

        path = self._model_path(ticker)
        hash_path = path.with_suffix(".pt.sha256")
        if not path.exists():
            return None

        with _get_lock(ticker):
            if hash_path.exists():
                expected = hash_path.read_text().strip()
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    logger.error("LSTM model integrity check failed for %s", ticker)
                    return None
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

        n_features = checkpoint["n_features"]
        model = _build_model(n_features, self.hidden)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint["scaler"]

    def predict_proba(self, data: EnrichedData) -> np.ndarray | None:
        """Return probability array shape (3,) for the most recent bar, or None."""
        result = self.load(data.ticker)
        if result is None:
            return None
        model, scaler = result

        X = self._extract_sequence(data, scaler)
        if X is None:
            return None

        import torch
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)
            logits = model(tensor)
            proba = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        return proba  # shape (3,)

    # ── private ───────────────────────────────────────────────────────────────

    def _model_path(self, ticker: str) -> Path:
        slug = _safe_ticker_slug(ticker)
        candidate = self.model_dir / f"{slug}_{self.horizon}d_lstm.pt"
        resolved = candidate.resolve()
        if not str(resolved).startswith(str(self.model_dir)):
            raise ValueError(f"LSTM model path {resolved} escapes model_dir")
        return resolved

    def _prepare_features(self, data: EnrichedData) -> np.ndarray | None:
        """Return (n_rows, n_features) float array with TRAINING_FEATURES."""
        df = _add_derived(data.df)
        mat = df.reindex(columns=TRAINING_FEATURES, fill_value=0.0).fillna(0.0)
        return mat.replace([np.inf, -np.inf], 0.0).values.astype(np.float32)

    def _extract_sequence(self, data: EnrichedData, scaler: dict) -> np.ndarray | None:
        """Return the last seq_len rows, scaled, shape (seq_len, n_features)."""
        mat = self._prepare_features(data)
        if mat is None or len(mat) < self.seq_len:
            return None
        seq = mat[-self.seq_len:].copy()
        mean = np.array(scaler["mean"], dtype=np.float32)
        std  = np.array(scaler["std"],  dtype=np.float32)
        std[std == 0] = 1.0
        seq = (seq - mean) / std
        return seq

    def _train(self, data: EnrichedData):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = _get_device()
        logger.debug("LSTM training device: %s", device)

        mat = self._prepare_features(data)
        if mat is None:
            return None, None

        labels = _build_labels(
            data.df["close"].reindex(data.df.index),
            self.horizon,
            self.threshold,
        )
        valid_idx = labels.dropna().index
        # Align matrix rows with valid label index
        df_idx = _add_derived(data.df).index
        valid_pos = [i for i, idx in enumerate(df_idx) if idx in valid_idx]
        if len(valid_pos) < self.seq_len + 30:
            return None, None

        mat_valid = mat[valid_pos]
        y_raw = labels.loc[valid_idx].astype(int).values
        y = np.array([_LABEL_MAP[v] for v in y_raw])

        if len(np.unique(y)) < 2:
            return None, None

        # Min-max scaler (fitted on training set)
        mean = mat_valid.mean(axis=0)
        std  = mat_valid.std(axis=0)
        std[std == 0] = 1.0
        mat_scaled = (mat_valid - mean) / std

        # Build sliding-window sequences
        X_seqs, y_seqs = [], []
        for i in range(self.seq_len, len(mat_scaled)):
            X_seqs.append(mat_scaled[i - self.seq_len: i])
            y_seqs.append(y[i])

        if len(X_seqs) < 30:
            return None, None

        X_t = torch.tensor(np.array(X_seqs, dtype=np.float32))
        y_t = torch.tensor(np.array(y_seqs, dtype=np.int64))

        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                             pin_memory=(device.type in ("cuda", "xpu")), num_workers=0)

        n_feat = X_t.shape[-1]
        model = _build_model(n_feat, self.hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
        criterion = nn.CrossEntropyLoss()


        model.train()
        for _epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        # Move model back to CPU for serialisation — keeps .pt files device-agnostic
        model = model.cpu()
        scaler_params = {
            "mean": mean.tolist(),
            "std":  std.tolist(),
        }
        return model, scaler_params

    def _save(self, model, scaler_params: dict, ticker: str) -> None:
        import torch
        path = self._model_path(ticker)
        hash_path = path.with_suffix(".pt.sha256")
        checkpoint = {
            "state_dict": model.state_dict(),
            "scaler": scaler_params,
            "n_features": len(TRAINING_FEATURES),
            "horizon": self.horizon,
            "seq_len": self.seq_len,
        }
        with _get_lock(ticker):
            torch.save(checkpoint, str(path))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_path.write_text(sha)
        logger.info("Saved LSTM model: %s", path.name)
