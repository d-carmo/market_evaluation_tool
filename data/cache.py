import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from data.models import MarketData, MarketType

logger = logging.getLogger(__name__)

_SAFE_RE = re.compile(r"^[A-Z0-9\-_]{1,20}$")
_MAX_PARQUET_BYTES = 100 * 1024 * 1024  # 100 MB


def _safe_slug(ticker: str) -> str:
    slug = re.sub(r"[^A-Z0-9\-]", "_", ticker.upper())
    if not _SAFE_RE.match(slug):
        raise ValueError(f"Cannot derive safe cache filename from ticker {ticker!r}")
    return slug


class DataCache:
    """Parquet-based per-ticker cache with TTL.

    Each ticker produces two files:
        {cache_dir}/{ticker}.parquet   — OHLCV DataFrame
        {cache_dir}/{ticker}.meta.json — fetched_at, source, market_type
    """

    def __init__(self, cache_dir: Path, ttl_hours: int) -> None:
        self._dir = cache_dir
        self._ttl = timedelta(hours=ttl_hours)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, ticker: str) -> MarketData | None:
        """Return cached MarketData if present and within TTL, else None."""
        if not self.is_valid(ticker):
            return None
        try:
            parquet_path = self._parquet_path(ticker)
            size = parquet_path.stat().st_size
            if size > _MAX_PARQUET_BYTES:
                logger.warning("Cache parquet for %s exceeds size limit (%d bytes); skipping", ticker, size)
                return None
            df = pd.read_parquet(parquet_path)
            meta = self._read_meta(ticker)
            return MarketData(
                ticker=ticker,
                market_type=meta["market_type"],
                fetched_at=datetime.fromisoformat(meta["fetched_at"]),
                source=meta["source"],
                df=df,
            )
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", ticker, exc)
            return None

    def set(self, data: MarketData) -> None:
        """Persist MarketData to Parquet + JSON sidecar."""
        try:
            data.df.to_parquet(self._parquet_path(data.ticker))
            meta = {
                "ticker": data.ticker,
                "market_type": data.market_type,
                "source": data.source,
                "fetched_at": data.fetched_at.isoformat(),
            }
            self._meta_path(data.ticker).write_text(json.dumps(meta))
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", data.ticker, exc)

    def is_valid(self, ticker: str) -> bool:
        """True if cache entry exists and is within TTL."""
        meta_path = self._meta_path(ticker)
        parquet_path = self._parquet_path(ticker)
        if not (meta_path.exists() and parquet_path.exists()):
            return False
        try:
            meta = self._read_meta(ticker)
            fetched_at = datetime.fromisoformat(meta["fetched_at"])
            return datetime.now(timezone.utc) - fetched_at < self._ttl
        except Exception:
            return False

    def invalidate(self, ticker: str | None = None) -> None:
        """Remove one ticker's cache files, or all entries if ticker is None."""
        if ticker is not None:
            self._parquet_path(ticker).unlink(missing_ok=True)
            self._meta_path(ticker).unlink(missing_ok=True)
        else:
            for f in self._dir.glob("*.parquet"):
                f.unlink(missing_ok=True)
            for f in self._dir.glob("*.meta.json"):
                f.unlink(missing_ok=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parquet_path(self, ticker: str) -> Path:
        return self._dir / f"{_safe_slug(ticker)}.parquet"

    def _meta_path(self, ticker: str) -> Path:
        return self._dir / f"{_safe_slug(ticker)}.meta.json"

    def _read_meta(self, ticker: str) -> dict:
        data = json.loads(self._meta_path(ticker).read_text())
        for key in ("ticker", "market_type", "source", "fetched_at"):
            if key not in data:
                raise ValueError(f"Cache metadata for {ticker!r} missing required key {key!r}")
        return data
