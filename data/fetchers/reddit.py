"""Reddit data fetcher using PRAW with file-based TTL caching."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Subreddits to query per market type
_SUBREDDITS_STOCKS: list[str] = ["wallstreetbets", "stocks", "investing"]
_SUBREDDITS_CRYPTO: list[str] = ["cryptocurrency", "CryptoCurrency", "CryptoMarkets"]
_SUBREDDITS_COMMODITY: list[str] = ["investing", "economics", "commodities"]


def _cache_key(ticker: str) -> str:
    return hashlib.md5(ticker.encode()).hexdigest()[:16]


class RedditFetcher:
    """Fetches Reddit posts mentioning a ticker from relevant subreddits.

    Results are cached on disk for `ttl_hours` to respect rate limits.
    Returns an empty list (not an error) when credentials are absent or API
    calls fail — callers should treat an empty list as "no data available".
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str = "market_eval/2.0 (by /u/market_eval_bot)",
        cache_dir: Path | None = None,
        ttl_hours: int = 1,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._cache_dir = (cache_dir or Path(".cache/reddit")).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._ttl = ttl_hours * 3600

    # ── public ────────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker: str,
        market_type: str = "stock",
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        """Return list of (text, engagement_weight) for recent ticker mentions.

        engagement_weight = sqrt(upvote_ratio * max(num_comments, 1))
        Text is truncated to 512 chars.  Returns [] on any failure.
        """
        cached = self._load_cache(ticker)
        if cached is not None:
            return cached

        subreddits = self._subreddits_for(market_type)
        # For crypto strip the "-USD" suffix: "BTC-USD" → "BTC"
        symbol = ticker.split("-")[0].upper()

        results: list[tuple[str, float]] = []
        try:
            reddit = self._build_client()
            per_sub = max(1, limit // len(subreddits))
            for sub_name in subreddits:
                try:
                    sub = reddit.subreddit(sub_name)
                    for post in sub.search(symbol, limit=per_sub, time_filter="day"):
                        text = f"{post.title} {post.selftext or ''}".strip()
                        if not text:
                            continue
                        engagement = float(
                            (post.upvote_ratio * max(post.num_comments, 1)) ** 0.5
                        )
                        results.append((text[:512], engagement))
                except Exception as exc:
                    logger.debug("Skipping r/%s for %s: %s", sub_name, ticker, exc)
        except ImportError:
            logger.warning(
                "praw is not installed — Reddit sentiment disabled. "
                "Run: pip install praw"
            )
            return []
        except Exception as exc:
            logger.warning("Reddit API error for %s: %s", ticker, exc)
            return []

        if results:
            self._save_cache(ticker, results)
        return results

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _subreddits_for(market_type: str) -> list[str]:
        if market_type == "crypto":
            return _SUBREDDITS_CRYPTO
        if market_type == "commodity":
            return _SUBREDDITS_COMMODITY
        return _SUBREDDITS_STOCKS

    def _build_client(self):
        import praw  # noqa: PLC0415
        return praw.Reddit(
            client_id=self._client_id,
            client_secret=self._client_secret,
            user_agent=self._user_agent,
        )

    def _cache_path(self, ticker: str) -> Path:
        return self._cache_dir / f"{_cache_key(ticker)}.json"

    def _load_cache(self, ticker: str) -> list[tuple[str, float]] | None:
        p = self._cache_path(ticker)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            if time.time() - data["ts"] < self._ttl:
                return [tuple(item) for item in data["posts"]]  # type: ignore[return-value]
        except Exception:
            pass
        return None

    def _save_cache(self, ticker: str, posts: list[tuple[str, float]]) -> None:
        p = self._cache_path(ticker)
        try:
            p.write_text(json.dumps({"ts": time.time(), "posts": posts}))
        except Exception as exc:
            logger.debug("Cache write failed for %s: %s", ticker, exc)
