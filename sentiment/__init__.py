"""Sentiment package: Reddit post fetching + NLP scoring → SentimentSnapshot.

Public API
----------
get_sentiment_snapshot(ticker, market_type, ...) -> SentimentSnapshot | None
    Returns None when Reddit credentials are absent or all fetch/score attempts fail.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from data.models import SentimentSnapshot

logger = logging.getLogger(__name__)


def get_sentiment_snapshot(
    ticker: str,
    market_type: str,
    reddit_client_id: str,
    reddit_client_secret: str,
    user_agent: str = "market_eval/2.0",
    cache_dir: Path | None = None,
    ttl_hours: int = 1,
    window_hours: int = 24,
    prev_score: float | None = None,
) -> SentimentSnapshot | None:
    """Fetch Reddit posts for `ticker` and return a SentimentSnapshot.

    Args:
        ticker:               Ticker symbol (e.g. "AAPL", "BTC-USD").
        market_type:          "stock" | "crypto" | "commodity".
        reddit_client_id:     PRAW application client ID.
        reddit_client_secret: PRAW application client secret.
        user_agent:           HTTP user-agent string for PRAW.
        cache_dir:            Directory for file-based TTL cache of raw posts.
        ttl_hours:            How long to reuse cached posts before re-fetching.
        window_hours:         Label for how wide the look-back window was.
        prev_score:           Previous sentiment_score for velocity calculation.

    Returns:
        SentimentSnapshot on success (mention_count may be 0 if nothing found),
        or None when credentials are missing or the entire pipeline fails.
    """
    if not reddit_client_id or not reddit_client_secret:
        return None

    from data.fetchers.reddit import RedditFetcher
    from sentiment.scorer import score_texts

    fetcher = RedditFetcher(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=user_agent,
        cache_dir=cache_dir,
        ttl_hours=ttl_hours,
    )

    try:
        posts = fetcher.fetch(ticker, market_type=market_type)
    except Exception as exc:
        logger.warning("Reddit fetch failed for %s: %s", ticker, exc)
        return None

    now = datetime.now(timezone.utc)

    if not posts:
        return SentimentSnapshot(
            ticker=ticker,
            window_hours=window_hours,
            mention_count=0,
            sentiment_score=0.0,
            sentiment_velocity=0.0,
            bullish_ratio=0.5,
            engagement_score=0.0,
            fetched_at=now,
        )

    try:
        sentiment_score, bullish_ratio, engagement = score_texts(posts)
    except Exception as exc:
        logger.warning("Sentiment scoring failed for %s: %s", ticker, exc)
        return None

    velocity = 0.0
    if prev_score is not None:
        velocity = float(sentiment_score - prev_score)

    return SentimentSnapshot(
        ticker=ticker,
        window_hours=window_hours,
        mention_count=len(posts),
        sentiment_score=float(sentiment_score),
        sentiment_velocity=float(velocity),
        bullish_ratio=float(bullish_ratio),
        engagement_score=float(engagement),
        fetched_at=now,
    )
