"""Tests for sentiment/__init__.py — get_sentiment_snapshot()."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestGetSentimentSnapshot:

    def test_returns_none_when_no_credentials(self):
        from sentiment import get_sentiment_snapshot
        result = get_sentiment_snapshot(
            ticker="AAPL", market_type="stock",
            reddit_client_id="", reddit_client_secret="",
        )
        assert result is None

    def test_returns_none_when_client_id_missing(self):
        from sentiment import get_sentiment_snapshot
        result = get_sentiment_snapshot(
            ticker="AAPL", market_type="stock",
            reddit_client_id="", reddit_client_secret="secret",
        )
        assert result is None

    def test_returns_none_when_fetcher_raises(self, tmp_path):
        from sentiment import get_sentiment_snapshot
        with patch("data.fetchers.reddit.RedditFetcher.fetch",
                   side_effect=RuntimeError("network error")):
            result = get_sentiment_snapshot(
                ticker="AAPL", market_type="stock",
                reddit_client_id="cid", reddit_client_secret="csec",
                cache_dir=tmp_path,
            )
        assert result is None

    def test_returns_zero_snapshot_when_no_posts(self, tmp_path):
        from sentiment import get_sentiment_snapshot
        with patch("data.fetchers.reddit.RedditFetcher.fetch", return_value=[]):
            snap = get_sentiment_snapshot(
                ticker="AAPL", market_type="stock",
                reddit_client_id="cid", reddit_client_secret="csec",
                cache_dir=tmp_path,
            )
        assert snap is not None
        assert snap.mention_count == 0
        assert snap.sentiment_score == 0.0
        assert snap.ticker == "AAPL"

    def test_returns_snapshot_with_posts(self, tmp_path):
        from sentiment import get_sentiment_snapshot
        fake_posts = [("AAPL is doing great today!", 2.0),
                      ("Terrible results, sell now.", 1.0)]
        with patch("data.fetchers.reddit.RedditFetcher.fetch", return_value=fake_posts), \
             patch("sentiment.scorer.score_texts", return_value=(0.4, 0.7, 1.5)):
            snap = get_sentiment_snapshot(
                ticker="AAPL", market_type="stock",
                reddit_client_id="cid", reddit_client_secret="csec",
                cache_dir=tmp_path,
            )
        assert snap is not None
        assert snap.mention_count == 2
        assert snap.sentiment_score == pytest.approx(0.4)
        assert snap.bullish_ratio == pytest.approx(0.7)
        assert snap.engagement_score == pytest.approx(1.5)

    def test_velocity_computed_from_prev_score(self, tmp_path):
        from sentiment import get_sentiment_snapshot
        with patch("data.fetchers.reddit.RedditFetcher.fetch",
                   return_value=[("great!", 1.0)]), \
             patch("sentiment.scorer.score_texts", return_value=(0.6, 0.8, 1.0)):
            snap = get_sentiment_snapshot(
                ticker="AAPL", market_type="stock",
                reddit_client_id="cid", reddit_client_secret="csec",
                cache_dir=tmp_path,
                prev_score=0.3,
            )
        assert snap is not None
        assert snap.sentiment_velocity == pytest.approx(0.6 - 0.3)

    def test_returns_none_when_scoring_raises(self, tmp_path):
        from sentiment import get_sentiment_snapshot
        with patch("data.fetchers.reddit.RedditFetcher.fetch",
                   return_value=[("text", 1.0)]), \
             patch("sentiment.scorer.score_texts",
                   side_effect=RuntimeError("model error")):
            result = get_sentiment_snapshot(
                ticker="AAPL", market_type="stock",
                reddit_client_id="cid", reddit_client_secret="csec",
                cache_dir=tmp_path,
            )
        assert result is None
