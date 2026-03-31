"""Tests for sentiment_signal() and updated compute_signals()."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analysis.signals import compute_signals, sentiment_signal
from data.models import SentimentSnapshot
from tests.test_analysis.conftest import make_enriched
from config import SignalWeights


def _snap(score: float, mentions: int = 10) -> SentimentSnapshot:
    return SentimentSnapshot(
        ticker="TEST", window_hours=24, mention_count=mentions,
        sentiment_score=score, sentiment_velocity=0.0,
        bullish_ratio=0.6, engagement_score=1.0,
        fetched_at=datetime.now(timezone.utc),
    )


class TestSentimentSignal:

    def test_none_snapshot_returns_hold(self):
        sig = sentiment_signal(None, weight=0.5)
        assert sig.value == 0
        assert sig.name == "sentiment"
        assert sig.raw == 0.0

    def test_too_few_mentions_returns_hold(self):
        snap = _snap(score=0.9, mentions=3)
        sig = sentiment_signal(snap, weight=0.5)
        assert sig.value == 0

    def test_exactly_five_mentions_uses_score(self):
        snap = _snap(score=0.9, mentions=5)
        sig = sentiment_signal(snap, weight=0.5)
        assert sig.value == 1   # > 0.3 threshold

    def test_strong_positive_gives_buy(self):
        sig = sentiment_signal(_snap(0.6), weight=0.5)
        assert sig.value == 1

    def test_at_threshold_gives_buy(self):
        # Exactly 0.3 is NOT > 0.3, so should be HOLD
        sig = sentiment_signal(_snap(0.3), weight=0.5)
        assert sig.value == 0

    def test_just_above_threshold_gives_buy(self):
        sig = sentiment_signal(_snap(0.31), weight=0.5)
        assert sig.value == 1

    def test_strong_negative_gives_sell(self):
        sig = sentiment_signal(_snap(-0.6), weight=0.5)
        assert sig.value == -1

    def test_at_negative_threshold_gives_hold(self):
        sig = sentiment_signal(_snap(-0.3), weight=0.5)
        assert sig.value == 0

    def test_just_below_negative_threshold_gives_sell(self):
        sig = sentiment_signal(_snap(-0.31), weight=0.5)
        assert sig.value == -1

    def test_neutral_sentiment_gives_hold(self):
        sig = sentiment_signal(_snap(0.0), weight=0.5)
        assert sig.value == 0

    def test_weight_stored_on_signal(self):
        sig = sentiment_signal(_snap(0.5), weight=1.23)
        assert sig.weight == pytest.approx(1.23)

    def test_raw_stores_sentiment_score(self):
        snap = _snap(0.55)
        sig = sentiment_signal(snap, weight=0.5)
        assert sig.raw == pytest.approx(0.55)


class TestComputeSignalsWithSentiment:

    def test_compute_signals_returns_seven_signals(self):
        data = make_enriched()
        weights = SignalWeights()
        sigs = compute_signals(data, weights, snapshot=None)
        assert len(sigs) == 7
        names = [s.name for s in sigs]
        assert "sentiment" in names

    def test_compute_signals_no_snapshot_sentiment_is_hold(self):
        data = make_enriched()
        weights = SignalWeights()
        sigs = compute_signals(data, weights, snapshot=None)
        sentiment = next(s for s in sigs if s.name == "sentiment")
        assert sentiment.value == 0

    def test_compute_signals_positive_snapshot_propagates(self):
        data = make_enriched()
        weights = SignalWeights(sentiment=0.5)
        snap = _snap(0.8)
        sigs = compute_signals(data, weights, snapshot=snap)
        sentiment = next(s for s in sigs if s.name == "sentiment")
        assert sentiment.value == 1

    def test_compute_signals_negative_snapshot_propagates(self):
        data = make_enriched()
        weights = SignalWeights(sentiment=0.5)
        snap = _snap(-0.8)
        sigs = compute_signals(data, weights, snapshot=snap)
        sentiment = next(s for s in sigs if s.name == "sentiment")
        assert sentiment.value == -1
