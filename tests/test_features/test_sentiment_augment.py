"""Tests for features/pipeline.py — augment_with_sentiment()."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from data.models import SentimentSnapshot
from features.pipeline import _SENTIMENT_COLS, augment_with_sentiment
from tests.test_analysis.conftest import make_enriched


def _snap(score=0.5, velocity=0.1, bullish=0.6, engagement=1.5):
    return SentimentSnapshot(
        ticker="TEST", window_hours=24, mention_count=10,
        sentiment_score=score, sentiment_velocity=velocity,
        bullish_ratio=bullish, engagement_score=engagement,
        fetched_at=datetime.now(timezone.utc),
    )


class TestAugmentWithSentiment:

    def test_adds_four_columns_when_snapshot_none(self):
        enriched = make_enriched()
        result = augment_with_sentiment(enriched, None)
        for col in _SENTIMENT_COLS:
            assert col in result.df.columns

    def test_all_rows_zero_when_snapshot_none(self):
        enriched = make_enriched()
        result = augment_with_sentiment(enriched, None)
        for col in _SENTIMENT_COLS:
            assert (result.df[col] == 0.0).all(), f"{col} should be all zeros"

    def test_last_row_gets_snapshot_values(self):
        enriched = make_enriched()
        snap = _snap(score=0.42, velocity=0.15, bullish=0.7, engagement=2.3)
        result = augment_with_sentiment(enriched, snap)
        last = result.df.iloc[-1]
        assert last["sentiment_score"] == pytest.approx(0.42)
        assert last["sentiment_velocity"] == pytest.approx(0.15)
        assert last["bullish_ratio"] == pytest.approx(0.7)
        assert last["engagement_score"] == pytest.approx(2.3)

    def test_non_last_rows_remain_zero_with_snapshot(self):
        enriched = make_enriched(n=100)
        snap = _snap()
        result = augment_with_sentiment(enriched, snap)
        # All rows except the last should be 0 for sentiment cols
        for col in _SENTIMENT_COLS:
            assert (result.df[col].iloc[:-1] == 0.0).all(), \
                f"Non-last row of {col} should be zero"

    def test_feature_names_extended(self):
        enriched = make_enriched()
        result = augment_with_sentiment(enriched, _snap())
        for col in _SENTIMENT_COLS:
            assert col in result.feature_names

    def test_returns_new_enriched_data_not_mutation(self):
        enriched = make_enriched()
        original_last_score = float(enriched.df["sentiment_score"].iloc[-1])
        result = augment_with_sentiment(enriched, _snap(score=0.77))
        # Original df should not have been mutated
        assert float(enriched.df["sentiment_score"].iloc[-1]) == original_last_score
        # Result should reflect the snapshot
        assert result.df["sentiment_score"].iloc[-1] == pytest.approx(0.77)

    def test_ticker_and_market_type_preserved(self):
        enriched = make_enriched(ticker="NVDA")
        result = augment_with_sentiment(enriched, _snap())
        assert result.ticker == "NVDA"
        assert result.market_type == enriched.market_type

    def test_duplicate_feature_names_not_added(self):
        enriched = make_enriched()
        # Augment twice
        result = augment_with_sentiment(enriched, _snap())
        result2 = augment_with_sentiment(result, _snap(score=0.9))
        count = sum(1 for n in result2.feature_names if n == "sentiment_score")
        assert count == 1, "Should not duplicate sentiment_score in feature_names"
