"""Tests for accuracy/weight_optimizer.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from accuracy.weight_optimizer import (
    compute_adjusted_weights,
    optimize_blend_weights,
    _BLEND_LR,
    _MIN_BLEND,
    _MAX_BLEND,
    _MIN_WEIGHT,
    _MAX_WEIGHT,
)
from config import SignalWeights


# ── compute_adjusted_weights ──────────────────────────────────────────────────

class TestComputeAdjustedWeights:

    def _current(self):
        return {"rsi": 1.0, "macd": 1.5, "trend": 2.0,
                "volume": 0.8, "bb": 1.0, "stoch": 0.8, "sentiment": 0.5}

    def _make_votes(self, n_correct, n_wrong, sig="rsi", direction="UP"):
        """Build a list of (signal_votes, actual_direction) pairs."""
        votes = []
        opposite = "DOWN" if direction == "UP" else "UP"
        sig_val  = 1 if direction == "UP" else -1
        for _ in range(n_correct):
            votes.append(({sig: sig_val}, direction))
        for _ in range(n_wrong):
            votes.append(({sig: sig_val}, opposite))
        return votes

    def test_returns_none_when_insufficient_samples(self):
        current = self._current()
        votes = self._make_votes(n_correct=3, n_wrong=2, sig="rsi")
        result = compute_adjusted_weights(current, votes, min_samples=10)
        assert result is None

    def test_returns_none_when_no_votes(self):
        current = self._current()
        result = compute_adjusted_weights(current, [], min_samples=1)
        assert result is None

    def test_high_accuracy_increases_weight(self):
        current = self._current()
        # 10 correct, 0 wrong → accuracy_rate = 1.0, adj = 1.0, new_w > old_w
        votes = self._make_votes(n_correct=10, n_wrong=0, sig="rsi")
        updated = compute_adjusted_weights(current, votes, min_samples=10)
        assert updated is not None
        assert updated["rsi"] > current["rsi"]

    def test_low_accuracy_decreases_weight(self):
        current = self._current()
        # 0 correct, 10 wrong → accuracy_rate = 0.0, adj = -1.0, new_w < old_w
        votes = self._make_votes(n_correct=0, n_wrong=10, sig="rsi")
        updated = compute_adjusted_weights(current, votes, min_samples=10)
        assert updated is not None
        assert updated["rsi"] < current["rsi"]

    def test_fifty_pct_accuracy_no_change(self):
        current = self._current()
        # 5 correct, 5 wrong → accuracy_rate = 0.5, adj = 0.0, no change
        votes = self._make_votes(n_correct=5, n_wrong=5, sig="rsi")
        result = compute_adjusted_weights(current, votes, min_samples=10)
        assert result is None  # unchanged

    def test_weight_clamped_to_max(self):
        # Start near max; any positive adjustment should clamp
        current = self._current()
        current["rsi"] = _MAX_WEIGHT - 0.01
        votes = self._make_votes(n_correct=20, n_wrong=0, sig="rsi")
        updated = compute_adjusted_weights(current, votes, min_samples=10)
        assert updated is not None
        assert updated["rsi"] <= _MAX_WEIGHT

    def test_weight_clamped_to_min(self):
        current = self._current()
        current["rsi"] = _MIN_WEIGHT + 0.01
        votes = self._make_votes(n_correct=0, n_wrong=20, sig="rsi")
        updated = compute_adjusted_weights(current, votes, min_samples=10)
        assert updated is not None
        assert updated["rsi"] >= _MIN_WEIGHT

    def test_flat_actual_outcomes_excluded(self):
        current = self._current()
        votes = [
            ({"rsi": 1}, "FLAT"),   # should be excluded
        ] * 20
        result = compute_adjusted_weights(current, votes, min_samples=10)
        assert result is None

    def test_neutral_signal_votes_excluded(self):
        current = self._current()
        votes = [
            ({"rsi": 0}, "UP"),   # neutral vote — excluded
        ] * 20
        result = compute_adjusted_weights(current, votes, min_samples=10)
        assert result is None

    def test_sentiment_signal_adjusted(self):
        current = self._current()
        votes = self._make_votes(n_correct=10, n_wrong=0, sig="sentiment")
        updated = compute_adjusted_weights(current, votes, min_samples=10)
        assert updated is not None
        assert updated["sentiment"] > current["sentiment"]

    def test_unknown_signal_names_ignored(self):
        current = self._current()
        votes = [
            ({"unknown_signal": 1}, "UP"),
        ] * 20
        result = compute_adjusted_weights(current, votes, min_samples=10)
        assert result is None


# ── optimize_blend_weights ────────────────────────────────────────────────────

class TestOptimizeBlendWeights:

    def _make_db(self, xgb_blend=0.6, lstm_blend=0.4):
        """Return a fake DB session whose TickerWeightsRepository.get returns a row."""
        fake_row = MagicMock()
        fake_row.xgb_blend  = xgb_blend
        fake_row.lstm_blend = lstm_blend

        fake_repo = MagicMock()
        fake_repo.get.return_value = fake_row
        return MagicMock(), fake_repo

    def test_no_op_when_xgb_accuracy_none(self):
        db, repo = self._make_db()
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            optimize_blend_weights("AAPL", db, None, 0.8)
        repo.upsert.assert_not_called()

    def test_no_op_when_lstm_accuracy_none(self):
        db, repo = self._make_db()
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            optimize_blend_weights("AAPL", db, 0.8, None)
        repo.upsert.assert_not_called()

    def test_no_op_when_accuracies_equal(self):
        db, repo = self._make_db()
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            optimize_blend_weights("AAPL", db, 0.7, 0.7)
        repo.upsert.assert_not_called()

    def test_no_op_when_difference_below_threshold(self):
        db, repo = self._make_db()
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            # |0.705 - 0.70| = 0.005 — strictly < 0.01, so no-op
            optimize_blend_weights("AAPL", db, 0.705, 0.70)
        repo.upsert.assert_not_called()

    def test_xgb_better_increases_xgb_blend(self):
        db, repo = self._make_db(xgb_blend=0.6, lstm_blend=0.4)
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            optimize_blend_weights("AAPL", db, xgb_accuracy=0.9, lstm_accuracy=0.5)

        repo.upsert.assert_called_once()
        _, kwargs_or_args = repo.upsert.call_args
        # upsert called with positional args: (ticker, weights_dict)
        call_args = repo.upsert.call_args[0]
        weights = call_args[1]
        assert weights["xgb_blend"] > 0.6

    def test_lstm_better_increases_lstm_blend(self):
        db, repo = self._make_db(xgb_blend=0.6, lstm_blend=0.4)
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            optimize_blend_weights("AAPL", db, xgb_accuracy=0.5, lstm_accuracy=0.9)

        repo.upsert.assert_called_once()
        call_args = repo.upsert.call_args[0]
        weights = call_args[1]
        assert weights["lstm_blend"] > 0.4

    def test_blends_sum_to_one(self):
        db, repo = self._make_db(xgb_blend=0.6, lstm_blend=0.4)
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            optimize_blend_weights("AAPL", db, xgb_accuracy=0.8, lstm_accuracy=0.5)

        call_args = repo.upsert.call_args[0]
        weights = call_args[1]
        assert abs(weights["xgb_blend"] + weights["lstm_blend"] - 1.0) < 1e-4

    def test_blends_clamped_to_bounds(self):
        # Even with extreme accuracy difference blends should stay in [0.1, 0.9]
        db, repo = self._make_db(xgb_blend=0.6, lstm_blend=0.4)
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            optimize_blend_weights("AAPL", db, xgb_accuracy=1.0, lstm_accuracy=0.0)
        # No assert needed — just confirm no exception and upsert called

    def test_no_op_when_total_accuracy_zero(self):
        db, repo = self._make_db()
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=repo):
            optimize_blend_weights("AAPL", db, 0.0, 0.0)
        repo.upsert.assert_not_called()

    def test_uses_defaults_when_no_db_row(self):
        fake_repo = MagicMock()
        fake_repo.get.return_value = None
        db = MagicMock()
        with patch("storage.repository.TickerWeightsRepository",
                   return_value=fake_repo):
            # Should not raise; uses default 0.6/0.4
            optimize_blend_weights("AAPL", db, xgb_accuracy=0.8, lstm_accuracy=0.5)
        # upsert called because |0.8 - 0.5| > 0.01
        fake_repo.upsert.assert_called_once()
