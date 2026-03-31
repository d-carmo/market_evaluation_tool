"""Tests for sentiment/scorer.py — scoring backends."""
from __future__ import annotations

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _texts(entries: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Build (text, 1.0) tuples from (text, _) pairs."""
    return [(t, 1.0) for t, _ in entries]


# ── VADER fallback tests ──────────────────────────────────────────────────────

class TestScoreTextsVader:
    """Tests that run with only VADER installed (no torch required)."""

    def test_empty_input_returns_neutral(self):
        from sentiment.scorer import score_texts
        score, bullish, eng = score_texts([])
        assert score == 0.0
        assert bullish == 0.5
        assert eng == 0.0

    def test_clearly_positive_texts(self, monkeypatch):
        """Force VADER path by disabling FinBERT."""
        from sentiment import scorer
        monkeypatch.setattr(scorer, "_load_finbert", lambda: None)

        texts = [
            ("The stock is absolutely booming and profits are incredible!", 1.0),
            ("Fantastic earnings, massive gains, investors love it!", 1.0),
            ("Best quarter ever, stock price soaring, bullish signal!", 1.0),
        ]
        score, bullish, eng = scorer.score_texts(texts)
        assert score > 0, f"Expected positive score, got {score}"
        assert 0.0 <= bullish <= 1.0
        assert eng > 0.0

    def test_clearly_negative_texts(self, monkeypatch):
        from sentiment import scorer
        monkeypatch.setattr(scorer, "_load_finbert", lambda: None)

        texts = [
            ("Terrible losses, the stock is crashing badly!", 1.0),
            ("Horrible results, disaster, sell everything now!", 1.0),
            ("Stock in free fall, awful earnings, horrible news!", 1.0),
        ]
        score, bullish, eng = scorer.score_texts(texts)
        assert score < 0, f"Expected negative score, got {score}"

    def test_weighted_engagement(self, monkeypatch):
        """Higher-engagement post should pull score toward its sentiment."""
        from sentiment import scorer
        monkeypatch.setattr(scorer, "_load_finbert", lambda: None)

        texts = [
            ("Absolutely terrible loss crash disaster", 1.0),   # negative, low weight
            ("Amazing gains profits incredible!", 100.0),         # positive, HIGH weight
        ]
        score, _, _ = scorer.score_texts(texts)
        assert score > 0, "High-weight positive post should dominate"

    def test_returns_tuple_of_three_floats(self, monkeypatch):
        from sentiment import scorer
        monkeypatch.setattr(scorer, "_load_finbert", lambda: None)

        result = scorer.score_texts([("ok news today", 1.0)])
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_score_in_range(self, monkeypatch):
        from sentiment import scorer
        monkeypatch.setattr(scorer, "_load_finbert", lambda: None)

        texts = [(f"word{i}", 1.0) for i in range(20)]
        score, bullish, eng = scorer.score_texts(texts)
        assert -1.0 <= score <= 1.0
        assert 0.0 <= bullish <= 1.0

    def test_no_backend_returns_neutral(self, monkeypatch):
        """When both FinBERT and VADER are unavailable, return zeros."""
        from sentiment import scorer
        monkeypatch.setattr(scorer, "_load_finbert", lambda: None)
        monkeypatch.setattr(scorer, "_load_vader", lambda: None)

        score, bullish, eng = scorer.score_texts([("some text", 1.0)])
        assert score == 0.0
        assert bullish == 0.5
        assert eng == 0.0
