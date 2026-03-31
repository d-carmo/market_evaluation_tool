import pytest

from analysis.scoring import BUY_THRESHOLD, SELL_THRESHOLD, compute_score
from data.models import Signal, TrendResult


def _make_trend(direction="SIDEWAYS", strength=0.5) -> TrendResult:
    return TrendResult(
        ticker="TEST", direction=direction, strength=strength,
        ma_aligned=False, slope=0.0, adx=25.0,
    )


def _signals(value: int, n: int = 4, weight: float = 1.0) -> list[Signal]:
    return [Signal(f"s{i}", value, 0.0, weight) for i in range(n)]


def test_all_buy_signals_gives_buy():
    score, action, _ = compute_score(_signals(1), _make_trend())
    assert action == "BUY"
    assert score > BUY_THRESHOLD


def test_all_sell_signals_gives_sell():
    score, action, _ = compute_score(_signals(-1), _make_trend())
    assert action == "SELL"
    assert score < SELL_THRESHOLD


def test_score_clamped_upper():
    trend = _make_trend("BULLISH", strength=1.0)
    score, _, _ = compute_score(_signals(1, weight=10.0), trend)
    assert score <= 1.0


def test_score_clamped_lower():
    trend = _make_trend("BEARISH", strength=1.0)
    score, _, _ = compute_score(_signals(-1, weight=10.0), trend)
    assert score >= -1.0


def test_bullish_trend_raises_score():
    signals = _signals(0)  # neutral signals
    _, _, _ = compute_score(signals, _make_trend("SIDEWAYS", 0.5))
    score_bull, _, _ = compute_score(signals, _make_trend("BULLISH", 0.5))
    score_side, _, _ = compute_score(signals, _make_trend("SIDEWAYS", 0.5))
    assert score_bull > score_side


def test_confidence_in_range():
    _, _, conf = compute_score(_signals(1), _make_trend())
    assert 0.0 <= conf <= 1.0


def test_confidence_all_agree():
    _, action, conf = compute_score(_signals(1), _make_trend())
    assert action == "BUY"
    assert conf == 1.0


def test_empty_signals_returns_hold():
    score, action, conf = compute_score([], _make_trend())
    assert action == "HOLD"
    assert score == 0.0
