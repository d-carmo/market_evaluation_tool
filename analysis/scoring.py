import logging

from data.models import Action, CompositeScore, Signal, TrendResult

logger = logging.getLogger(__name__)

BUY_THRESHOLD = 0.25
SELL_THRESHOLD = -0.25


def compute_score(
    signals: list[Signal], trend: TrendResult
) -> tuple[float, Action, float]:
    """Compute weighted composite score, action, and confidence.

    Returns (score, action, confidence) where:
        score      — weighted sum normalized to [-1.0, 1.0] with trend bias
        action     — BUY / HOLD / SELL based on thresholds
        confidence — fraction of signals agreeing with the final action
    """
    if not signals:
        return 0.0, "HOLD", 0.0

    weighted_sum = sum(s.value * s.weight for s in signals)
    max_possible = sum(abs(s.weight) for s in signals)
    score = weighted_sum / max_possible if max_possible > 0 else 0.0

    # Trend bias
    if trend.direction == "BULLISH":
        score += 0.1 * trend.strength
    elif trend.direction == "BEARISH":
        score -= 0.1 * trend.strength

    score = max(-1.0, min(1.0, score))

    action: Action
    if score > BUY_THRESHOLD:
        action = "BUY"
    elif score < SELL_THRESHOLD:
        action = "SELL"
    else:
        action = "HOLD"

    # Confidence: fraction of signals whose value agrees with the action
    target_value = {"BUY": 1, "SELL": -1, "HOLD": 0}[action]
    agreeing = sum(1 for s in signals if s.value == target_value)
    confidence = agreeing / len(signals)

    return score, action, confidence


def build_composite(
    ticker: str,
    signals: list[Signal],
    trend: TrendResult,
) -> CompositeScore:
    """Assemble CompositeScore from signals and trend."""
    score, action, confidence = compute_score(signals, trend)
    return CompositeScore(
        ticker=ticker,
        score=score,
        action=action,
        confidence=confidence,
        signals=signals,
        trend=trend,
    )
