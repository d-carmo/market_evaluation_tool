"""NLP sentiment scoring: FinBERT (preferred) with VADER fallback.

Both backends return (sentiment_score, bullish_ratio, engagement) where:
  sentiment_score ∈ [-1, 1]   — negative to positive
  bullish_ratio   ∈ [0, 1]    — fraction of clearly bullish texts
  engagement      ∈ [0, ∞)    — mean engagement weight across posts
"""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Minimum confidence delta before a FinBERT prediction is counted as bullish/bearish
_FINBERT_THRESHOLD = 0.3
_VADER_THRESHOLD = 0.05


@lru_cache(maxsize=1)
def _load_finbert():
    """Load FinBERT pipeline once and cache.  Returns None on import failure."""
    try:
        from transformers import pipeline  # noqa: PLC0415
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
            device=-1,  # CPU; set to 0 for GPU
        )
        logger.info("Loaded FinBERT sentiment model (ProsusAI/finbert)")
        return pipe
    except Exception as exc:
        logger.info("FinBERT not available (%s) — will try VADER", exc)
        return None


@lru_cache(maxsize=1)
def _load_vader():
    """Load VADER SentimentIntensityAnalyzer.  Returns None if not installed."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: PLC0415
        logger.info("Loaded VADER sentiment analyser")
        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning(
            "vaderSentiment not installed — no sentiment backend available. "
            "Run: pip install vaderSentiment"
        )
        return None


def score_texts(
    texts: list[tuple[str, float]],
) -> tuple[float, float, float]:
    """Score a batch of (text, engagement_weight) pairs.

    Tries FinBERT first, falls back to VADER.

    Returns:
        (sentiment_score, bullish_ratio, mean_engagement)
    """
    if not texts:
        return 0.0, 0.5, 0.0

    pipe = _load_finbert()
    if pipe is not None:
        try:
            return _score_finbert(texts, pipe)
        except Exception as exc:
            logger.warning("FinBERT scoring failed, falling back to VADER: %s", exc)

    vader = _load_vader()
    if vader is not None:
        return _score_vader(texts, vader)

    return 0.0, 0.5, 0.0


# ── backend implementations ───────────────────────────────────────────────────

def _score_finbert(
    texts: list[tuple[str, float]],
    pipe,
) -> tuple[float, float, float]:
    batch = [t[:512] for t, _ in texts]
    weights = [w for _, w in texts]

    results = pipe(batch, batch_size=8, truncation=True, max_length=512)

    total_w = 0.0
    weighted_score = 0.0
    bullish_count = 0

    for i, label_scores in enumerate(results):
        w = weights[i]
        # label_scores is a list of {"label": ..., "score": ...} dicts
        scores_map = {d["label"].lower(): d["score"] for d in label_scores}
        pos = scores_map.get("positive", 0.0)
        neg = scores_map.get("negative", 0.0)
        text_score = pos - neg          # ∈ [-1, 1]

        weighted_score += text_score * w
        total_w += w
        if text_score > _FINBERT_THRESHOLD:
            bullish_count += 1

    if total_w == 0:
        return 0.0, 0.5, 0.0

    avg_score = float(weighted_score / total_w)
    bullish_ratio = float(bullish_count / len(texts))
    mean_engagement = float(total_w / len(texts))
    return avg_score, bullish_ratio, mean_engagement


def _score_vader(
    texts: list[tuple[str, float]],
    vader,
) -> tuple[float, float, float]:
    total_w = 0.0
    weighted_score = 0.0
    bullish_count = 0

    for text, w in texts:
        vs = vader.polarity_scores(text)
        text_score = float(vs["compound"])   # VADER compound ∈ [-1, 1]

        weighted_score += text_score * w
        total_w += w
        if text_score > _VADER_THRESHOLD:
            bullish_count += 1

    if total_w == 0:
        return 0.0, 0.5, 0.0

    avg_score = float(weighted_score / total_w)
    bullish_ratio = float(bullish_count / len(texts))
    mean_engagement = float(total_w / len(texts))
    return avg_score, bullish_ratio, mean_engagement
