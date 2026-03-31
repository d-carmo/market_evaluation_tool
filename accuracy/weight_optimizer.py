"""Per-ticker signal weight optimizer.

After each accuracy evaluation batch, correlates individual signal votes with
actual outcomes and nudges weights in the direction of higher accuracy. Weights
are stored per-ticker in the `ticker_weights` DB table and loaded at analysis
time to replace the global config defaults.

Algorithm (for each signal):
    accuracy_rate  = correct_non_neutral_predictions / total_non_neutral_predictions
    adjustment     = (accuracy_rate - 0.5) * 2          # maps [0%→-1, 50%→0, 100%→+1]
    new_weight     = old_weight * (1 + LEARNING_RATE * adjustment)
    new_weight     = clamp(new_weight, MIN_WEIGHT, MAX_WEIGHT)

FLAT actual outcomes and neutral signal values (0) are excluded — both carry
no directional information.
"""
from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from config import SignalWeights

logger = logging.getLogger(__name__)

_LEARNING_RATE = 0.25
_MIN_WEIGHT    = 0.05
_MAX_WEIGHT    = 5.0
_MIN_SAMPLES   = 10   # min non-neutral, non-FLAT observations before adjusting a signal

# Blend weight learning rate (Phase 4 — smaller, blends are more stable)
_BLEND_LR      = 0.10
_MIN_BLEND     = 0.1
_MAX_BLEND     = 0.9

# Maps signal name (as produced by analysis/signals.py) → SignalWeights field name
_SIGNAL_TO_KEY: dict[str, str] = {
    "rsi":          "rsi",
    "macd_cross":   "macd",
    "bb_position":  "bb",
    "ma_cross":     "trend",
    "volume_trend": "volume",
    "stoch":        "stoch",
    "sentiment":    "sentiment",
}

_DIR_TO_INT: dict[str, int] = {"UP": 1, "DOWN": -1, "FLAT": 0}


def compute_adjusted_weights(
    current: dict[str, float],
    votes: list[tuple[dict[str, int], str]],
    min_samples: int = _MIN_SAMPLES,
) -> dict[str, float] | None:
    """Compute updated weight dict from signal votes vs actual outcomes.

    Pure function — no DB, no side effects.

    Args:
        current:     Current weight dict keyed by SignalWeights field names.
        votes:       List of (signal_votes_dict, actual_direction) pairs.
                     signal_votes_dict: {signal_name: int_value (-1/0/1)}
                     actual_direction:  "UP" | "DOWN" | "FLAT"
        min_samples: Minimum non-zero, non-FLAT signal observations per signal
                     before that signal's weight is adjusted.

    Returns:
        Updated weight dict, or None if nothing changed.
    """
    # Collect per-signal accuracy stats, excluding FLAT actual and neutral signal
    stats: dict[str, dict[str, int]] = {}
    for sig_votes_dict, actual_dir in votes:
        actual_int = _DIR_TO_INT.get(actual_dir, 0)
        if actual_int == 0:
            continue  # skip FLAT outcomes
        for sig_name, sig_val in sig_votes_dict.items():
            if sig_val == 0:
                continue  # skip neutral signals
            s = stats.setdefault(sig_name, {"correct": 0, "total": 0})
            s["total"] += 1
            if sig_val == actual_int:
                s["correct"] += 1

    updated = dict(current)
    changed = False

    for sig_name, weight_key in _SIGNAL_TO_KEY.items():
        s = stats.get(sig_name)
        if s is None or s["total"] < min_samples:
            continue
        accuracy_rate = s["correct"] / s["total"]
        adjustment = (accuracy_rate - 0.5) * 2   # ∈ [-1, +1]
        old_w = current.get(weight_key, 1.0)
        new_w = old_w * (1.0 + _LEARNING_RATE * adjustment)
        new_w = max(_MIN_WEIGHT, min(_MAX_WEIGHT, new_w))
        new_w = round(new_w, 4)
        updated[weight_key] = new_w
        if abs(new_w - old_w) > 0.0001:
            changed = True
            logger.info(
                "weight_optimizer: %s  %s  %.4f → %.4f  (acc=%.1f%%  n=%d)",
                sig_name, weight_key, old_w, new_w,
                accuracy_rate * 100, s["total"],
            )

    return updated if changed else None


def compute_adjusted_blend_weights(
    xgb_blend: float,
    lstm_blend: float,
    votes: list[tuple[str | None, str | None, str]],
    min_samples: int = _MIN_SAMPLES,
) -> tuple[float, float] | None:
    """Compute updated XGB/LSTM blend weights from per-model prediction accuracy.

    Pure function — no DB, no side effects.  Mirror of compute_adjusted_weights
    for blend weights; used by the walk-forward simulation in training.py.

    Args:
        xgb_blend:   Current XGB blend weight (will be renormalised with lstm_blend).
        lstm_blend:  Current LSTM blend weight.
        votes:       List of (xgb_dir, lstm_dir, actual_dir).
                     xgb_dir / lstm_dir: "UP"/"DOWN"/"FLAT" or None (unavailable).
                     actual_dir:         "UP"/"DOWN"/"FLAT".
        min_samples: Minimum non-FLAT, non-None observations before adjusting.

    Returns:
        (new_xgb_blend, new_lstm_blend) renormalised to sum to 1.0, or None if
        insufficient data or the change is negligible (<= 0.001).
    """
    xgb_stats  = {"correct": 0, "total": 0}
    lstm_stats = {"correct": 0, "total": 0}

    for xgb_d, lstm_d, actual_d in votes:
        if actual_d == "FLAT":
            continue   # non-informative outcome
        if xgb_d is not None and xgb_d != "FLAT":
            xgb_stats["total"] += 1
            if xgb_d == actual_d:
                xgb_stats["correct"] += 1
        if lstm_d is not None and lstm_d != "FLAT":
            lstm_stats["total"] += 1
            if lstm_d == actual_d:
                lstm_stats["correct"] += 1

    if xgb_stats["total"] < min_samples or lstm_stats["total"] < min_samples:
        return None

    xgb_acc  = xgb_stats["correct"]  / xgb_stats["total"]
    lstm_acc = lstm_stats["correct"] / lstm_stats["total"]

    if abs(xgb_acc - lstm_acc) < 0.01:
        return None   # negligible difference — don't nudge

    total_acc = xgb_acc + lstm_acc
    if total_acc == 0:
        return None

    target_xgb  = xgb_acc  / total_acc
    target_lstm = lstm_acc / total_acc

    new_xgb  = xgb_blend  + _BLEND_LR * (target_xgb  - xgb_blend)
    new_lstm = lstm_blend + _BLEND_LR * (target_lstm - lstm_blend)

    new_xgb  = max(_MIN_BLEND, min(_MAX_BLEND, new_xgb))
    new_lstm = max(_MIN_BLEND, min(_MAX_BLEND, new_lstm))

    s = new_xgb + new_lstm
    new_xgb  = round(new_xgb  / s, 4)
    new_lstm = round(new_lstm / s, 4)

    if abs(new_xgb - xgb_blend) <= 0.001:
        return None   # no meaningful change

    logger.info(
        "compute_adjusted_blend_weights: xgb %.3f→%.3f  lstm %.3f→%.3f  "
        "(xgb_acc=%.1f%%  lstm_acc=%.1f%%  n_xgb=%d  n_lstm=%d)",
        xgb_blend, new_xgb, lstm_blend, new_lstm,
        xgb_acc * 100, lstm_acc * 100,
        xgb_stats["total"], lstm_stats["total"],
    )
    return new_xgb, new_lstm


def compute_adjusted_3way_blend_weights(
    xgb_blend: float,
    lstm_blend: float,
    rf_blend: float,
    votes: list[tuple[str | None, str | None, str | None, str]],
    min_samples: int = _MIN_SAMPLES,
) -> tuple[float, float, float] | None:
    """Compute updated XGB/LSTM/RF blend weights from per-model prediction accuracy.

    Pure function — no DB, no side effects.

    Args:
        xgb_blend, lstm_blend, rf_blend: Current blend weights (will be renormalised).
        votes:       List of (xgb_dir, lstm_dir, rf_dir, actual_dir).
                     Each dir: "UP"/"DOWN"/"FLAT" or None (unavailable).
        min_samples: Minimum non-FLAT, non-None observations before adjusting.

    Returns:
        (new_xgb, new_lstm, new_rf) renormalised to sum to 1.0, or None if
        insufficient data or the change is negligible.
    """
    stats = {
        "xgb":  {"correct": 0, "total": 0},
        "lstm": {"correct": 0, "total": 0},
        "rf":   {"correct": 0, "total": 0},
    }
    for xgb_d, lstm_d, rf_d, actual_d in votes:
        if actual_d == "FLAT":
            continue
        for key, pred_d in (("xgb", xgb_d), ("lstm", lstm_d), ("rf", rf_d)):
            if pred_d is not None and pred_d != "FLAT":
                stats[key]["total"] += 1
                if pred_d == actual_d:
                    stats[key]["correct"] += 1

    if any(stats[k]["total"] < min_samples for k in stats):
        return None

    accs = {k: stats[k]["correct"] / stats[k]["total"] for k in stats}
    total_acc = sum(accs.values())
    if total_acc == 0:
        return None

    # Move each blend toward accuracy-proportional target by _BLEND_LR
    targets = {k: accs[k] / total_acc for k in accs}
    blends  = {"xgb": xgb_blend, "lstm": lstm_blend, "rf": rf_blend}
    new_blends = {
        k: max(_MIN_BLEND, min(_MAX_BLEND, blends[k] + _BLEND_LR * (targets[k] - blends[k])))
        for k in blends
    }
    s = sum(new_blends.values())
    new_blends = {k: round(v / s, 4) for k, v in new_blends.items()}

    if max(abs(new_blends[k] - blends[k]) for k in blends) <= 0.001:
        return None

    logger.info(
        "compute_adjusted_3way_blend_weights: "
        "xgb %.3f→%.3f  lstm %.3f→%.3f  rf %.3f→%.3f  "
        "(accs: xgb=%.1f%%  lstm=%.1f%%  rf=%.1f%%)",
        xgb_blend, new_blends["xgb"],
        lstm_blend, new_blends["lstm"],
        rf_blend,   new_blends["rf"],
        accs["xgb"] * 100, accs["lstm"] * 100, accs["rf"] * 100,
    )
    return new_blends["xgb"], new_blends["lstm"], new_blends["rf"]


def optimize_3way_blend_weights(
    ticker: str,
    db,
    xgb_accuracy: float | None,
    lstm_accuracy: float | None,
    rf_accuracy: float | None,
) -> None:
    """Adjust XGB / LSTM / RF blend weights based on each model's recent accuracy.

    Args:
        ticker:        Ticker symbol.
        db:            SQLAlchemy Session.
        xgb_accuracy:  Recent direction accuracy [0, 1] for XGBoost, or None.
        lstm_accuracy: Recent direction accuracy [0, 1] for LSTM, or None.
        rf_accuracy:   Recent direction accuracy [0, 1] for Random Forest, or None.
    """
    available = {k: v for k, v in
                 {"xgb": xgb_accuracy, "lstm": lstm_accuracy, "rf": rf_accuracy}.items()
                 if v is not None}
    if len(available) < 2:
        return

    from storage.repository import TickerWeightsRepository
    repo = TickerWeightsRepository(db)
    row  = repo.get(ticker)

    blends = {
        "xgb":  float(getattr(row, "xgb_blend",  0.50) if row else 0.50),
        "lstm": float(getattr(row, "lstm_blend", 0.25) if row else 0.25),
        "rf":   float(getattr(row, "rf_blend",   0.25) if row else 0.25),
    }

    total_acc = sum(available.values())
    if total_acc == 0:
        return

    changed = False
    new_blends = dict(blends)
    for k in available:
        target = available[k] / total_acc
        new_v  = blends[k] + _BLEND_LR * (target - blends[k])
        new_v  = max(_MIN_BLEND, min(_MAX_BLEND, new_v))
        new_blends[k] = new_v

    # Re-normalise to sum to 1.0
    s = sum(new_blends[k] for k in ("xgb", "lstm", "rf"))
    new_blends = {k: round(new_blends[k] / s, 4) for k in ("xgb", "lstm", "rf")}

    if any(abs(new_blends[k] - blends[k]) > 0.001 for k in new_blends):
        changed = True

    if changed:
        repo.upsert(ticker, {
            "xgb_blend":  new_blends["xgb"],
            "lstm_blend": new_blends["lstm"],
            "rf_blend":   new_blends["rf"],
        })
        logger.info(
            "optimize_3way_blend(%s): xgb %.3f→%.3f  lstm %.3f→%.3f  rf %.3f→%.3f",
            ticker,
            blends["xgb"],  new_blends["xgb"],
            blends["lstm"], new_blends["lstm"],
            blends["rf"],   new_blends["rf"],
        )


def optimize_weights(
    ticker: str,
    db: Session,
    global_weights: SignalWeights,
    min_records: int = 20,
) -> SignalWeights | None:
    """Load accuracy records from DB, compute updated weights, persist, and return.

    Returns updated SignalWeights for `ticker`, or None if insufficient data
    or no meaningful change.
    """
    from storage.orm_models import AccuracyORM, PredictionRecord
    from storage.repository import TickerWeightsRepository

    # Join accuracy records with their parent predictions to get signal snapshots
    rows = (
        db.query(AccuracyORM, PredictionRecord)
        .join(PredictionRecord, AccuracyORM.prediction_id == PredictionRecord.id)
        .filter(AccuracyORM.ticker == ticker)
        .filter(PredictionRecord.signals_json.isnot(None))
        .order_by(AccuracyORM.evaluated_at.desc())
        .limit(500)
        .all()
    )

    if len(rows) < min_records:
        logger.debug(
            "optimize_weights(%s): %d records with signals (need %d), skipping",
            ticker, len(rows), min_records,
        )
        return None

    votes: list[tuple[dict[str, int], str]] = []
    for acc, pred in rows:
        try:
            sig_dict = json.loads(pred.signals_json)
        except (TypeError, json.JSONDecodeError):
            continue
        votes.append((sig_dict, acc.actual_direction))

    # Load current weights (DB record or global defaults)
    repo = TickerWeightsRepository(db)
    row = repo.get(ticker)
    if row:
        current = {
            "rsi": row.rsi, "macd": row.macd, "trend": row.trend,
            "volume": row.volume, "bb": row.bb, "stoch": row.stoch,
            "sentiment": getattr(row, "sentiment", global_weights.sentiment),
        }
    else:
        current = {
            "rsi": global_weights.rsi, "macd": global_weights.macd,
            "trend": global_weights.trend, "volume": global_weights.volume,
            "bb": global_weights.bb, "stoch": global_weights.stoch,
            "sentiment": global_weights.sentiment,
        }

    updated = compute_adjusted_weights(current, votes)
    if updated is None:
        return None

    repo.upsert(ticker, updated)
    logger.info("optimize_weights(%s): weights updated (update #%d)",
                ticker, (row.update_count + 1) if row else 1)
    return SignalWeights(**updated)


def optimize_blend_weights(
    ticker: str,
    db,
    xgb_accuracy: float | None,
    lstm_accuracy: float | None,
) -> None:
    """Adjust XGB / LSTM blend weights based on each model's recent accuracy.

    Uses a proportional update: the model with higher accuracy grows its blend
    share.  Only runs when both accuracy values are available and differ by more
    than 1 percentage point.

    Args:
        ticker:        Ticker symbol.
        db:            SQLAlchemy Session.
        xgb_accuracy:  Recent direction accuracy [0, 1] for XGBoost, or None.
        lstm_accuracy: Recent direction accuracy [0, 1] for LSTM, or None.
    """
    if xgb_accuracy is None or lstm_accuracy is None:
        return
    if abs(xgb_accuracy - lstm_accuracy) < 0.01:
        return

    from storage.repository import TickerWeightsRepository
    repo = TickerWeightsRepository(db)
    row  = repo.get(ticker)

    xgb_blend  = getattr(row, "xgb_blend",  0.6) if row else 0.6
    lstm_blend = getattr(row, "lstm_blend", 0.4) if row else 0.4

    total = xgb_accuracy + lstm_accuracy
    if total == 0:
        return

    # Target blend proportional to accuracy
    target_xgb  = xgb_accuracy  / total
    target_lstm = lstm_accuracy / total

    # Move current blend toward target by _BLEND_LR
    new_xgb  = xgb_blend  + _BLEND_LR * (target_xgb  - xgb_blend)
    new_lstm = lstm_blend + _BLEND_LR * (target_lstm - lstm_blend)

    # Clamp
    new_xgb  = max(_MIN_BLEND, min(_MAX_BLEND, new_xgb))
    new_lstm = max(_MIN_BLEND, min(_MAX_BLEND, new_lstm))

    # Renormalise to sum to 1
    s = new_xgb + new_lstm
    new_xgb  = round(new_xgb  / s, 4)
    new_lstm = round(new_lstm / s, 4)

    if abs(new_xgb - xgb_blend) > 0.001:
        repo.upsert(ticker, {"xgb_blend": new_xgb, "lstm_blend": new_lstm})
        logger.info(
            "optimize_blend(%s): xgb %.3f→%.3f  lstm %.3f→%.3f  "
            "(xgb_acc=%.1f%%  lstm_acc=%.1f%%)",
            ticker, xgb_blend, new_xgb, lstm_blend, new_lstm,
            xgb_accuracy * 100, lstm_accuracy * 100,
        )
