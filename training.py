#!/usr/bin/env python3
"""training.py — Walk-forward backtesting & model training simulation.

Fetches full historical data for all active tickers, then steps through
time in weekly increments — making predictions at each step, evaluating
them against actual future prices, and periodically retraining the
XGBoost model based on accumulated accuracy records.

This compresses months of real-world feedback cycles into a single run,
letting you observe how the model improves (or degrades) over time.

Usage:
    python training.py [options]

Options:
    --days INT              Total history to download (default: 730)
    --step INT              Simulation step in trading bars (default: 5 ≈ 1 week)
    --warmup INT            Min bars before first prediction (default: 252 ≈ 1 yr)
    --retrain-threshold N   Accuracy records before retraining (default: 30)
    --tickers T1,T2,...     Override DB tickers (comma-separated)
    --model-dir PATH        Simulation model directory (default: .models_sim)
    --promote               Copy trained models to production dir on completion
    --output PATH           Write all accuracy records to this CSV file
    --no-bloomberg          Skip Bloomberg even if BLOOMBERG_ENABLED=true
    --save-to-db            Persist results to the app database (same DB used by the API)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis import analyze
from config import load_config
from data import fetch_all
from data.cache import DataCache
from data.models import AccuracyRecord, EnrichedData
from features import build_features
from prediction import get_predictor
from prediction.trainer import XGBoostTrainer
from storage.repository import TickerRepository, _detect_market_type
from storage.session import get_session, init_db

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Simulation data structures ─────────────────────────────────────────────

@dataclass
class _Pending:
    """A prediction waiting for its horizon date before it can be evaluated."""
    generated_date: datetime
    entry_price: float
    horizon_days: int
    pred_low: float
    pred_mid: float
    pred_high: float
    pred_dir: str        # UP | DOWN | FLAT
    pred_conf: float
    composite_score: float
    action: str          # BUY | HOLD | SELL
    score_confidence: float
    market_type: str
    model_used: str
    retrain_cycle: int   # model retrain generation at time of prediction
    signals: dict       # {signal_name: int_value} snapshot for weight optimizer
    xgb_dir: str | None = None    # individual XGB direction for blend adaptation
    lstm_dir: str | None = None   # individual LSTM direction for blend adaptation
    rf_dir:  str | None = None    # individual RF direction for blend adaptation


@dataclass
class _EvalRecord:
    """A completed prediction evaluated against the actual market outcome."""
    ticker: str
    generated_date: datetime
    evaluated_date: datetime
    horizon_days: int
    entry_price: float
    pred_low: float
    pred_mid: float
    pred_high: float
    pred_dir: str
    pred_conf: float
    composite_score: float
    action: str
    score_confidence: float
    market_type: str
    actual_price: float
    actual_dir: str
    correct: bool
    price_err_pct: float
    model_used: str
    retrain_cycle: int
    signals: dict       # carried from _Pending for DB persistence + post-run optimizer
    xgb_dir: str | None = None
    lstm_dir: str | None = None
    rf_dir:  str | None = None


@dataclass
class _RetrainEvent:
    """Record of a single model retraining trigger."""
    ticker: str
    horizon_days: int
    cycle: int
    sim_date: datetime
    n_records: int
    batch_acc_pct: float   # direction accuracy of the triggering batch


# ── Low-level helpers ──────────────────────────────────────────────────────

def _direction(actual: float, entry: float, threshold: float) -> str:
    if entry == 0:
        return "FLAT"
    r = (actual - entry) / entry
    if r > threshold:
        return "UP"
    if r < -threshold:
        return "DOWN"
    return "FLAT"


def _price_at_or_after(df: pd.DataFrame, target: datetime) -> float | None:
    """Return the close price of the first bar on or after *target*."""
    ts = pd.Timestamp(target)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    future = df.index[df.index >= ts]
    if len(future) == 0:
        return None
    return float(df.loc[future[0], "close"])


def _make_enriched_slice(full: EnrichedData, end_exclusive: int) -> EnrichedData:
    """Return an EnrichedData whose df covers bars [0, end_exclusive)."""
    return EnrichedData(
        ticker=full.ticker,
        market_type=full.market_type,
        df=full.df.iloc[:end_exclusive].copy(),
        feature_names=full.feature_names,
    )


_DIRECTION_MAP_SIM: dict[int, str] = {0: "DOWN", 1: "FLAT", 2: "UP"}


def _get_per_model_dirs(
    sliced: EnrichedData,
    horizons: list[int],
    model_dir: Path,
) -> dict[int, tuple[str | None, str | None, str | None]]:
    """Return {horizon: (xgb_dir, lstm_dir, rf_dir)} for the current bar.

    Each direction is "UP"/"DOWN"/"FLAT" or None when the model is unavailable.
    Failures are silently swallowed — blend voting is best-effort.
    """
    from prediction.trainer import XGBoostTrainer as _XGBTrainer
    from prediction.lstm_trainer import LSTMTrainer as _LSTMTrainer
    from prediction.rf_trainer import RandomForestTrainer as _RFTrainer
    import numpy as _np

    result: dict[int, tuple[str | None, str | None, str | None]] = {}
    for h in horizons:
        xgb_dir: str | None = None
        lstm_dir: str | None = None
        rf_dir:  str | None = None

        try:
            xtrainer = _XGBTrainer(model_dir, h)
            xmodel = xtrainer.load(sliced.ticker)
            if xmodel is not None:
                feats = xtrainer.extract_features(sliced)
                if feats is not None:
                    proba = xmodel.predict_proba(feats)[0]
                    xgb_dir = _DIRECTION_MAP_SIM[int(_np.argmax(proba))]
        except Exception:
            pass

        try:
            ltrainer = _LSTMTrainer(model_dir, h)
            proba = ltrainer.predict_proba(sliced)
            if proba is not None:
                lstm_dir = _DIRECTION_MAP_SIM[int(_np.argmax(proba))]
        except Exception:
            pass

        try:
            rtrainer = _RFTrainer(model_dir, h)
            rmodel = rtrainer.load(sliced.ticker)
            if rmodel is not None:
                feats = rtrainer.extract_features(sliced)
                if feats is not None:
                    proba = rmodel.predict_proba(feats)[0]
                    rf_dir = _DIRECTION_MAP_SIM[int(_np.argmax(proba))]
        except Exception:
            pass

        result[h] = (xgb_dir, lstm_dir, rf_dir)
    return result


# ── Per-ticker walk-forward simulation ────────────────────────────────────

def simulate_ticker(
    full: EnrichedData,
    config,
    horizons: list[int],
    model_dir: Path,
    args: argparse.Namespace,
    progress: Progress,
    task_id,
) -> tuple[list[_EvalRecord], list[_RetrainEvent], object]:
    """Walk-forward simulation for a single ticker.

    At each step:
      1. Evaluate pending predictions whose horizon has elapsed.
      2. Trigger retraining if enough new accuracy records have accumulated.
      3. Make a new prediction using the model trained so far.
    """
    df = full.df
    ticker = full.ticker
    dates = df.index
    n = len(dates)

    if n <= args.warmup:
        console.print(
            f"  [yellow]{ticker}[/yellow]: only {n} bars available, "
            f"need > {args.warmup} — skipping"
        )
        return [], [], config.signal_weights, (0.50, 0.25, 0.25)

    # ── Warmup: train initial model on the first `warmup` bars ────────
    warmup_slice = _make_enriched_slice(full, args.warmup)

    progress.update(task_id, description=f"[cyan]{ticker:<10}[/cyan] warmup XGB...")
    for h in horizons:
        trainer = XGBoostTrainer(model_dir, h, config.ml_label_threshold)
        ok = trainer.train_and_save(warmup_slice)
        if not ok:
            console.print(
                f"  [dim]{ticker} h={h}d: warmup train insufficient data, "
                f"statistical fallback will be used initially[/dim]"
            )

    # LSTM warmup — no-op when PyTorch is absent
    from prediction.lstm_trainer import LSTMTrainer as _LSTMTrainer, _torch_available
    if _torch_available():
        progress.update(task_id, description=f"[cyan]{ticker:<10}[/cyan] warmup LSTM...")
        for h in horizons:
            _LSTMTrainer(model_dir, h, label_threshold=config.ml_label_threshold,
                         epochs=args.lstm_epochs).train_and_save(warmup_slice)

    # RF warmup
    from prediction.rf_trainer import RandomForestTrainer as _RFTrainer
    progress.update(task_id, description=f"[cyan]{ticker:<10}[/cyan] warmup RF...")
    for h in horizons:
        ok = _RFTrainer(model_dir, h, config.ml_label_threshold).train_and_save(warmup_slice)
        if not ok:
            console.print(
                f"  [dim]{ticker} h={h}d: RF warmup train insufficient data[/dim]"
            )

    progress.update(task_id, description=f"[cyan]{ticker:<10}[/cyan] simulating...")

    predictor = get_predictor(config.predictor_method, model_dir)

    pending: list[_Pending] = []
    acc_since_retrain: dict[int, list[AccuracyRecord]] = {h: [] for h in horizons}
    retrain_cycle: dict[int, int] = {h: 0 for h in horizons}
    eval_records: list[_EvalRecord] = []
    retrain_events: list[_RetrainEvent] = []
    # Per-ticker signal weight adaptation (starts from global config defaults)
    from config import SignalWeights as _SW
    current_weights: _SW = config.signal_weights
    # Track (signal_votes, actual_direction) per horizon for weight optimizer
    sig_votes: dict[int, list[tuple[dict, str]]] = {h: [] for h in horizons}
    # Track (xgb_dir, lstm_dir, rf_dir, actual_dir) per horizon for blend weight adaptation
    blend_votes: dict[int, list[tuple[str | None, str | None, str | None, str]]] = {h: [] for h in horizons}
    current_xgb_blend: float = 0.50
    current_lstm_blend: float = 0.25
    current_rf_blend: float = 0.25

    sim_steps = list(range(args.warmup, n, args.step))
    progress.update(task_id, total=len(sim_steps), completed=0)

    for step_i, bar in enumerate(sim_steps):
        current_date = dates[bar].to_pydatetime().replace(tzinfo=None)
        sliced = _make_enriched_slice(full, bar + 1)

        # ── 1. Evaluate pending predictions ───────────────────────────
        still_pending: list[_Pending] = []
        for pred in pending:
            target = pred.generated_date + timedelta(days=pred.horizon_days)
            if current_date < target:
                still_pending.append(pred)
                continue

            actual = _price_at_or_after(df, target)
            if actual is None:
                still_pending.append(pred)
                continue

            actual_dir = _direction(actual, pred.entry_price, config.ml_label_threshold)
            err_pct = (
                (actual - pred.pred_mid) / pred.pred_mid * 100
                if pred.pred_mid else 0.0
            )

            eval_records.append(_EvalRecord(
                ticker=ticker,
                generated_date=pred.generated_date,
                evaluated_date=current_date,
                horizon_days=pred.horizon_days,
                entry_price=pred.entry_price,
                pred_low=pred.pred_low,
                pred_mid=pred.pred_mid,
                pred_high=pred.pred_high,
                pred_dir=pred.pred_dir,
                pred_conf=pred.pred_conf,
                composite_score=pred.composite_score,
                action=pred.action,
                score_confidence=pred.score_confidence,
                market_type=pred.market_type,
                actual_price=actual,
                actual_dir=actual_dir,
                correct=(actual_dir == pred.pred_dir),
                price_err_pct=err_pct,
                model_used=pred.model_used,
                retrain_cycle=pred.retrain_cycle,
                signals=pred.signals,
                xgb_dir=pred.xgb_dir,
                lstm_dir=pred.lstm_dir,
                rf_dir=pred.rf_dir,
            ))
            sig_votes[pred.horizon_days].append((pred.signals, actual_dir))
            if pred.xgb_dir is not None or pred.lstm_dir is not None or pred.rf_dir is not None:
                blend_votes[pred.horizon_days].append(
                    (pred.xgb_dir, pred.lstm_dir, pred.rf_dir, actual_dir)
                )

            acc_since_retrain[pred.horizon_days].append(AccuracyRecord(
                id=-1,
                prediction_id=-1,
                ticker=ticker,
                horizon_days=pred.horizon_days,
                evaluated_at=current_date,
                actual_price=actual,
                predicted_direction=pred.pred_dir,
                actual_direction=actual_dir,
                direction_correct=(actual_dir == pred.pred_dir),
                predicted_mid=pred.pred_mid,
                price_error_pct=err_pct,
            ))
        pending = still_pending

        # ── 2. Retrain when threshold met for any horizon ──────────────
        for h in horizons:
            batch = acc_since_retrain[h]
            if len(batch) < args.retrain_threshold:
                continue

            batch_acc = sum(1 for r in batch if r.direction_correct) / len(batch)
            trainer = XGBoostTrainer(model_dir, h, config.ml_label_threshold)
            ok = trainer.retrain_with_feedback(sliced, batch)

            if ok:
                retrain_cycle[h] += 1
                if _torch_available():
                    _LSTMTrainer(model_dir, h, label_threshold=config.ml_label_threshold,
                                 epochs=args.lstm_epochs).train_and_save(sliced)
                # RF retrain alongside XGB
                try:
                    _RFTrainer(model_dir, h,
                               config.ml_label_threshold).retrain_with_feedback(sliced, batch)
                except Exception:
                    pass
                retrain_events.append(_RetrainEvent(
                    ticker=ticker,
                    horizon_days=h,
                    cycle=retrain_cycle[h],
                    sim_date=current_date,
                    n_records=len(batch),
                    batch_acc_pct=round(batch_acc * 100, 1),
                ))
            else:
                console.print(
                    f"  [dim]{ticker} h={h}d: retrain rejected "
                    f"(batch acc {batch_acc*100:.0f}% too low)[/dim]"
                )

            # Adapt signal weights based on signal vote accuracy for this batch
            from accuracy.weight_optimizer import (
                compute_adjusted_weights,
                compute_adjusted_blend_weights,
                compute_adjusted_3way_blend_weights,
            )
            current_dict = {
                "rsi": current_weights.rsi, "macd": current_weights.macd,
                "trend": current_weights.trend, "volume": current_weights.volume,
                "bb": current_weights.bb, "stoch": current_weights.stoch,
                "sentiment": current_weights.sentiment,
            }
            updated = compute_adjusted_weights(current_dict, sig_votes[h])
            if updated is not None:
                current_weights = _SW(**updated)
                logger.debug("%s h=%dd: signal weights updated after retrain cycle %d",
                             ticker, h, retrain_cycle[h])

            # Adapt 3-way blend weights (XGB / LSTM / RF)
            blend_result_3 = compute_adjusted_3way_blend_weights(
                current_xgb_blend, current_lstm_blend, current_rf_blend, blend_votes[h]
            )
            if blend_result_3 is not None:
                current_xgb_blend, current_lstm_blend, current_rf_blend = blend_result_3
                logger.debug(
                    "%s h=%dd: blend weights → xgb=%.3f lstm=%.3f rf=%.3f (retrain cycle %d)",
                    ticker, h, current_xgb_blend, current_lstm_blend, current_rf_blend,
                    retrain_cycle[h],
                )

            # Reset regardless — avoid infinite retry on bad batches
            acc_since_retrain[h] = []
            sig_votes[h] = []
            blend_votes[h] = []

        # ── 3. Predict at current bar ──────────────────────────────────
        try:
            score = analyze(sliced, current_weights)
            prediction = predictor.predict(sliced, score, horizons)
            entry = float(df.iloc[bar]["close"])
            signals_snapshot = {s.name: s.value for s in score.signals}
            per_model = _get_per_model_dirs(sliced, horizons, model_dir)

            for i, h in enumerate(horizons):
                fc = prediction.short_forecast if i == 0 else prediction.long_forecast
                xgb_d, lstm_d, rf_d = per_model.get(h, (None, None, None))
                pending.append(_Pending(
                    generated_date=current_date,
                    entry_price=entry,
                    horizon_days=h,
                    pred_low=fc.low,
                    pred_mid=fc.mid,
                    pred_high=fc.high,
                    pred_dir=fc.direction,
                    pred_conf=fc.confidence,
                    composite_score=score.score,
                    action=score.action,
                    score_confidence=score.confidence,
                    market_type=full.market_type,
                    model_used=prediction.model,
                    retrain_cycle=retrain_cycle[h],
                    signals=signals_snapshot,
                    xgb_dir=xgb_d,
                    lstm_dir=lstm_d,
                    rf_dir=rf_d,
                ))
        except Exception as exc:
            logger.debug("Prediction skipped %s @ %s: %s", ticker, current_date, exc)

        # ── Progress ───────────────────────────────────────────────────
        n_eval = len(eval_records)
        n_correct = sum(1 for r in eval_records if r.correct)
        acc_str = f"{n_correct / n_eval * 100:.0f}%" if n_eval else "—%"
        retrains = sum(retrain_cycle.values())
        progress.update(
            task_id,
            completed=step_i + 1,
            description=(
                f"[cyan]{ticker:<10}[/cyan] "
                f"[dim]{current_date.strftime('%Y-%m-%d')}[/dim]  "
                f"acc {acc_str}  retrains {retrains}"
            ),
        )

    return eval_records, retrain_events, current_weights, (current_xgb_blend, current_lstm_blend, current_rf_blend)


# ── Summary reporting ──────────────────────────────────────────────────────

def _colour(pct: float) -> str:
    if pct >= 60:
        return "green"
    if pct >= 50:
        return "yellow"
    return "red"


def print_summary(
    all_eval: list[_EvalRecord],
    all_retrain: list[_RetrainEvent],
    horizons: list[int],
) -> None:
    console.rule("[bold]Simulation Summary[/bold]")
    tickers = sorted(set(r.ticker for r in all_eval))

    # ── Table 1: per-ticker accuracy ────────────────────────────────────
    t1 = Table(title="Per-Ticker Accuracy", show_lines=True, header_style="bold")
    t1.add_column("Ticker", style="bold")
    for h in horizons:
        t1.add_column(f"{h}d Dir Acc", justify="right")
        t1.add_column(f"{h}d Price Err", justify="right")
        t1.add_column(f"{h}d N", justify="right")
    t1.add_column("Retrains", justify="right")
    t1.add_column("Model", style="dim")

    for ticker in tickers:
        recs = [r for r in all_eval if r.ticker == ticker]
        row: list[str] = [ticker]
        for h in horizons:
            h_recs = [r for r in recs if r.horizon_days == h]
            if h_recs:
                acc = sum(1 for r in h_recs if r.correct) / len(h_recs) * 100
                err = sum(abs(r.price_err_pct) for r in h_recs) / len(h_recs)
                c = _colour(acc)
                row += [f"[{c}]{acc:.1f}%[/{c}]", f"{err:.2f}%", str(len(h_recs))]
            else:
                row += ["—", "—", "0"]
        retrains = sum(e.cycle for e in all_retrain if e.ticker == ticker)
        models = sorted({r.model_used for r in recs})
        row += [str(retrains), "+".join(models) if models else "—"]
        t1.add_row(*row)

    console.print(t1)

    # ── Table 2: accuracy progression over time (thirds of simulation) ──
    all_dates = sorted(set(r.generated_date for r in all_eval))
    n_d = len(all_dates)

    if n_d >= 6:
        t2 = Table(
            title="Accuracy Progression (all tickers combined)",
            show_lines=True,
            header_style="bold",
        )
        t2.add_column("Period")
        t2.add_column("Date Range", style="dim")
        for h in horizons:
            t2.add_column(f"{h}d Dir Acc", justify="right")
            t2.add_column(f"{h}d Err", justify="right")
        t2.add_column("Predictions", justify="right")

        thirds = [
            ("Early",  all_dates[:n_d // 3]),
            ("Mid",    all_dates[n_d // 3: 2 * n_d // 3]),
            ("Late",   all_dates[2 * n_d // 3:]),
        ]
        for label, date_list in thirds:
            if not date_list:
                continue
            d_set = set(date_list)
            p_recs = [r for r in all_eval if r.generated_date in d_set]
            date_range = (
                f"{min(date_list).strftime('%Y-%m-%d')} – "
                f"{max(date_list).strftime('%Y-%m-%d')}"
            )
            row = [label, date_range]
            for h in horizons:
                h_recs = [r for r in p_recs if r.horizon_days == h]
                if h_recs:
                    acc = sum(1 for r in h_recs if r.correct) / len(h_recs) * 100
                    err = sum(abs(r.price_err_pct) for r in h_recs) / len(h_recs)
                    c = _colour(acc)
                    row += [f"[{c}]{acc:.1f}%[/{c}]", f"{err:.2f}%"]
                else:
                    row += ["—", "—"]
            row.append(str(len(p_recs)))
            t2.add_row(*row)

        console.print(t2)

    # ── Table 3: accuracy by model generation (retrain cycle) ───────────
    cycles = sorted(set(r.retrain_cycle for r in all_eval))
    if len(cycles) > 1:
        t3 = Table(
            title="Accuracy by Retrain Cycle (all tickers)",
            show_lines=True,
            header_style="bold",
        )
        t3.add_column("Retrain Cycle", justify="right")
        t3.add_column("Label", style="dim")
        for h in horizons:
            t3.add_column(f"{h}d Dir Acc", justify="right")
            t3.add_column(f"{h}d Err", justify="right")
        t3.add_column("Predictions", justify="right")

        for cycle in cycles:
            c_recs = [r for r in all_eval if r.retrain_cycle == cycle]
            label = "Initial" if cycle == 0 else f"After retrain {cycle}"
            row = [str(cycle), label]
            for h in horizons:
                h_recs = [r for r in c_recs if r.horizon_days == h]
                if h_recs:
                    acc = sum(1 for r in h_recs if r.correct) / len(h_recs) * 100
                    err = sum(abs(r.price_err_pct) for r in h_recs) / len(h_recs)
                    c = _colour(acc)
                    row += [f"[{c}]{acc:.1f}%[/{c}]", f"{err:.2f}%"]
                else:
                    row += ["—", "—"]
            row.append(str(len(c_recs)))
            t3.add_row(*row)

        console.print(t3)

    # ── Table 4: retrain events log ─────────────────────────────────────
    if all_retrain:
        t4 = Table(title="Retrain Events", show_lines=False, header_style="bold")
        t4.add_column("Ticker")
        t4.add_column("Horizon", justify="right")
        t4.add_column("Cycle", justify="right")
        t4.add_column("Triggered On")
        t4.add_column("Records", justify="right")
        t4.add_column("Batch Acc", justify="right")

        for e in sorted(all_retrain, key=lambda x: (x.ticker, x.horizon_days, x.cycle)):
            c = _colour(e.batch_acc_pct)
            t4.add_row(
                e.ticker,
                f"{e.horizon_days}d",
                str(e.cycle),
                e.sim_date.strftime("%Y-%m-%d"),
                str(e.n_records),
                f"[{c}]{e.batch_acc_pct:.1f}%[/{c}]",
            )
        console.print(t4)

    # ── Overall totals ──────────────────────────────────────────────────
    total = len(all_eval)
    correct = sum(1 for r in all_eval if r.correct)
    mean_err = sum(abs(r.price_err_pct) for r in all_eval) / total
    n_ml = sum(1 for r in all_eval if r.model_used == "ml")
    n_retrains = len(all_retrain)
    c = _colour(correct / total * 100)

    console.print(
        f"\n[bold]Overall[/bold]  {total:,} predictions evaluated  "
        f"direction accuracy [{c}]{correct / total * 100:.1f}%[/{c}]  "
        f"mean price error [bold]{mean_err:.2f}%[/bold]  "
        f"ML model used [bold]{n_ml / total * 100:.0f}%[/bold] of the time  "
        f"total retrains [bold]{n_retrains}[/bold]"
    )


# ── CSV export ─────────────────────────────────────────────────────────────

def _write_csv(records: list[_EvalRecord], path: Path) -> None:
    rows = [
        {
            "ticker":           r.ticker,
            "generated_date":   r.generated_date.strftime("%Y-%m-%d"),
            "evaluated_date":   r.evaluated_date.strftime("%Y-%m-%d"),
            "horizon_days":     r.horizon_days,
            "entry_price":      round(r.entry_price, 6),
            "pred_mid":         round(r.pred_mid, 6),
            "pred_direction":   r.pred_dir,
            "actual_price":     round(r.actual_price, 6),
            "actual_direction": r.actual_dir,
            "correct":          r.correct,
            "price_err_pct":    round(r.price_err_pct, 4),
            "model_used":       r.model_used,
            "retrain_cycle":    r.retrain_cycle,
        }
        for r in records
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    console.print(f"[dim]Results written → {path}[/dim]")


# ── Database persistence ───────────────────────────────────────────────────

def _persist_results_to_db(
    all_eval: list[_EvalRecord],
    horizons: list[int],
    config,
    ticker_final_weights: dict | None = None,
    ticker_final_blends: dict[str, tuple[float, float]] | None = None,
) -> int:
    """Write simulation eval records to the app database.

    Groups _EvalRecord entries by (ticker, generated_date). Only inserts a
    PredictionRecord when both short and long horizon records exist for the
    same generation date (required by the schema). Skips pairs that already
    exist in the database so re-runs are idempotent.

    Returns the number of PredictionRecord rows inserted.
    """
    import json as _json
    from storage.orm_models import AccuracyORM as _AccORM, PredictionRecord as _PredORM

    short_h, long_h = horizons[0], horizons[1]

    # Group by (ticker, generated_date) → {horizon: _EvalRecord}
    groups: dict[tuple[str, datetime], dict[int, _EvalRecord]] = {}
    for r in all_eval:
        groups.setdefault((r.ticker, r.generated_date), {})[r.horizon_days] = r

    init_db(config.db_url)
    db = next(get_session())
    inserted = 0
    try:
        for (ticker, gen_date), h_map in sorted(groups.items()):
            if short_h not in h_map or long_h not in h_map:
                continue  # need both horizons to satisfy schema NOT NULL constraints

            short_r = h_map[short_h]
            long_r  = h_map[long_h]

            # Idempotent: skip if this (ticker, generated_at) already in DB
            exists = (
                db.query(_PredORM)
                .filter(
                    _PredORM.ticker == ticker,
                    _PredORM.generated_at == gen_date,
                )
                .first()
            )
            if exists:
                continue

            pred = _PredORM(
                ticker=ticker,
                market_type=short_r.market_type,
                generated_at=gen_date,
                model_used=short_r.model_used,
                current_price=short_r.entry_price,
                short_horizon=short_h,
                short_low=short_r.pred_low,
                short_mid=short_r.pred_mid,
                short_high=short_r.pred_high,
                short_direction=short_r.pred_dir,
                short_confidence=short_r.pred_conf,
                long_horizon=long_h,
                long_low=long_r.pred_low,
                long_mid=long_r.pred_mid,
                long_high=long_r.pred_high,
                long_direction=long_r.pred_dir,
                long_confidence=long_r.pred_conf,
                composite_score=short_r.composite_score,
                action=short_r.action,
                score_confidence=short_r.score_confidence,
                signals_json=_json.dumps(short_r.signals) if short_r.signals else None,
            )
            db.add(pred)
            db.flush()  # populate pred.id before creating AccuracyORM rows

            for r in (short_r, long_r):
                db.add(_AccORM(
                    prediction_id=pred.id,
                    ticker=ticker,
                    horizon_days=r.horizon_days,
                    evaluated_at=r.evaluated_date,
                    actual_price=r.actual_price,
                    predicted_direction=r.pred_dir,
                    actual_direction=r.actual_dir,
                    direction_correct=r.correct,
                    predicted_mid=r.pred_mid,
                    price_error_pct=r.price_err_pct,
                    model_used=r.model_used,
                    xgb_direction=r.xgb_dir,
                    lstm_direction=r.lstm_dir,
                    rf_direction=r.rf_dir,
                ))

            inserted += 1

        db.commit()

        # Persist final learned signal + blend weights for each ticker
        if ticker_final_weights or ticker_final_blends:
            from storage.repository import TickerWeightsRepository
            tw_repo = TickerWeightsRepository(db)
            all_tickers = set(ticker_final_weights or {}) | set(ticker_final_blends or {})
            for t in all_tickers:
                weights_dict: dict[str, float] = {}
                if ticker_final_weights and t in ticker_final_weights:
                    sw = ticker_final_weights[t]
                    weights_dict.update({
                        "rsi": sw.rsi, "macd": sw.macd, "trend": sw.trend,
                        "volume": sw.volume, "bb": sw.bb, "stoch": sw.stoch,
                        "sentiment": sw.sentiment,
                    })
                if ticker_final_blends and t in ticker_final_blends:
                    blends = ticker_final_blends[t]
                    weights_dict["xgb_blend"] = blends[0]
                    weights_dict["lstm_blend"] = blends[1]
                    if len(blends) > 2:
                        weights_dict["rf_blend"] = blends[2]
                if weights_dict:
                    tw_repo.upsert(t, weights_dict)

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return inserted


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    from dotenv import load_dotenv
    load_dotenv()
    _default_days = int(os.getenv("HISTORICAL_DAYS", "730"))

    p = argparse.ArgumentParser(
        description="Walk-forward backtesting and model training simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--days", type=int, default=_default_days,
                   help="Total historical days to fetch (default: HISTORICAL_DAYS from .env)")
    p.add_argument("--step", type=int, default=5,
                   help="Simulation step in trading bars (5 ≈ 1 week)")
    p.add_argument("--warmup", type=int, default=252,
                   help="Bars before first prediction (252 ≈ 1 trading year)")
    p.add_argument("--retrain-threshold", dest="retrain_threshold",
                   type=int, default=30,
                   help="New accuracy records needed to trigger a retrain")
    p.add_argument("--tickers", type=str, default="",
                   help="Comma-separated tickers (default: all DB tickers)")
    p.add_argument("--model-dir", dest="model_dir",
                   type=str, default=".models_sim",
                   help="Directory for simulation model files")
    p.add_argument("--promote", action="store_true",
                   help="Copy trained models to production model dir when done")
    p.add_argument("--output", type=str, default="",
                   help="Write accuracy records to this CSV file")
    p.add_argument("--no-bloomberg", dest="no_bloomberg", action="store_true",
                   help="Skip Bloomberg even if BLOOMBERG_ENABLED=true")
    p.add_argument("--save-to-db", dest="save_to_db", action="store_true",
                   help="Persist simulation results to the app database")
    p.add_argument("--lstm-epochs", dest="lstm_epochs", type=int, default=20,
                   help="Max LSTM epochs for warmup/retrain during simulation (default: 20)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.no_bloomberg:
        os.environ["BLOOMBERG_ENABLED"] = "false"

    config = load_config()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # ── Resolve tickers ────────────────────────────────────────────────
    if args.tickers:
        tickers: dict[str, str] = {
            sym.strip().upper(): _detect_market_type(sym.strip().upper())
            for sym in args.tickers.split(",")
            if sym.strip()
        }
    else:
        init_db(config.db_url)
        db = next(get_session())
        try:
            tickers = TickerRepository(db).get_all_active()
        finally:
            db.close()

    if not tickers:
        console.print(
            "[red]No tickers found. Add tickers via the API or use --tickers.[/red]"
        )
        sys.exit(1)

    horizons = [config.forecast_short, config.forecast_long]
    console.print(
        f"[bold]Walk-forward simulation[/bold]  "
        f"{len(tickers)} ticker(s)  "
        f"{args.days}d history  "
        f"step {args.step} bars  "
        f"warmup {args.warmup} bars  "
        f"retrain threshold {args.retrain_threshold} records  "
        f"horizons {horizons}"
    )
    console.print(f"[dim]Simulation models → {model_dir}[/dim]")

    from prediction.lstm_trainer import _torch_available as _ta, _get_device
    if _ta():
        _device = _get_device()
        if _device.type == "cuda":
            import torch as _torch
            _dev = f"[green]CUDA[/green] ({_torch.cuda.get_device_name(0)})"
        elif _device.type == "xpu":
            _dev = "[green]XPU[/green] (Intel oneAPI)"
        else:
            _dev = "[yellow]CPU[/yellow] (no GPU detected)"
        console.print(f"[dim]LSTM device      → {_dev}[/dim]")
    else:
        console.print("[dim]LSTM device      → [yellow]disabled[/yellow] (PyTorch not installed)[/dim]")
    console.print()

    # ── Per-ticker: fetch → build features → simulate ──────────────────
    cache = DataCache(Path(".cache"), config.cache_ttl_hours)
    all_eval: list[_EvalRecord] = []
    all_retrain: list[_RetrainEvent] = []
    ticker_final_weights: dict[str, object] = {}
    ticker_final_blends: dict[str, tuple[float, float]] = {}
    fetch_ok = 0
    feature_ok = 0

    console.rule("[bold]Running Simulation[/bold]")
    with Progress(
        TextColumn("[bold cyan]{task.description:<72}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        for ticker in sorted(tickers):
            market_type = tickers[ticker]

            # Fetch
            market_data = fetch_all(
                {ticker: market_type}, args.days, cache, config.max_fetch_workers
            )
            if ticker not in market_data:
                logger.warning("No market data for %s — skipping", ticker)
                continue
            fetch_ok += 1

            # Build features
            try:
                enriched = build_features(market_data[ticker])
            except Exception as exc:
                logger.warning("Feature engineering failed for %s: %s", ticker, exc)
                continue
            feature_ok += 1

            # Simulate
            n_steps = max(
                0,
                (len(enriched.df) - args.warmup + args.step - 1) // args.step,
            )
            task = prog.add_task(
                f"[cyan]{ticker:<10}[/cyan] initialising...",
                total=n_steps,
            )
            ev, re, final_w, final_blends = simulate_ticker(
                full=enriched,
                config=config,
                horizons=horizons,
                model_dir=model_dir,
                args=args,
                progress=prog,
                task_id=task,
            )
            all_eval.extend(ev)
            all_retrain.extend(re)
            ticker_final_weights[ticker] = final_w
            ticker_final_blends[ticker] = final_blends

            if args.save_to_db and ev:
                n = _persist_results_to_db(
                    ev, horizons, config,
                    {ticker: final_w},
                    {ticker: final_blends},
                )
                logger.info("DB: inserted %d record(s) for %s", n, ticker)

    console.print(
        f"[green]✓[/green] Processed {feature_ok}/{len(tickers)} ticker(s) "
        f"({fetch_ok} fetched, {feature_ok} features built)\n"
    )

    # ── Results ────────────────────────────────────────────────────────
    if not all_eval:
        console.print(
            "[yellow]No predictions were evaluated.\n"
            "Try increasing --days or decreasing --warmup.[/yellow]"
        )
        return

    print_summary(all_eval, all_retrain, horizons)

    if args.output:
        _write_csv(all_eval, Path(args.output))

    # ── Promote to production ──────────────────────────────────────────
    if args.promote:
        prod = Path(config.model_dir).resolve()
        prod.mkdir(parents=True, exist_ok=True, mode=0o700)
        copied = 0
        for f in model_dir.glob("*.json"):
            shutil.copy2(f, prod / f.name)
            sidecar = f.with_suffix(".json.sha256")
            if sidecar.exists():
                shutil.copy2(sidecar, prod / sidecar.name)
            copied += 1
        console.print(
            f"\n[green]✓[/green] Promoted {copied} model file(s) → [bold]{prod}[/bold]"
        )


if __name__ == "__main__":
    main()
