import logging
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from data.models import Action, TickerReport

logger = logging.getLogger(__name__)

_console = Console()

_ACTION_STYLE: dict[Action, str] = {
    "BUY": "bold green",
    "SELL": "bold red",
    "HOLD": "bold yellow",
}
_SIGNAL_STYLE = {1: "green", -1: "red", 0: "yellow"}
_DIRECTION_SYMBOL = {"UP": "▲", "DOWN": "▼", "FLAT": "→"}


def _action_text(action: Action) -> Text:
    return Text(f" {action} ", style=_ACTION_STYLE[action])


def render_ticker_report(report: TickerReport) -> None:
    """Print a detailed analysis block for one ticker."""
    score = report.score
    pred = report.prediction
    trend = score.trend

    # ── Header ───────────────────────────────────────────────────────────────
    header = (
        f"[bold]{report.ticker}[/bold]  "
        f"[dim]{report.market_type.upper()}[/dim]  │  "
        f"Price: [cyan]{pred.current_price:,.4f}[/cyan]"
    )
    _console.print(Panel(header, expand=False, border_style="blue"))

    # ── Signals table ─────────────────────────────────────────────────────────
    sig_table = Table(title="Signals", show_header=True, header_style="bold")
    sig_table.add_column("Indicator", style="dim", width=16)
    sig_table.add_column("Signal", justify="center", width=8)
    sig_table.add_column("Raw Value", justify="right", width=12)
    sig_table.add_column("Weight", justify="right", width=8)

    for sig in score.signals:
        label_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        style = _SIGNAL_STYLE[sig.value]
        sig_table.add_row(
            sig.name,
            Text(label_map[sig.value], style=style),
            f"{sig.raw:.4f}",
            f"{sig.weight:.1f}",
        )
    _console.print(sig_table)

    # ── Trend summary ─────────────────────────────────────────────────────────
    aligned_str = "Yes" if trend.ma_aligned else "No"
    trend_style = {"BULLISH": "green", "BEARISH": "red", "SIDEWAYS": "yellow"}[trend.direction]
    _console.print(
        f"  Trend: [{trend_style}]{trend.direction}[/{trend_style}]  │  "
        f"Strength: {trend.strength:.2f}  │  "
        f"MA Aligned: {aligned_str}  │  "
        f"ADX: {trend.adx:.1f}"
    )

    # ── Composite score ───────────────────────────────────────────────────────
    action_text = _action_text(score.action)
    _console.print(
        f"  Score: {score.score:+.3f}  │  Confidence: {score.confidence:.0%}  │  Action: ",
        action_text,
    )

    # ── Forecast table ────────────────────────────────────────────────────────
    fc_table = Table(title="Forecast", show_header=True, header_style="bold")
    fc_table.add_column("Horizon", justify="center", width=10)
    fc_table.add_column("Low", justify="right", width=12)
    fc_table.add_column("Mid", justify="right", width=12)
    fc_table.add_column("High", justify="right", width=12)
    fc_table.add_column("Direction", justify="center", width=12)
    fc_table.add_column("Confidence", justify="right", width=12)

    for fc in (pred.short_forecast, pred.long_forecast):
        sym = _DIRECTION_SYMBOL[fc.direction]
        dir_style = {"UP": "green", "DOWN": "red", "FLAT": "yellow"}[fc.direction]
        fc_table.add_row(
            f"{fc.horizon_days}d",
            f"{fc.low:,.4f}",
            f"{fc.mid:,.4f}",
            f"{fc.high:,.4f}",
            Text(f"{sym} {fc.direction}", style=dir_style),
            f"{fc.confidence:.0%}",
        )
    _console.print(fc_table)
    _console.print()


def render_summary_table(reports: list[TickerReport]) -> None:
    """Print a ranked summary table across all tickers, sorted by signal strength."""
    sorted_reports = sorted(reports, key=lambda r: abs(r.score.score), reverse=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    table = Table(
        title=f"Market Evaluation Summary — {ts}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="right", width=6)
    table.add_column("Ticker", width=10)
    table.add_column("Market", width=12)
    table.add_column("Action", justify="center", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Confidence", justify="right", width=12)

    # Determine horizon labels from first report
    if reports:
        short_h = reports[0].prediction.short_forecast.horizon_days
        long_h = reports[0].prediction.long_forecast.horizon_days
    else:
        short_h, long_h = 3, 7

    table.add_column(f"{short_h}d Mid", justify="right", width=12)
    table.add_column(f"{long_h}d Mid", justify="right", width=12)

    for rank, report in enumerate(sorted_reports, start=1):
        sc = report.score
        pred = report.prediction
        action_text = _action_text(sc.action)
        table.add_row(
            str(rank),
            report.ticker,
            report.market_type,
            action_text,
            f"{sc.score:+.3f}",
            f"{sc.confidence:.0%}",
            f"{pred.short_forecast.mid:,.4f}",
            f"{pred.long_forecast.mid:,.4f}",
        )

    _console.print(table)


def render_all(reports: list[TickerReport]) -> None:
    """Render each ticker report then the summary table."""
    for report in reports:
        render_ticker_report(report)
    render_summary_table(reports)
