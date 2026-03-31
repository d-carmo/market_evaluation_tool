"""Market Evaluation Tool — Ticker SDK.

Programmatic interface for managing tickers in the database.

Quick start:
    from sdk.tickers import TickerClient

    with TickerClient() as tc:
        tc.add("NVDA")
        tc.add("BTC-USD")          # market_type auto-detected
        tc.add("GC=F", "commodity")
        for t in tc.list():
            print(t.symbol, t.market_type)
        tc.remove("NVDA")

CLI:
    python -m sdk.tickers list
    python -m sdk.tickers add NVDA
    python -m sdk.tickers add BTC-USD --type crypto
    python -m sdk.tickers remove AAPL
    python -m sdk.tickers get AAPL
    python -m sdk.tickers seed --file ticker.cfg
    python -m sdk.tickers status
"""
from sdk.tickers import Ticker, TickerClient

__all__ = ["Ticker", "TickerClient"]
