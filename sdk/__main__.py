"""Allows `python -m sdk.tickers` to work as a CLI entry point."""
import sys
from sdk.tickers import _cli

sys.exit(_cli())
