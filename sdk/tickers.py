"""sdk.tickers — Ticker management SDK for Market Evaluation Tool.

Usable both as a library and as a CLI script.

Library usage:
    from sdk.tickers import TickerClient

    # Connect via .env / environment variables (default)
    with TickerClient() as tc:
        tc.add("AAPL")
        tc.add("BTC-USD")                 # market_type auto-detected
        tc.add("GC=F", "commodity")       # explicit market_type
        tickers = tc.list()               # [Ticker(...), ...]
        t = tc.get("AAPL")                # Ticker | None
        tc.remove("AAPL")                 # bool
        tc.seed("ticker.cfg")             # int (number added)

    # Connect with an explicit URL
    with TickerClient("mysql+pymysql://user:pass@host/market_eval") as tc:
        tc.add("NVDA")

    # Without context manager
    tc = TickerClient()
    tc.add("MSFT")
    tc.close()

CLI usage:
    python -m sdk.tickers list [--all]
    python -m sdk.tickers add SYMBOL [--type stock|crypto|commodity]
    python -m sdk.tickers remove SYMBOL
    python -m sdk.tickers get SYMBOL
    python -m sdk.tickers seed [--file ticker.cfg]
    python -m sdk.tickers status
    python -m sdk.tickers bulk-add SYMBOL1 SYMBOL2 ...
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Allow running as `python sdk/tickers.py` from project root
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

from storage.orm_models import Base, TickerRecord


# ── Return type ────────────────────────────────────────────────────────────

@dataclass
class Ticker:
    """Immutable view of a ticker record."""
    symbol: str
    market_type: str
    active: bool
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "symbol":      self.symbol,
            "market_type": self.market_type,
            "active":      self.active,
            "created_at":  self.created_at.isoformat(),
        }


# ── Market type detection ──────────────────────────────────────────────────

def _detect_market_type(symbol: str) -> str:
    if symbol.endswith("-USD"):
        return "crypto"
    if symbol.endswith("=F"):
        return "commodity"
    return "stock"


# ── Connection URL helpers ─────────────────────────────────────────────────

def _resolve_url(database_url: str | None) -> str:
    """Return the SQLAlchemy connection URL to use.

    Priority:
      1. Explicit ``database_url`` constructor argument
      2. DATABASE_URL environment variable
      3. DB_TYPE=mysql + MYSQL_* environment variables
      4. DB_PATH environment variable (SQLite file path)
      5. Default: sqlite:///market_eval.db
    """
    if database_url:
        return database_url

    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        return env_url

    if os.getenv("DB_TYPE", "").strip().lower() == "mysql":
        host     = os.getenv("MYSQL_HOST",     "localhost")
        port     = os.getenv("MYSQL_PORT",     "3306")
        user     = os.getenv("MYSQL_USER",     "market_eval")
        password = os.getenv("MYSQL_PASSWORD", "")
        database = os.getenv("MYSQL_DATABASE", "market_eval")
        return (
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            "?charset=utf8mb4"
        )

    db_path = os.getenv("DB_PATH", "market_eval.db").strip()
    return f"sqlite:///{db_path}"


# ── Client ─────────────────────────────────────────────────────────────────

class TickerClient:
    """Manage market tickers in the configured database.

    Args:
        database_url: SQLAlchemy connection URL. If omitted, the URL is
            resolved from environment variables / .env (see module docstring).

    The client can be used as a context manager (recommended) or manually
    closed with :meth:`close`.
    """

    def __init__(self, database_url: str | None = None) -> None:
        load_dotenv()
        url = _resolve_url(database_url)
        is_sqlite = url.startswith("sqlite")

        if is_sqlite:
            from sqlalchemy.pool import StaticPool
            self._engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
            from sqlalchemy import event
            @event.listens_for(self._engine, "connect")
            def _pragmas(conn, _rec):
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
        else:
            self._engine = create_engine(
                url, pool_pre_ping=True, pool_recycle=3600
            )

        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(
            bind=self._engine, autocommit=False, autoflush=False
        )
        display = url.split("@")[-1] if "@" in url else url
        self._display_url = display

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self) -> TickerClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Dispose the connection pool."""
        self._engine.dispose()

    # ── Read operations ───────────────────────────────────────────────

    def list(self, include_inactive: bool = False) -> list[Ticker]:
        """Return all tickers.

        Args:
            include_inactive: When True, also return soft-deleted tickers.

        Returns:
            List of :class:`Ticker` sorted by symbol.
        """
        with self._session() as db:
            q = db.query(TickerRecord)
            if not include_inactive:
                q = q.filter(TickerRecord.active == True)
            rows = q.order_by(TickerRecord.symbol).all()
            return [_to_ticker(r) for r in rows]

    def get(self, symbol: str) -> Ticker | None:
        """Return a single ticker by symbol, or None if not found.

        Matches both active and inactive records.
        """
        with self._session() as db:
            row = _find(db, symbol)
            return _to_ticker(row) if row else None

    def count(self, include_inactive: bool = False) -> int:
        """Return the number of tickers in the database."""
        with self._session() as db:
            q = db.query(func.count(TickerRecord.id))
            if not include_inactive:
                q = q.filter(TickerRecord.active == True)
            return q.scalar() or 0

    # ── Write operations ──────────────────────────────────────────────

    def add(self, symbol: str, market_type: str | None = None) -> Ticker:
        """Add a ticker to the database.

        If the ticker already exists (even if inactive), it is reactivated.
        Market type is auto-detected from the symbol if not provided:
          ``BTC-USD`` → crypto, ``GC=F`` → commodity, everything else → stock.

        Args:
            symbol:      Ticker symbol (e.g. ``AAPL``, ``BTC-USD``, ``GC=F``).
            market_type: One of ``"stock"``, ``"crypto"``, ``"commodity"``.
                         Auto-detected if omitted.

        Returns:
            The :class:`Ticker` as stored in the database.

        Raises:
            ValueError: If market_type is provided but not a valid value.
        """
        symbol = symbol.strip().upper()
        _validate_symbol(symbol)

        if market_type is not None:
            market_type = market_type.strip().lower()
            _validate_market_type(market_type)
        else:
            market_type = _detect_market_type(symbol)

        with self._session() as db:
            existing = _find(db, symbol)
            if existing:
                if not existing.active:
                    existing.active = True
                    db.commit()
                    db.refresh(existing)
                return _to_ticker(existing)

            row = TickerRecord(
                symbol=symbol,
                market_type=market_type,
                active=True,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return _to_ticker(row)

    def remove(self, symbol: str) -> bool:
        """Soft-delete a ticker (sets active=False).

        Returns:
            True if the ticker was found and deactivated, False if not found.
        """
        symbol = symbol.strip().upper()
        with self._session() as db:
            row = _find(db, symbol)
            if row is None:
                return False
            row.active = False
            db.commit()
            return True

    def bulk_add(
        self,
        symbols: list[str],
        market_type: str | None = None,
    ) -> list[Ticker]:
        """Add multiple tickers in a single transaction.

        Args:
            symbols:     List of ticker symbols.
            market_type: Applied to all symbols. Auto-detected per symbol if omitted.

        Returns:
            List of resulting :class:`Ticker` objects (added or already existing).
        """
        return [self.add(s, market_type) for s in symbols]

    def seed(self, cfg_path: str = "ticker.cfg") -> int:
        """Populate the database from a ``ticker.cfg`` file.

        The file format uses ``[stocks]``, ``[crypto]``, and ``[commodities]``
        sections with one symbol per line.  Lines starting with ``#`` and
        blank lines are ignored.  Existing tickers are not duplicated.

        Args:
            cfg_path: Path to the configuration file.

        Returns:
            Number of new tickers added.
        """
        path = Path(cfg_path)
        if not path.exists():
            raise FileNotFoundError(f"Ticker config not found: {path}")

        section_map = {
            "[stocks]":      "stock",
            "[crypto]":      "crypto",
            "[commodities]": "commodity",
        }
        current_type: str | None = None
        added = 0

        with self._session() as db:
            for raw in path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower() in section_map:
                    current_type = section_map[line.lower()]
                    continue
                if current_type is None:
                    continue

                symbol = line.upper()
                try:
                    _validate_symbol(symbol)
                except ValueError:
                    continue

                if not _find(db, symbol):
                    db.add(TickerRecord(
                        symbol=symbol,
                        market_type=current_type,
                        active=True,
                    ))
                    added += 1

            if added:
                db.commit()

        return added

    # ── Iteration ─────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Ticker]:
        """Iterate over all active tickers."""
        return iter(self.list())

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, symbol: str) -> bool:
        return self.get(symbol.strip().upper()) is not None

    # ── Representation ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"TickerClient(url={self._display_url!r}, tickers={self.count()})"

    # ── Internal ──────────────────────────────────────────────────────

    def _session(self) -> Session:
        return self._Session()


# ── Helpers ────────────────────────────────────────────────────────────────

def _find(db: Session, symbol: str) -> TickerRecord | None:
    return db.query(TickerRecord).filter(TickerRecord.symbol == symbol).first()


def _to_ticker(row: TickerRecord) -> Ticker:
    return Ticker(
        symbol=row.symbol,
        market_type=row.market_type,
        active=row.active,
        created_at=row.created_at,
    )


def _validate_symbol(symbol: str) -> None:
    import re
    if not re.match(r"^[A-Z0-9\-=\.]{1,20}$", symbol):
        raise ValueError(
            f"Invalid ticker symbol {symbol!r}. "
            "Use 1-20 chars: A-Z, 0-9, hyphens, equals signs, dots."
        )


def _validate_market_type(market_type: str) -> None:
    valid = {"stock", "crypto", "commodity"}
    if market_type not in valid:
        raise ValueError(
            f"Invalid market_type {market_type!r}. Must be one of: {valid}"
        )


# ── CLI ────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sdk.tickers",
        description="Manage market tickers in the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m sdk.tickers list\n"
            "  python -m sdk.tickers add NVDA\n"
            "  python -m sdk.tickers add BTC-USD --type crypto\n"
            "  python -m sdk.tickers bulk-add AAPL MSFT GOOGL AMZN\n"
            "  python -m sdk.tickers remove AAPL\n"
            "  python -m sdk.tickers get AAPL\n"
            "  python -m sdk.tickers seed --file ticker.cfg\n"
            "  python -m sdk.tickers status\n"
        ),
    )
    p.add_argument(
        "--url", metavar="DATABASE_URL",
        help="SQLAlchemy connection URL (overrides .env)",
    )

    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # list
    ls = sub.add_parser("list", help="List tickers")
    ls.add_argument("--all", dest="include_inactive", action="store_true",
                    help="Include inactive (removed) tickers")
    ls.add_argument("--type", dest="market_type",
                    choices=["stock", "crypto", "commodity"],
                    help="Filter by market type")

    # add
    add = sub.add_parser("add", help="Add a ticker")
    add.add_argument("symbol", help="Ticker symbol (e.g. AAPL, BTC-USD, GC=F)")
    add.add_argument("--type", dest="market_type",
                     choices=["stock", "crypto", "commodity"],
                     help="Market type (auto-detected if omitted)")

    # bulk-add
    bulk = sub.add_parser("bulk-add", help="Add multiple tickers")
    bulk.add_argument("symbols", nargs="+", help="One or more ticker symbols")
    bulk.add_argument("--type", dest="market_type",
                      choices=["stock", "crypto", "commodity"],
                      help="Market type applied to all symbols")

    # remove
    rm = sub.add_parser("remove", help="Remove (deactivate) a ticker")
    rm.add_argument("symbol", help="Ticker symbol")

    # get
    get = sub.add_parser("get", help="Get details for a single ticker")
    get.add_argument("symbol", help="Ticker symbol")

    # seed
    seed = sub.add_parser("seed", help="Seed tickers from a .cfg file")
    seed.add_argument("--file", default="ticker.cfg",
                      help="Path to ticker config file (default: ticker.cfg)")

    # status
    sub.add_parser("status", help="Show database connection info and ticker counts")

    return p


def _cli(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        client = TickerClient(database_url=args.url or None)
    except Exception as exc:
        print(f"ERROR: Cannot connect to database: {exc}", file=sys.stderr)
        return 1

    with client:
        if args.command == "status":
            total = client.count(include_inactive=True)
            active = client.count()
            by_type: dict[str, int] = {}
            for t in client.list():
                by_type[t.market_type] = by_type.get(t.market_type, 0) + 1
            print(f"Database : {client._display_url}")
            print(f"Active   : {active}")
            print(f"Inactive : {total - active}")
            for mtype, n in sorted(by_type.items()):
                print(f"  {mtype:<12}: {n}")

        elif args.command == "list":
            tickers = client.list(include_inactive=args.include_inactive)
            if hasattr(args, "market_type") and args.market_type:
                tickers = [t for t in tickers if t.market_type == args.market_type]
            if not tickers:
                print("No tickers found.")
            else:
                _print_table(tickers)

        elif args.command == "add":
            try:
                t = client.add(args.symbol, args.market_type)
                status = "reactivated" if t.active else "added"
                print(f"✓  {t.symbol:<15} {t.market_type:<12} ({status})")
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

        elif args.command == "bulk-add":
            errors = []
            for sym in args.symbols:
                try:
                    t = client.add(sym, args.market_type)
                    print(f"✓  {t.symbol:<15} {t.market_type}")
                except ValueError as exc:
                    print(f"✗  {sym:<15} {exc}", file=sys.stderr)
                    errors.append(sym)
            if errors:
                return 1

        elif args.command == "remove":
            found = client.remove(args.symbol)
            if found:
                print(f"✓  {args.symbol.upper()} deactivated.")
            else:
                print(f"Not found: {args.symbol.upper()}", file=sys.stderr)
                return 1

        elif args.command == "get":
            t = client.get(args.symbol)
            if t is None:
                print(f"Not found: {args.symbol.upper()}", file=sys.stderr)
                return 1
            _print_table([t])

        elif args.command == "seed":
            try:
                n = client.seed(args.file)
                print(f"✓  Added {n} ticker(s) from {args.file}")
            except FileNotFoundError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

    return 0


def _print_table(tickers: list[Ticker]) -> None:
    header = f"{'Symbol':<15}  {'Type':<12}  {'Active':<7}  Created"
    print(header)
    print("─" * len(header))
    for t in tickers:
        created = t.created_at.strftime("%Y-%m-%d") if t.created_at else "—"
        active_str = "yes" if t.active else "no"
        print(f"{t.symbol:<15}  {t.market_type:<12}  {active_str:<7}  {created}")


# Allow `python sdk/tickers.py` and `python -m sdk.tickers`
if __name__ == "__main__":
    sys.exit(_cli())
