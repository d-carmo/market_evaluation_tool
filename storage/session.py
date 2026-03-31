"""SQLAlchemy engine + session factory.

Supported databases:
  SQLite (default) — set DB_PATH in .env
  MySQL            — set DATABASE_URL=mysql+pymysql://user:pass@host:port/db
                     OR set DB_TYPE=mysql with MYSQL_HOST / MYSQL_USER / etc.
"""
from __future__ import annotations

import logging
import os
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from storage.orm_models import Base

logger = logging.getLogger(__name__)

_SessionLocal: sessionmaker | None = None
_engine = None


def _resolve_url(db_path_or_url: str) -> str:
    """Return a SQLAlchemy connection URL from config or environment.

    Resolution order:
      1. DATABASE_URL env var (full override — any supported dialect)
      2. Argument is already a URL (contains "://")
      3. DB_TYPE=mysql → build from MYSQL_* env vars
      4. Treat argument as a SQLite file path
    """
    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        return env_url

    if "://" in db_path_or_url:
        return db_path_or_url

    if os.getenv("DB_TYPE", "").strip().lower() == "mysql":
        host = os.getenv("MYSQL_HOST", "localhost")
        port = os.getenv("MYSQL_PORT", "3306")
        user = os.getenv("MYSQL_USER", "market_eval")
        password = os.getenv("MYSQL_PASSWORD", "")
        database = os.getenv("MYSQL_DATABASE", "market_eval")
        return (
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            "?charset=utf8mb4"
        )

    return f"sqlite:///{db_path_or_url}"


def _migrate(engine, url: str) -> None:
    """Apply additive column migrations that CREATE TABLE cannot handle.

    Only adds columns that are missing — safe to run on every startup.
    """
    from sqlalchemy import text as _text
    is_sqlite = url.startswith("sqlite")

    with engine.connect() as conn:
        if is_sqlite:
            def _sqlite_cols(table: str) -> set[str]:
                return {
                    row[1]
                    for row in conn.execute(_text(f"PRAGMA table_info({table})"))
                }

            # predictions table
            pred_cols = _sqlite_cols("predictions")
            if "signals_json" not in pred_cols:
                conn.execute(_text("ALTER TABLE predictions ADD COLUMN signals_json TEXT"))
                conn.commit()
                logger.info("Migration: added predictions.signals_json")

            # ticker_weights table
            wt_cols = _sqlite_cols("ticker_weights")
            for col, typedef in [
                ("sentiment",  "REAL NOT NULL DEFAULT 0.5"),
                ("xgb_blend",  "REAL NOT NULL DEFAULT 0.50"),
                ("lstm_blend", "REAL NOT NULL DEFAULT 0.25"),
                ("rf_blend",   "REAL NOT NULL DEFAULT 0.25"),
            ]:
                if col not in wt_cols:
                    conn.execute(_text(f"ALTER TABLE ticker_weights ADD COLUMN {col} {typedef}"))
                    conn.commit()
                    logger.info("Migration: added ticker_weights.%s", col)

            # accuracy_records table
            acc_cols = _sqlite_cols("accuracy_records")
            for col, typedef in [
                ("model_used",      "TEXT"),
                ("xgb_direction",   "TEXT"),
                ("lstm_direction",  "TEXT"),
                ("rf_direction",    "TEXT"),
            ]:
                if col not in acc_cols:
                    conn.execute(_text(f"ALTER TABLE accuracy_records ADD COLUMN {col} {typedef}"))
                    conn.commit()
                    logger.info("Migration: added accuracy_records.%s", col)

        else:
            # MySQL / PostgreSQL — use INFORMATION_SCHEMA
            def _mysql_has(table: str, col: str) -> bool:
                r = conn.execute(_text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    f"WHERE TABLE_NAME = '{table}' AND COLUMN_NAME = '{col}'"
                ))
                return r.fetchone() is not None

            if not _mysql_has("predictions", "signals_json"):
                conn.execute(_text("ALTER TABLE predictions ADD COLUMN signals_json TEXT"))
                conn.commit()
                logger.info("Migration: added predictions.signals_json")

            for col, typedef in [
                ("sentiment",  "FLOAT NOT NULL DEFAULT 0.5"),
                ("xgb_blend",  "FLOAT NOT NULL DEFAULT 0.50"),
                ("lstm_blend", "FLOAT NOT NULL DEFAULT 0.25"),
                ("rf_blend",   "FLOAT NOT NULL DEFAULT 0.25"),
            ]:
                if not _mysql_has("ticker_weights", col):
                    conn.execute(_text(
                        f"ALTER TABLE ticker_weights ADD COLUMN {col} {typedef}"
                    ))
                    conn.commit()
                    logger.info("Migration: added ticker_weights.%s", col)

            for col, typedef in [
                ("model_used",     "TEXT"),
                ("xgb_direction",  "TEXT"),
                ("lstm_direction", "TEXT"),
                ("rf_direction",   "TEXT"),
            ]:
                if not _mysql_has("accuracy_records", col):
                    conn.execute(_text(
                        f"ALTER TABLE accuracy_records ADD COLUMN {col} {typedef}"
                    ))
                    conn.commit()
                    logger.info("Migration: added accuracy_records.%s", col)


def init_db(db_path: str = "market_eval.db") -> None:
    """Create tables and configure the session factory.

    Accepts either a SQLite file path (legacy) or a full SQLAlchemy URL.
    Safe to call multiple times — subsequent calls reinitialise the engine.
    """
    global _SessionLocal, _engine

    url = _resolve_url(db_path)
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _rec):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    else:
        # MySQL / PostgreSQL / other — use connection pooling with health checks
        engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    Base.metadata.create_all(engine)
    _migrate(engine, url)
    _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    _engine = engine

    display_url = url.split("@")[-1] if "@" in url else url
    logger.info("Database initialised: %s", display_url)


def get_session() -> Generator[Session, None, None]:
    """FastAPI / APScheduler dependency: yields a DB session and closes on exit."""
    if _SessionLocal is None:
        raise RuntimeError("init_db() must be called before get_session()")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
