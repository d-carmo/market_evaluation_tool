#!/usr/bin/env python3
"""setup_mysql.py — Initialize the MySQL database for Market Evaluation Tool.

This script:
  1. Connects to MySQL as an admin user (root or equivalent)
  2. Creates the application database (utf8mb4 charset)
  3. Creates the application user and grants the necessary privileges
  4. Creates all required tables via SQLAlchemy
  5. Optionally seeds tickers from ticker.cfg

After running this script, add the following to your .env:

    DB_TYPE=mysql
    MYSQL_HOST=localhost
    MYSQL_PORT=3306
    MYSQL_USER=market_eval
    MYSQL_PASSWORD=<the password you chose>
    MYSQL_DATABASE=market_eval

Usage:
    python setup_mysql.py [options]

Options:
    --admin-host HOST     MySQL admin host        (default: localhost)
    --admin-port PORT     MySQL admin port        (default: 3306)
    --admin-user USER     Admin MySQL user        (default: root)
    --admin-password PWD  Admin password          (prompted if omitted)
    --db-host HOST        Host for app connection (default: localhost)
    --db-name NAME        Database to create      (default: market_eval)
    --db-user USER        App user to create      (default: market_eval)
    --db-password PWD     App user password       (prompted if omitted)
    --skip-user           Skip user creation (grant directly to admin user)
    --seed                Seed tickers from ticker.cfg after setup
    --seed-file FILE      Path to ticker config   (default: ticker.cfg)
    --drop-existing       Drop and recreate the database (DESTROYS ALL DATA)
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_REQUIRED_GRANTS = (
    "SELECT", "INSERT", "UPDATE", "DELETE",
    "CREATE", "ALTER", "INDEX", "DROP", "REFERENCES",
)


def _connect_admin(host: str, port: int, user: str, password: str):
    """Return a raw PyMySQL connection as the admin user."""
    try:
        import pymysql
    except ImportError:
        print("ERROR: pymysql is required.  Install it with:  pip install pymysql",
              file=sys.stderr)
        sys.exit(1)

    try:
        conn = pymysql.connect(
            host=host, port=port, user=user, password=password,
            charset="utf8mb4",
            autocommit=True,
        )
        return conn
    except pymysql.OperationalError as exc:
        print(f"ERROR: Cannot connect to MySQL as {user!r}@{host}:{port}: {exc}",
              file=sys.stderr)
        sys.exit(1)


def _run(cursor, sql: str, args=None, label: str = "") -> None:
    try:
        cursor.execute(sql, args)
        print(f"  ✓  {label or sql[:60]}")
    except Exception as exc:
        print(f"  ✗  {label or sql[:60]}\n     {exc}", file=sys.stderr)
        raise


def create_database(cursor, db_name: str, drop_existing: bool) -> None:
    print(f"\n[1/4] Database: {db_name!r}")
    if drop_existing:
        _run(cursor, f"DROP DATABASE IF EXISTS `{db_name}`",
             label=f"Dropped existing database {db_name!r}")
    _run(cursor,
         f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
         f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
         label=f"CREATE DATABASE {db_name!r}")


def create_user(cursor, db_name: str, db_user: str, db_password: str,
                db_host: str) -> None:
    print(f"\n[2/4] User: {db_user!r}@{db_host!r}")
    # CREATE USER IF NOT EXISTS requires MySQL 5.7.6+
    _run(cursor,
         f"CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s",
         args=(db_user, db_host, db_password),
         label=f"CREATE USER {db_user!r}@{db_host!r}")

    grants = ", ".join(_REQUIRED_GRANTS)
    _run(cursor,
         f"GRANT {grants} ON `{db_name}`.* TO %s@%s",
         args=(db_user, db_host),
         label=f"GRANT privileges on {db_name!r}")

    _run(cursor, "FLUSH PRIVILEGES", label="FLUSH PRIVILEGES")


def create_tables(db_url: str) -> None:
    print("\n[3/4] Creating tables")
    from sqlalchemy import create_engine
    from storage.orm_models import Base

    engine = create_engine(db_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    for table in Base.metadata.sorted_tables:
        print(f"  ✓  table {table.name!r}")
    engine.dispose()


def verify_connection(db_url: str) -> None:
    print("\n[4/4] Verifying app connection")
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1
    engine.dispose()
    print("  ✓  Connection verified")


def seed_tickers(db_url: str, seed_file: str) -> None:
    from sdk.tickers import TickerClient
    print(f"\n[+] Seeding tickers from {seed_file!r}")
    with TickerClient(db_url) as tc:
        n = tc.seed(seed_file)
    print(f"  ✓  Added {n} ticker(s)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Initialize the MySQL database for Market Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--admin-host",     default="localhost", metavar="HOST",
                   help="MySQL admin host")
    p.add_argument("--admin-port",     default=3306, type=int, metavar="PORT",
                   help="MySQL admin port")
    p.add_argument("--admin-user",     default="root", metavar="USER",
                   help="Admin MySQL user")
    p.add_argument("--admin-password", default=None, metavar="PWD",
                   help="Admin password (prompted if omitted)")
    p.add_argument("--db-host",        default="localhost", metavar="HOST",
                   help="Host for app user connection (use % for any host)")
    p.add_argument("--db-name",        default="market_eval", metavar="NAME",
                   help="Database name to create")
    p.add_argument("--db-user",        default="market_eval", metavar="USER",
                   help="Application user to create")
    p.add_argument("--db-password",    default=None, metavar="PWD",
                   help="Application user password (prompted if omitted)")
    p.add_argument("--skip-user",      action="store_true",
                   help="Skip user creation (admin user is used directly)")
    p.add_argument("--drop-existing",  action="store_true",
                   help="Drop and recreate the database — DESTROYS ALL DATA")
    p.add_argument("--seed",           action="store_true",
                   help="Seed tickers from --seed-file after setup")
    p.add_argument("--seed-file",      default="ticker.cfg", metavar="FILE",
                   help="Path to ticker.cfg for --seed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("  Market Evaluation Tool — MySQL Setup")
    print("=" * 60)

    if args.drop_existing:
        confirm = input(
            f"\nWARNING: --drop-existing will delete all data in "
            f"{args.db_name!r}.\nType the database name to confirm: "
        ).strip()
        if confirm != args.db_name:
            print("Aborted.")
            sys.exit(0)

    # Collect passwords
    admin_password = args.admin_password
    if admin_password is None:
        admin_password = getpass.getpass(
            f"Admin password for {args.admin_user!r}@{args.admin_host}: "
        )

    if args.skip_user:
        db_user     = args.admin_user
        db_password = admin_password
        db_host_grant = args.db_host
    else:
        db_user     = args.db_user
        db_password = args.db_password
        if db_password is None:
            db_password = getpass.getpass(
                f"Set password for new app user {db_user!r}: "
            )
        db_host_grant = args.db_host

    # Admin operations
    conn = _connect_admin(
        args.admin_host, args.admin_port, args.admin_user, admin_password
    )
    try:
        with conn.cursor() as cur:
            create_database(cur, args.db_name, args.drop_existing)
            if not args.skip_user:
                create_user(cur, args.db_name, db_user, db_password, db_host_grant)
    finally:
        conn.close()

    # Build app connection URL
    app_url = (
        f"mysql+pymysql://{db_user}:{db_password}"
        f"@{args.admin_host}:{args.admin_port}/{args.db_name}"
        "?charset=utf8mb4"
    )

    create_tables(app_url)
    verify_connection(app_url)

    if args.seed:
        seed_tickers(app_url, args.seed_file)

    # Print .env snippet
    print("\n" + "=" * 60)
    print("  Setup complete!  Add these lines to your .env:")
    print("=" * 60)
    print(f"\n  DB_TYPE=mysql")
    print(f"  MYSQL_HOST={args.admin_host}")
    print(f"  MYSQL_PORT={args.admin_port}")
    print(f"  MYSQL_USER={db_user}")
    print(f"  MYSQL_PASSWORD=<your password>")
    print(f"  MYSQL_DATABASE={args.db_name}")
    print()
    print("  Or use the full URL:")
    safe_url = app_url.replace(db_password, "***") if db_password else app_url
    print(f"  DATABASE_URL={safe_url}")
    print()


if __name__ == "__main__":
    main()
