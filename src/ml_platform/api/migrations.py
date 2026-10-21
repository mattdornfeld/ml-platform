"""Apply ordered, immutable PostgreSQL schema migrations."""

import os
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).with_name("migrations")


def migrate_database() -> None:
    """Apply every unapplied SQL migration using the configured PostgreSQL URL."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be configured before API startup.")
    migrations = sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )"""
            )
            cursor.execute("SELECT version FROM schema_migrations")
            applied = {row[0] for row in cursor.fetchall()}
            for migration in migrations:
                if migration.name in applied:
                    continue
                cursor.execute(migration.read_text())
                cursor.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (migration.name,),
                )
                connection.commit()
