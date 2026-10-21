"""PostgreSQL connection dependency for the materialization API."""

import os
from collections.abc import Generator

import psycopg
from fastapi import HTTPException, status
from psycopg import Connection


def get_connection() -> Generator[Connection, None, None]:
    """Yield one transaction-scoped PostgreSQL connection from DATABASE_URL."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DATABASE_URL must be configured for materialization services.",
        )
    try:
        with psycopg.connect(database_url) as connection:
            yield connection
    except psycopg.OperationalError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Materialization database is unavailable.",
        ) from error
