"""Synchronous SQLAlchemy session dependency for the materialization API."""

import os
from collections.abc import Generator
from functools import lru_cache

import psycopg
from fastapi import HTTPException, status
from psycopg import Connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session


@lru_cache
def engine_for(database_url: str) -> Engine:
    """Create the synchronous Psycopg-backed engine for one configured URL."""
    url = make_url(database_url)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    return create_engine(url, pool_pre_ping=True)


def _database_url() -> str:
    """Read the required PostgreSQL endpoint from the process environment."""
    database_url = os.environ.get("DATABASE_URL")
    if database_url is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DATABASE_URL must be configured for materialization services.",
        )
    return database_url


def get_session() -> Generator[Session, None, None]:
    """Yield a synchronous SQLAlchemy session for one API request."""
    try:
        with Session(engine_for(_database_url())) as session:
            yield session
    except OperationalError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Materialization database is unavailable.",
        ) from error


def get_connection() -> Generator[Connection, None, None]:
    """Yield a Psycopg connection until every route has migrated to a session."""
    try:
        with psycopg.connect(_database_url()) as connection:
            yield connection
    except psycopg.OperationalError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Materialization database is unavailable.",
        ) from error
