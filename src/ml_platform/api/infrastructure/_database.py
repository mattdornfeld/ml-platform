"""Synchronous SQLAlchemy session dependency for the materialization API."""

from collections.abc import Generator
from functools import lru_cache

import psycopg
from psycopg import Connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ml_platform.api._configuration import Dependencies
from ml_platform.api._exceptions import DatabaseUrlNotConfiguredError
from ml_platform.api.infrastructure.exceptions import (
    MaterializationDatabaseUnavailableError,
)


@lru_cache
def engine_for(database_url: str) -> Engine:
    """Create the synchronous Psycopg-backed engine for one configured URL."""
    url = make_url(database_url)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    return create_engine(url, pool_pre_ping=True)


def database_url() -> str:
    """Return the configured PostgreSQL endpoint from typed API dependencies."""
    configured_database_url = Dependencies.from_environment().database_url
    if not configured_database_url:
        raise DatabaseUrlNotConfiguredError
    return configured_database_url


def get_session() -> Generator[Session, None, None]:
    """Yield a synchronous SQLAlchemy session for one API request."""
    try:
        with Session(engine_for(database_url())) as session:
            yield session
    except OperationalError as error:
        raise MaterializationDatabaseUnavailableError from error


def get_connection() -> Generator[Connection, None, None]:
    """Yield a Psycopg connection until every route has migrated to a session."""
    try:
        with psycopg.connect(database_url()) as connection:
            yield connection
    except psycopg.OperationalError as error:
        raise MaterializationDatabaseUnavailableError from error
