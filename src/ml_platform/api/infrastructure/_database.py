"""Synchronous database access primitives for the materialization API."""

from collections.abc import Generator
from functools import lru_cache

import psycopg
from psycopg import Connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ml_platform.api._configuration import Dependencies
from ml_platform.api.infrastructure.exceptions import (
    MaterializationDatabaseUnavailableError,
)
from ml_platform.common.types import DatabaseUrl


@lru_cache
def engine_for(database_url: DatabaseUrl) -> Engine:
    """Create the synchronous Psycopg-backed engine for one configured URL."""
    url = make_url(database_url)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    return create_engine(url, pool_pre_ping=True)


def get_session(dependencies: Dependencies) -> Generator[Session, None, None]:
    """Yield a synchronous SQLAlchemy session for one dependency context."""
    try:
        with Session(engine_for(dependencies.database_url)) as session:
            yield session
    except OperationalError as error:
        raise MaterializationDatabaseUnavailableError from error


def get_connection(
    dependencies: Dependencies,
) -> Generator[Connection, None, None]:
    """Yield a Psycopg connection for one dependency context."""
    try:
        with psycopg.connect(dependencies.database_url) as connection:
            yield connection
    except psycopg.OperationalError as error:
        raise MaterializationDatabaseUnavailableError from error
