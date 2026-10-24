"""Shared PostgreSQL integration-test infrastructure for the API suite."""

import os
from collections.abc import Generator

import psycopg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from testcontainers.community.postgres import PostgresContainer

from ml_platform.api import app
from ml_platform.api.infrastructure import (
    engine_for,
    get_connection,
    get_session,
    upgrade_database,
)


@pytest.fixture(scope="session")
def postgres_database() -> Generator[str, None, None]:
    """Start one disposable PostgreSQL database for API integration tests."""
    previous_database_url = os.environ.get("DATABASE_URL")
    previous_test_database_url = os.environ.get("ML_PLATFORM_TEST_DATABASE_URL")
    previous_ryuk_setting = os.environ.get("TESTCONTAINERS_RYUK_DISABLED")
    os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

    with PostgresContainer(
        "postgres:17-alpine",
        username="integration",
        password="integration",
        dbname="integration",
        driver=None,
    ) as database:
        database_url = database.get_connection_url()
        os.environ["DATABASE_URL"] = database_url
        os.environ["ML_PLATFORM_TEST_DATABASE_URL"] = database_url
        upgrade_database()
        yield database_url

    if previous_database_url is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = previous_database_url
    if previous_test_database_url is None:
        os.environ.pop("ML_PLATFORM_TEST_DATABASE_URL", None)
    else:
        os.environ["ML_PLATFORM_TEST_DATABASE_URL"] = previous_test_database_url
    if previous_ryuk_setting is None:
        os.environ.pop("TESTCONTAINERS_RYUK_DISABLED", None)
    else:
        os.environ["TESTCONTAINERS_RYUK_DISABLED"] = previous_ryuk_setting


@pytest.fixture
def clean_postgres(postgres_database: str) -> str:
    """Remove persisted API state before one integration test."""
    with psycopg.connect(postgres_database) as connection:
        connection.execute("TRUNCATE datasets, features, labels, jobs CASCADE")
    return postgres_database


@pytest.fixture
def api_client(clean_postgres: str) -> Generator[TestClient, None, None]:
    """Create an API client backed by the disposable PostgreSQL database."""
    engine = engine_for(clean_postgres)

    def session_dependency() -> Generator[Session, None, None]:
        """Yield an isolated SQLAlchemy session for one test HTTP request."""
        with Session(engine) as session:
            yield session

    def connection_dependency() -> Generator[psycopg.Connection, None, None]:
        """Yield an isolated Psycopg connection for one test HTTP request."""
        with psycopg.connect(clean_postgres) as connection:
            yield connection

    app.dependency_overrides[get_session] = session_dependency
    app.dependency_overrides[get_connection] = connection_dependency
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
        engine_for.cache_clear()
