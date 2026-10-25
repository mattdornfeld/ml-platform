"""Shared PostgreSQL integration-test infrastructure for the API suite."""

import os
from collections.abc import Generator

import psycopg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from testcontainers.community.postgres import PostgresContainer

# Permit module import; each test call path receives its own Dependencies fixture.
os.environ.setdefault("DATABASE_URL", "postgresql://unused")

from ml_platform.api import (
    Dependencies,
    get_database_connection,
    get_database_session,
)
from ml_platform.api.app import create_app
from ml_platform.api.infrastructure import (
    engine_for,
    upgrade_database,
)
from ml_platform.common.types import Environment


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
        dependencies = Dependencies(
            environment=Environment.TEST,
            database_url=database_url,
            kubernetes_namespace="ml-platform",
            materialization_image="ml-platform:latest",
            warehouse_dir="s3a://warehouse",
            s3_endpoint="http://minio:9000",
            minio_root_user="minio",
            minio_root_password="miniosecret",
            headlamp_url="http://127.0.0.1:8002",
            minio_console_url="http://127.0.0.1:9003",
        )
        upgrade_database(dependencies)
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
def dependencies(clean_postgres: str) -> Dependencies:
    """Create one validated test dependency context for a test call path."""
    return Dependencies(
        environment=Environment.TEST,
        database_url=clean_postgres,
        kubernetes_namespace="ml-platform",
        materialization_image="ml-platform:latest",
        warehouse_dir="s3a://warehouse",
        s3_endpoint="http://minio:9000",
        minio_root_user="minio",
        minio_root_password="miniosecret",
        headlamp_url="http://127.0.0.1:8002",
        minio_console_url="http://127.0.0.1:9003",
    )


@pytest.fixture
def clean_postgres(postgres_database: str) -> str:
    """Remove persisted API state before one integration test."""
    with psycopg.connect(postgres_database) as connection:
        connection.execute("TRUNCATE datasets, features, labels, jobs CASCADE")
    return postgres_database


@pytest.fixture
def api_client(
    dependencies: Dependencies,
) -> Generator[TestClient, None, None]:
    """Create an API client backed by the disposable PostgreSQL database."""
    engine = engine_for(dependencies.database_url)

    def session_dependency() -> Generator[Session, None, None]:
        """Yield an isolated SQLAlchemy session for one test HTTP request."""
        with Session(engine) as session:
            yield session

    def connection_dependency() -> Generator[psycopg.Connection, None, None]:
        """Yield an isolated Psycopg connection for one test HTTP request."""
        with psycopg.connect(dependencies.database_url) as connection:
            yield connection

    application = create_app(dependencies)
    application.dependency_overrides[get_database_session] = session_dependency
    application.dependency_overrides[get_database_connection] = (
        connection_dependency
    )
    try:
        yield TestClient(application)
    finally:
        application.dependency_overrides.clear()
        engine.dispose()
        engine_for.cache_clear()


@pytest.fixture
def runtime_dependencies() -> Dependencies:
    """Create validated dependencies for a unit-test execution path."""
    return Dependencies(
        environment=Environment.TEST,
        database_url="postgresql://unused",
        kubernetes_namespace="ml-platform",
        materialization_image="ml-platform:latest",
        warehouse_dir="s3a://warehouse",
        s3_endpoint="http://minio:9000",
        minio_root_user="minio",
        minio_root_password="miniosecret",
        headlamp_url="http://127.0.0.1:8002",
        minio_console_url="http://127.0.0.1:9003",
    )
