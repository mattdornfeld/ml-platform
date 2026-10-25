"""Historic Alembic upgrade coverage for persisted resource definitions."""

from uuid import uuid4

import psycopg
from alembic import command
from alembic.config import Config
from sqlalchemy import make_url

from ml_platform.api import Dependencies
from ml_platform.common.types import Environment


def _migration_config(dependencies: Dependencies) -> Config:
    """Build an Alembic configuration bound to one disposable database."""
    configuration = Config(toml_file="pyproject.toml")
    configuration.attributes["dependencies"] = dependencies
    return configuration


def _dependencies(database_url: str) -> Dependencies:
    """Return the complete dependency context required by Alembic."""
    return Dependencies(
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


def test_upgrade_from_024_embeds_dataset_type_in_existing_config(
    postgres_database: str,
) -> None:
    """The current migration upgrades rows written by the preceding schema."""
    database_name = f"migration_{uuid4().hex}"
    source_url = make_url(postgres_database)
    database_url = source_url.set(database=database_name).render_as_string(
        hide_password=False
    )
    with psycopg.connect(postgres_database, autocommit=True) as connection:
        connection.execute(
            psycopg.sql.SQL("CREATE DATABASE {}").format(
                psycopg.sql.Identifier(database_name)
            )
        )
    dependencies = _dependencies(database_url)
    configuration = _migration_config(dependencies)
    try:
        command.upgrade(configuration, "024")
        with psycopg.connect(database_url) as connection:
            connection.execute(
                """INSERT INTO datasets (name, version, dataset_type, config)
                VALUES ('prices', 1, 'market_data', '{"description": "Prices"}'::jsonb)"""
            )
        command.upgrade(configuration, "head")
        with psycopg.connect(database_url) as connection:
            config = connection.execute(
                "SELECT config FROM datasets WHERE name = 'prices' AND version = 1"
            ).fetchone()
        assert config == (
            {"description": "Prices", "dataset_type": "market_data"},
        )
    finally:
        with psycopg.connect(postgres_database, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    psycopg.sql.Identifier(database_name)
                )
            )
