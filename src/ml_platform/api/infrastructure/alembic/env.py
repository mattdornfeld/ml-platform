"""Alembic environment for the ML Platform control-plane schema."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from alembic import context
from sqlalchemy import engine_from_config, make_url, pool

from ml_platform.api.repositories import Base

if TYPE_CHECKING:
    from alembic.config import Config
    from sqlalchemy.schema import MetaData


def _alembic_config() -> Config:
    """Configure Alembic's process-local database URL when one is provided."""
    config = context.config
    database_url = os.environ.get("DATABASE_URL")
    if database_url is None:
        return config
    url = make_url(database_url)
    if url.drivername in {"postgresql", "postgresql+psycopg2"}:
        url = url.set(drivername="postgresql+psycopg")
    config.set_main_option(
        "sqlalchemy.url", url.render_as_string(hide_password=False)
    )
    return config


def _target_metadata() -> MetaData:
    """Return ORM metadata used when Alembic generates schema revisions."""
    return Base.metadata


def run_migrations_offline() -> None:
    """Emit migrations without opening a database connection."""
    config = _alembic_config()
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=_target_metadata(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations through a short-lived SQLAlchemy connection."""
    config = _alembic_config()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=_target_metadata()
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
