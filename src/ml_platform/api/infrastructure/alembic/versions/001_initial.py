"""Alembic metadata wrapper for SQL revision 001."""

from ml_platform.api.infrastructure.alembic._sql import sql_migration

migration = sql_migration(
    revision="001",
    down_revision=None,
    sql_file="001_initial.sql",
)
revision = migration.revision
down_revision = migration.down_revision
branch_labels = migration.branch_labels
depends_on = migration.depends_on
upgrade = migration.upgrade
downgrade = migration.downgrade
