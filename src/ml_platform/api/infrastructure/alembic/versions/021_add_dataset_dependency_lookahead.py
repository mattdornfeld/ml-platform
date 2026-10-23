"""Alembic metadata wrapper for SQL revision 021."""

from ml_platform.api.infrastructure.alembic._sql import sql_migration

migration = sql_migration(
    revision="021",
    down_revision="020",
    sql_file="021_add_dataset_dependency_lookahead.sql",
)
revision = migration.revision
down_revision = migration.down_revision
branch_labels = migration.branch_labels
depends_on = migration.depends_on
upgrade = migration.upgrade
downgrade = migration.downgrade
