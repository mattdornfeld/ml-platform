"""Alembic metadata wrapper for SQL revision 011."""

from ml_platform.api.infrastructure.alembic._sql import sql_migration

migration = sql_migration(
    revision="011",
    down_revision="010",
    sql_file="011_dataset_active_partition_requests.sql",
)
revision = migration.revision
down_revision = migration.down_revision
branch_labels = migration.branch_labels
depends_on = migration.depends_on
upgrade = migration.upgrade
downgrade = migration.downgrade
