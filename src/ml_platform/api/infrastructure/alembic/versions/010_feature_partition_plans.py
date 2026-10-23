"""Alembic metadata wrapper for SQL revision 010."""

from ml_platform.api.infrastructure.alembic._sql import sql_migration

migration = sql_migration(
    revision="010",
    down_revision="009",
    sql_file="010_feature_partition_plans.sql",
)
revision = migration.revision
down_revision = migration.down_revision
branch_labels = migration.branch_labels
depends_on = migration.depends_on
upgrade = migration.upgrade
downgrade = migration.downgrade
