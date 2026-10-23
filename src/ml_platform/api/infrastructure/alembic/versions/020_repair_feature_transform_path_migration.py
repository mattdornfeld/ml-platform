"""Alembic metadata wrapper for SQL revision 020."""

from ml_platform.api.infrastructure.alembic._sql import sql_migration

migration = sql_migration(
    revision="020",
    down_revision="019",
    sql_file="020_repair_feature_transform_path_migration.sql",
)
revision = migration.revision
down_revision = migration.down_revision
branch_labels = migration.branch_labels
depends_on = migration.depends_on
upgrade = migration.upgrade
downgrade = migration.downgrade
