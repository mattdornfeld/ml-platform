"""Alembic metadata wrapper for SQL revision 016."""

from ml_platform.api.infrastructure.alembic._sql import sql_migration

migration = sql_migration(
    revision="016",
    down_revision="015",
    sql_file="016_label_layout.sql",
)
revision = migration.revision
down_revision = migration.down_revision
branch_labels = migration.branch_labels
depends_on = migration.depends_on
upgrade = migration.upgrade
downgrade = migration.downgrade
