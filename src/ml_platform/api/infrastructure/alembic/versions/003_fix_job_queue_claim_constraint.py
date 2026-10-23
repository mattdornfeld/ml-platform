"""Alembic metadata wrapper for SQL revision 003."""

from ml_platform.api.infrastructure.alembic._sql import sql_migration

migration = sql_migration(
    revision="003",
    down_revision="002",
    sql_file="003_fix_job_queue_claim_constraint.sql",
)
revision = migration.revision
down_revision = migration.down_revision
branch_labels = migration.branch_labels
depends_on = migration.depends_on
upgrade = migration.upgrade
downgrade = migration.downgrade
