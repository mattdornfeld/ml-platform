"""Fix job queue claim constraint."""

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "ALTER TABLE job_queue DROP CONSTRAINT job_queue_check;\nALTER TABLE job_queue ADD CONSTRAINT job_queue_check\n    CHECK (completed_at IS NULL OR claimed_at IS NOT NULL);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
