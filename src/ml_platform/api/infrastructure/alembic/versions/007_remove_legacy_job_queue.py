"""Remove legacy job queue."""

from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Kueue holds submitted Kubernetes workloads; PostgreSQL retains only durable job outcomes.\nDROP TABLE IF EXISTS job_queue;\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
