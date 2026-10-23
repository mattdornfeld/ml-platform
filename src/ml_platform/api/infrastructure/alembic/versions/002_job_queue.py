"""Job queue."""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Authoritative queue for materialization jobs claimed by API scheduler instances.\nCREATE TABLE job_queue (\n    job_id UUID PRIMARY KEY REFERENCES jobs (id),\n    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n    claimed_at TIMESTAMPTZ,\n    completed_at TIMESTAMPTZ,\n    worker_id TEXT,\n    CHECK (completed_at IS NULL OR claimed_at IS NOT NULL)\n);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
