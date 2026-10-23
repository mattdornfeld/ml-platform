"""Normalize job status enum."""

from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "ALTER TABLE jobs ALTER COLUMN status DROP DEFAULT;\nCREATE TYPE job_status_v2 AS ENUM ('submitted', 'succeeded', 'failed', 'cancelled');\nALTER TABLE jobs ALTER COLUMN status TYPE job_status_v2 USING status::text::job_status_v2;\nDROP TYPE job_status;\nALTER TYPE job_status_v2 RENAME TO job_status;\nALTER TABLE jobs ALTER COLUMN status SET DEFAULT 'submitted';\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
