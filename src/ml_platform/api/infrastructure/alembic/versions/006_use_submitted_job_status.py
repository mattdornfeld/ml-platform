"""Use submitted job status."""

from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "ALTER TABLE jobs ALTER COLUMN status SET DEFAULT 'submitted';\n-- Older deployments may still have these legacy values. Cast to text so a\n-- fresh schema whose enum never contained them remains migratable.\nUPDATE jobs SET status = 'submitted' WHERE status::text IN ('queued', 'running');\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
