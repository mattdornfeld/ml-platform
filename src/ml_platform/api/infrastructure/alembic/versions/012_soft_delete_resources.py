"""Soft delete resources."""

from alembic import op

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Resource definitions are immutable; retirement only changes their control-plane visibility.\nALTER TABLE datasets ADD COLUMN deleted_at TIMESTAMPTZ;\nALTER TABLE datasets ADD COLUMN deprecated_at TIMESTAMPTZ;\nALTER TABLE features ADD COLUMN deleted_at TIMESTAMPTZ;\nALTER TABLE features ADD COLUMN deprecated_at TIMESTAMPTZ;\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
