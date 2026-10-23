"""Dataset active partition requests."""

from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- A logical dataset partition has at most one active writer. Completed request rows\n-- remain immutable revision history and may be replaced only through an explicit\n-- refresh API when that is introduced.\nCREATE UNIQUE INDEX partition_update_requests_active_identity_index\n    ON partition_update_requests (dataset_name, dataset_version, timestamp)\n    WHERE status = 'pending';\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
