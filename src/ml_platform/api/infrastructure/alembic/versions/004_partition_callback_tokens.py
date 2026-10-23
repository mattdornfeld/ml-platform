"""Partition callback tokens."""

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "ALTER TABLE partition_update_requests ADD COLUMN callback_token TEXT;\nUPDATE partition_update_requests SET callback_token = encode(gen_random_bytes(32), 'hex')\n    WHERE callback_token IS NULL;\nALTER TABLE partition_update_requests ALTER COLUMN callback_token SET NOT NULL;\nCREATE UNIQUE INDEX partition_update_requests_callback_token_index\n    ON partition_update_requests (callback_token);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
