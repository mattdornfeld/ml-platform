"""Feature partition callback tokens."""

from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "ALTER TABLE feature_partition_update_requests ADD COLUMN callback_token TEXT;\nUPDATE feature_partition_update_requests SET callback_token = encode(gen_random_bytes(32), 'hex')\n    WHERE callback_token IS NULL;\nALTER TABLE feature_partition_update_requests ALTER COLUMN callback_token SET NOT NULL;\nCREATE UNIQUE INDEX feature_partition_update_requests_callback_token_index\n    ON feature_partition_update_requests (callback_token);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
