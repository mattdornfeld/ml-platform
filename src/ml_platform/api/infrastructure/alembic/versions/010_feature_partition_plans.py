"""Feature partition plans."""

from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Immutable upstream revision plan for each feature partition request.\nCREATE TABLE feature_partition_update_request_dataset_dependencies (\n    feature_partition_request_id UUID NOT NULL REFERENCES feature_partition_update_requests (partition_id),\n    dataset_partition_id UUID NOT NULL REFERENCES dataset_partitions (partition_id),\n    PRIMARY KEY (feature_partition_request_id, dataset_partition_id)\n);\n\n-- A logical output partition has at most one active writer. Terminal request rows\n-- remain as immutable revision history.\nCREATE UNIQUE INDEX feature_partition_update_requests_active_identity_index\n    ON feature_partition_update_requests (feature_name, feature_version, timestamp)\n    WHERE status = 'pending';\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
