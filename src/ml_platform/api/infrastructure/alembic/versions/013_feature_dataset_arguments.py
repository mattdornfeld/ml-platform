"""Feature dataset arguments."""

from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Each feature dependency names the transform constructor argument it supplies.\nALTER TABLE feature_dataset_dependencies ADD COLUMN argument_name TEXT;\nUPDATE feature_dataset_dependencies SET argument_name = 'market_data';\nALTER TABLE feature_dataset_dependencies ALTER COLUMN argument_name SET NOT NULL;\nALTER TABLE feature_dataset_dependencies\n    ADD CONSTRAINT feature_dataset_argument_name_format\n    CHECK (argument_name ~ '^[a-z][a-z0-9_]*$');\nALTER TABLE feature_dataset_dependencies\n    ADD CONSTRAINT feature_dataset_argument_name_unique\n    UNIQUE (feature_name, feature_version, argument_name);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
