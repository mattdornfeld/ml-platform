"""Label dataset dependencies."""

from alembic import op

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Persist label definitions' declared dataset dependencies for API job planning.\nCREATE TABLE label_dataset_dependencies (\n    label_name TEXT NOT NULL,\n    label_version INTEGER NOT NULL,\n    dataset_name TEXT NOT NULL,\n    dataset_version INTEGER NOT NULL,\n    argument_name TEXT NOT NULL,\n    num_lookback_partitions INTEGER NOT NULL CHECK (num_lookback_partitions >= 0),\n    PRIMARY KEY (label_name, label_version, dataset_name, dataset_version),\n    UNIQUE (label_name, label_version, argument_name),\n    FOREIGN KEY (label_name, label_version) REFERENCES labels (name, version),\n    FOREIGN KEY (dataset_name, dataset_version) REFERENCES datasets (name, version)\n);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
