"""Add dataset dependency lookahead."""

from alembic import op

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Triple-barrier labels need a bounded future window as well as historical data.\nALTER TABLE feature_dataset_dependencies\nADD COLUMN num_lookahead_partitions INTEGER NOT NULL DEFAULT 0\nCHECK (num_lookahead_partitions >= 0);\n\nALTER TABLE label_dataset_dependencies\nADD COLUMN num_lookahead_partitions INTEGER NOT NULL DEFAULT 0\nCHECK (num_lookahead_partitions >= 0);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
