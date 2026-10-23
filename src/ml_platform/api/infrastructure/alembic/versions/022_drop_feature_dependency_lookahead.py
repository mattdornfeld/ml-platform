"""Drop feature dependency lookahead."""

from alembic import op

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Point-in-time feature inputs may only depend on historical source partitions.\nALTER TABLE feature_dataset_dependencies\nDROP COLUMN num_lookahead_partitions;\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
