"""Rename resource keys."""

from alembic import op

revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Features and labels expose their ordered entity keys through one shared name.\nALTER TABLE features RENAME COLUMN feature_keys TO keys;\nALTER TABLE features RENAME CONSTRAINT features_feature_keys_array TO features_keys_array;\n\nALTER TABLE labels RENAME COLUMN feature_keys TO keys;\nALTER TABLE labels RENAME CONSTRAINT labels_feature_keys_array TO labels_keys_array;\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
