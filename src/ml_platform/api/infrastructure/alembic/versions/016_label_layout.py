"""Label layout."""

from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Labels use the same canonical identity and partition metadata as features.\nALTER TABLE labels ADD COLUMN dtype TEXT NOT NULL DEFAULT 'int' CHECK (dtype IN ('double', 'float', 'int', 'bigint'));\nALTER TABLE labels ADD COLUMN frequency TEXT NOT NULL DEFAULT 'daily' CHECK (frequency IN ('hourly', 'daily'));\nALTER TABLE labels ADD COLUMN feature_keys JSONB NOT NULL DEFAULT '[\"symbol\"]'::jsonb;\nALTER TABLE labels ADD CONSTRAINT labels_feature_keys_array CHECK (jsonb_typeof(feature_keys) = 'array' AND jsonb_array_length(feature_keys) > 0);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
