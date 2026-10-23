"""Feature keys."""

from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Feature identity may use multiple ordered entity-key columns.\nALTER TABLE features ADD COLUMN feature_keys JSONB;\nUPDATE features SET feature_keys = jsonb_build_array(feature_key) WHERE feature_keys IS NULL;\nALTER TABLE features ALTER COLUMN feature_keys SET NOT NULL;\nALTER TABLE features ADD CONSTRAINT features_feature_keys_array CHECK (jsonb_typeof(feature_keys) = 'array' AND jsonb_array_length(feature_keys) > 0);\nALTER TABLE features DROP COLUMN feature_key;\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
