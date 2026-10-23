"""Normalize validation arrays."""

from alembic import op

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Validation overrides are an ordered, discriminated list.  Earlier resource\n-- versions stored a legacy object shape that the strict materializer contract\n-- correctly rejects.  Those objects only duplicated built-in validations, so\n-- replace them with the canonical empty override list.\nUPDATE features\nSET validations = '[]'::jsonb\nWHERE jsonb_typeof(validations) = 'object';\n\nALTER TABLE features\nALTER COLUMN validations SET DEFAULT '[]'::jsonb;\n\nALTER TABLE labels\nADD COLUMN validations JSONB NOT NULL DEFAULT '[]'::jsonb;\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
