"""Migrate feature transform paths."""

from alembic import op

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- The feature-transform package was consolidated into the public data SDK.\n-- Persisted resource definitions must point at the current import contract.\nUPDATE features\nSET transform = regexp_replace(\n    transform,\n    '^ml_platform\\\\.data_sdk\\\\.features\\\\.[^.]+\\\\.',\n    'ml_platform.data_sdk.'\n)\nWHERE transform ~ '^ml_platform\\\\.data_sdk\\\\.features\\\\.[^.]+\\\\.';\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
