"""Repair feature transform path migration."""

from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Migration 019 used doubled regular-expression escape characters, which match\n-- literal backslashes under PostgreSQL's standard-conforming SQL strings.\nUPDATE features\nSET transform = regexp_replace(\n    transform,\n    '^ml_platform\\.data_sdk\\.features\\.[^.]+\\.',\n    'ml_platform.data_sdk.'\n)\nWHERE transform ~ '^ml_platform\\.data_sdk\\.features\\.[^.]+\\.';\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
