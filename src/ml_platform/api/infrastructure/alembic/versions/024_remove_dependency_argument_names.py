"""Remove dependency argument names."""

from alembic import op

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        "-- Dataset names are valid Python identifiers and are injected directly into transforms.\nALTER TABLE feature_dataset_dependencies DROP COLUMN argument_name;\nALTER TABLE feature_dataset_dependencies\n    ADD CONSTRAINT feature_dataset_dependency_name_unique\n    UNIQUE (feature_name, feature_version, dataset_name);\n\nALTER TABLE label_dataset_dependencies DROP COLUMN argument_name;\nALTER TABLE label_dataset_dependencies\n    ADD CONSTRAINT label_dataset_dependency_name_unique\n    UNIQUE (label_name, label_version, dataset_name);\n"
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
