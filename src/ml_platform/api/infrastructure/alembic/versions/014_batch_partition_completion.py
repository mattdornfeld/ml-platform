"""Batch partition completion."""

from alembic import op

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this PostgreSQL schema revision."""
    op.execute(
        '-- Completion receipts now carry committed row counts and full Spark schemas.\nALTER TYPE job_type ADD VALUE IF NOT EXISTS \'label_materialization\';\n-- Historical row counts cannot be reconstructed from PostgreSQL alone. The\n-- NOT VALID checks enforce complete receipts for every new revision without\n-- inventing row counts for existing Iceberg snapshots.\nALTER TABLE dataset_partitions ADD COLUMN row_count BIGINT;\nALTER TABLE dataset_partitions ADD COLUMN schema JSONB;\nUPDATE dataset_partitions SET schema = \'{"timestamp":"timestamp","symbol":"string","market_session":"string","Open":"double","High":"double","Low":"double","Close":"double","Volume":"bigint"}\'::jsonb;\nALTER TABLE dataset_partitions ADD CONSTRAINT dataset_partition_receipt_details\n    CHECK (row_count >= 0 AND schema IS NOT NULL) NOT VALID;\n\nALTER TABLE feature_partitions ADD COLUMN row_count BIGINT;\nALTER TABLE feature_partitions ADD COLUMN schema JSONB;\nUPDATE feature_partitions AS receipt SET schema = jsonb_build_object(\n    \'timestamp\', \'timestamp\', \'symbol\', \'string\', \'feature_value\', feature.dtype\n)\nFROM feature_partition_update_requests AS request\nJOIN features AS feature ON feature.name = request.feature_name AND feature.version = request.feature_version\nWHERE receipt.partition_id = request.partition_id;\nALTER TABLE feature_partitions ADD CONSTRAINT feature_partition_receipt_details\n    CHECK (row_count >= 0 AND schema IS NOT NULL) NOT VALID;\n\nCREATE TABLE labels (\n    name TEXT NOT NULL,\n    version INTEGER NOT NULL CHECK (version > 0),\n    description TEXT NOT NULL,\n    transform TEXT NOT NULL,\n    parameters JSONB NOT NULL DEFAULT \'{}\'::jsonb,\n    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n    deleted_at TIMESTAMPTZ,\n    deprecated_at TIMESTAMPTZ,\n    PRIMARY KEY (name, version)\n);\n\nCREATE TABLE label_partition_update_requests (\n    partition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n    job_id UUID NOT NULL REFERENCES jobs (id),\n    label_name TEXT NOT NULL,\n    label_version INTEGER NOT NULL,\n    timestamp TIMESTAMPTZ NOT NULL,\n    status partition_status NOT NULL DEFAULT \'pending\',\n    callback_token TEXT NOT NULL UNIQUE,\n    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n    FOREIGN KEY (label_name, label_version) REFERENCES labels (name, version)\n);\nCREATE UNIQUE INDEX label_partition_update_requests_active_identity_index\n    ON label_partition_update_requests (label_name, label_version, timestamp)\n    WHERE status = \'pending\';\n\nCREATE TABLE label_partition_update_request_dataset_dependencies (\n    label_partition_request_id UUID NOT NULL REFERENCES label_partition_update_requests (partition_id),\n    dataset_partition_id UUID NOT NULL REFERENCES dataset_partitions (partition_id),\n    PRIMARY KEY (label_partition_request_id, dataset_partition_id)\n);\n\nCREATE TABLE label_partitions (\n    partition_id UUID PRIMARY KEY REFERENCES label_partition_update_requests (partition_id),\n    content_hash TEXT NOT NULL,\n    snapshot_id BIGINT NOT NULL,\n    row_count BIGINT NOT NULL CHECK (row_count >= 0),\n    schema JSONB NOT NULL,\n    created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n);\nCREATE TABLE label_partition_dataset_dependencies (\n    label_partition_id UUID NOT NULL REFERENCES label_partitions (partition_id),\n    dataset_partition_id UUID NOT NULL REFERENCES dataset_partitions (partition_id),\n    PRIMARY KEY (label_partition_id, dataset_partition_id)\n);\n'
    )


def downgrade() -> None:
    """Signal that this production migration is forward-only."""
    raise NotImplementedError("Schema migrations are forward-only.")
