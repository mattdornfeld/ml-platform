-- Completion receipts now carry committed row counts and full Spark schemas.
ALTER TYPE job_type ADD VALUE IF NOT EXISTS 'label_materialization';
-- Historical row counts cannot be reconstructed from PostgreSQL alone. The
-- NOT VALID checks enforce complete receipts for every new revision without
-- inventing row counts for existing Iceberg snapshots.
ALTER TABLE dataset_partitions ADD COLUMN row_count BIGINT;
ALTER TABLE dataset_partitions ADD COLUMN schema JSONB;
UPDATE dataset_partitions SET schema = '{"timestamp":"timestamp","symbol":"string","market_session":"string","Open":"double","High":"double","Low":"double","Close":"double","Volume":"bigint"}'::jsonb;
ALTER TABLE dataset_partitions ADD CONSTRAINT dataset_partition_receipt_details
    CHECK (row_count >= 0 AND schema IS NOT NULL) NOT VALID;

ALTER TABLE feature_partitions ADD COLUMN row_count BIGINT;
ALTER TABLE feature_partitions ADD COLUMN schema JSONB;
UPDATE feature_partitions AS receipt SET schema = jsonb_build_object(
    'timestamp', 'timestamp', 'symbol', 'string', 'feature_value', feature.dtype
)
FROM feature_partition_update_requests AS request
JOIN features AS feature ON feature.name = request.feature_name AND feature.version = request.feature_version
WHERE receipt.partition_id = request.partition_id;
ALTER TABLE feature_partitions ADD CONSTRAINT feature_partition_receipt_details
    CHECK (row_count >= 0 AND schema IS NOT NULL) NOT VALID;

CREATE TABLE labels (
    name TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    description TEXT NOT NULL,
    transform TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    PRIMARY KEY (name, version)
);

CREATE TABLE label_partition_update_requests (
    partition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs (id),
    label_name TEXT NOT NULL,
    label_version INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    status partition_status NOT NULL DEFAULT 'pending',
    callback_token TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (label_name, label_version) REFERENCES labels (name, version)
);
CREATE UNIQUE INDEX label_partition_update_requests_active_identity_index
    ON label_partition_update_requests (label_name, label_version, timestamp)
    WHERE status = 'pending';

CREATE TABLE label_partition_update_request_dataset_dependencies (
    label_partition_request_id UUID NOT NULL REFERENCES label_partition_update_requests (partition_id),
    dataset_partition_id UUID NOT NULL REFERENCES dataset_partitions (partition_id),
    PRIMARY KEY (label_partition_request_id, dataset_partition_id)
);

CREATE TABLE label_partitions (
    partition_id UUID PRIMARY KEY REFERENCES label_partition_update_requests (partition_id),
    content_hash TEXT NOT NULL,
    snapshot_id BIGINT NOT NULL,
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    schema JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE label_partition_dataset_dependencies (
    label_partition_id UUID NOT NULL REFERENCES label_partitions (partition_id),
    dataset_partition_id UUID NOT NULL REFERENCES dataset_partitions (partition_id),
    PRIMARY KEY (label_partition_id, dataset_partition_id)
);
