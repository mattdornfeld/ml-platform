-- PostgreSQL control-plane schema for dataset and feature materialization.
-- Dataset configs are opaque API payloads. Spark/Iceberg owns table schemas.
-- Partition rows are immutable revisions: corrections insert a new row rather
-- than updating content_hash or snapshot_id on an existing row.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$ BEGIN
    CREATE TYPE job_type AS ENUM ('dataset_materialization', 'feature_materialization');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE job_status AS ENUM ('submitted', 'succeeded', 'failed', 'cancelled');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE partition_status AS ENUM ('pending', 'created', 'failed', 'cancelled');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE datasets (
    name TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    dataset_type TEXT NOT NULL CHECK (dataset_type IN ('market_data')),
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (name, version)
);

CREATE TABLE features (
    name TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    dtype TEXT NOT NULL CHECK (dtype IN ('double', 'float', 'int', 'bigint')),
    frequency TEXT NOT NULL CHECK (frequency IN ('hourly', 'daily')),
    feature_key TEXT NOT NULL CHECK (feature_key IN ('symbol')),
    validations JSONB NOT NULL DEFAULT '{}'::JSONB,
    transform TEXT NOT NULL,
    description TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (name, version)
);

-- A feature may depend only on registered dataset versions.
CREATE TABLE feature_dataset_dependencies (
    feature_name TEXT NOT NULL,
    feature_version INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_version INTEGER NOT NULL,
    num_lookback_partitions INTEGER NOT NULL CHECK (num_lookback_partitions >= 0),
    PRIMARY KEY (
        feature_name,
        feature_version,
        dataset_name,
        dataset_version
    ),
    FOREIGN KEY (feature_name, feature_version)
        REFERENCES features (name, version),
    FOREIGN KEY (dataset_name, dataset_version)
        REFERENCES datasets (name, version)
);

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type job_type NOT NULL,
    status job_status NOT NULL DEFAULT 'submitted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A request identifies one dataset partition to be materialized. It owns the
-- dataset identity, logical partition timestamp, dispatching job, and status.
CREATE TABLE partition_update_requests (
    partition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs (id),
    dataset_name TEXT NOT NULL,
    dataset_version INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    status partition_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (dataset_name, dataset_version)
        REFERENCES datasets (name, version)
);

-- A request identifies one feature partition to be materialized. It owns the
-- feature identity, logical partition timestamp, dispatching job, and status.
CREATE TABLE feature_partition_update_requests (
    partition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs (id),
    feature_name TEXT NOT NULL,
    feature_version INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    status partition_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (feature_name, feature_version)
        REFERENCES features (name, version)
);

-- Written only after the Spark job has successfully committed the Iceberg
-- transaction. The logical timestamp is partition_update_requests.timestamp.
CREATE TABLE dataset_partitions (
    partition_id UUID PRIMARY KEY
        REFERENCES partition_update_requests (partition_id),
    content_hash TEXT NOT NULL,
    snapshot_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Written only after the Spark job has successfully committed the Iceberg
-- transaction. The logical timestamp is feature_partition_update_requests.timestamp.
CREATE TABLE feature_partitions (
    partition_id UUID PRIMARY KEY
        REFERENCES feature_partition_update_requests (partition_id),
    content_hash TEXT NOT NULL,
    snapshot_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Actual lineage: exact dataset-partition revisions used for a feature partition.
CREATE TABLE feature_partition_dataset_dependencies (
    feature_partition_id UUID NOT NULL REFERENCES feature_partitions (partition_id),
    dataset_partition_id UUID NOT NULL REFERENCES dataset_partitions (partition_id),
    PRIMARY KEY (feature_partition_id, dataset_partition_id)
);

CREATE INDEX IF NOT EXISTS feature_partition_dependencies_parent_index
    ON feature_partition_dataset_dependencies (dataset_partition_id);
