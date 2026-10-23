-- Immutable upstream revision plan for each feature partition request.
CREATE TABLE feature_partition_update_request_dataset_dependencies (
    feature_partition_request_id UUID NOT NULL REFERENCES feature_partition_update_requests (partition_id),
    dataset_partition_id UUID NOT NULL REFERENCES dataset_partitions (partition_id),
    PRIMARY KEY (feature_partition_request_id, dataset_partition_id)
);

-- A logical output partition has at most one active writer. Terminal request rows
-- remain as immutable revision history.
CREATE UNIQUE INDEX feature_partition_update_requests_active_identity_index
    ON feature_partition_update_requests (feature_name, feature_version, timestamp)
    WHERE status = 'pending';
