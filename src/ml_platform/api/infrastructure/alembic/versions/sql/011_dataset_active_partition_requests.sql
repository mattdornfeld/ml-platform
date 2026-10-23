-- A logical dataset partition has at most one active writer. Completed request rows
-- remain immutable revision history and may be replaced only through an explicit
-- refresh API when that is introduced.
CREATE UNIQUE INDEX partition_update_requests_active_identity_index
    ON partition_update_requests (dataset_name, dataset_version, timestamp)
    WHERE status = 'pending';
