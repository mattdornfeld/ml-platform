-- Point-in-time feature inputs may only depend on historical source partitions.
ALTER TABLE feature_dataset_dependencies
DROP COLUMN num_lookahead_partitions;
