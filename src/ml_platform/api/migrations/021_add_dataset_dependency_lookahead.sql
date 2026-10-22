-- Triple-barrier labels need a bounded future window as well as historical data.
ALTER TABLE feature_dataset_dependencies
ADD COLUMN num_lookahead_partitions INTEGER NOT NULL DEFAULT 0
CHECK (num_lookahead_partitions >= 0);

ALTER TABLE label_dataset_dependencies
ADD COLUMN num_lookahead_partitions INTEGER NOT NULL DEFAULT 0
CHECK (num_lookahead_partitions >= 0);
