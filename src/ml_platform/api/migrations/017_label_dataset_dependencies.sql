-- Persist label definitions' declared dataset dependencies for API job planning.
CREATE TABLE label_dataset_dependencies (
    label_name TEXT NOT NULL,
    label_version INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_version INTEGER NOT NULL,
    argument_name TEXT NOT NULL,
    num_lookback_partitions INTEGER NOT NULL CHECK (num_lookback_partitions >= 0),
    PRIMARY KEY (label_name, label_version, dataset_name, dataset_version),
    UNIQUE (label_name, label_version, argument_name),
    FOREIGN KEY (label_name, label_version) REFERENCES labels (name, version),
    FOREIGN KEY (dataset_name, dataset_version) REFERENCES datasets (name, version)
);
