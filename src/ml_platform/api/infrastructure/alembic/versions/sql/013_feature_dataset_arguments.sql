-- Each feature dependency names the transform constructor argument it supplies.
ALTER TABLE feature_dataset_dependencies ADD COLUMN argument_name TEXT;
UPDATE feature_dataset_dependencies SET argument_name = 'market_data';
ALTER TABLE feature_dataset_dependencies ALTER COLUMN argument_name SET NOT NULL;
ALTER TABLE feature_dataset_dependencies
    ADD CONSTRAINT feature_dataset_argument_name_format
    CHECK (argument_name ~ '^[a-z][a-z0-9_]*$');
ALTER TABLE feature_dataset_dependencies
    ADD CONSTRAINT feature_dataset_argument_name_unique
    UNIQUE (feature_name, feature_version, argument_name);
