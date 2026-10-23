-- Dataset names are valid Python identifiers and are injected directly into transforms.
ALTER TABLE feature_dataset_dependencies DROP COLUMN argument_name;
ALTER TABLE feature_dataset_dependencies
    ADD CONSTRAINT feature_dataset_dependency_name_unique
    UNIQUE (feature_name, feature_version, dataset_name);

ALTER TABLE label_dataset_dependencies DROP COLUMN argument_name;
ALTER TABLE label_dataset_dependencies
    ADD CONSTRAINT label_dataset_dependency_name_unique
    UNIQUE (label_name, label_version, dataset_name);
