-- Dataset configuration is type-discriminated and carries its own dataset type.
UPDATE datasets
SET config = jsonb_set(
    config,
    '{dataset_type}',
    to_jsonb(dataset_type),
    true
);
