-- Features and labels expose their ordered entity keys through one shared name.
ALTER TABLE features RENAME COLUMN feature_keys TO keys;
ALTER TABLE features RENAME CONSTRAINT features_feature_keys_array TO features_keys_array;

ALTER TABLE labels RENAME COLUMN feature_keys TO keys;
ALTER TABLE labels RENAME CONSTRAINT labels_feature_keys_array TO labels_keys_array;
