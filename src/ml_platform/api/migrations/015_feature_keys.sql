-- Feature identity may use multiple ordered entity-key columns.
ALTER TABLE features ADD COLUMN feature_keys JSONB;
UPDATE features SET feature_keys = jsonb_build_array(feature_key) WHERE feature_keys IS NULL;
ALTER TABLE features ALTER COLUMN feature_keys SET NOT NULL;
ALTER TABLE features ADD CONSTRAINT features_feature_keys_array CHECK (jsonb_typeof(feature_keys) = 'array' AND jsonb_array_length(feature_keys) > 0);
ALTER TABLE features DROP COLUMN feature_key;
