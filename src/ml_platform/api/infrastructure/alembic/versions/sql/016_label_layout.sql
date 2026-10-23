-- Labels use the same canonical identity and partition metadata as features.
ALTER TABLE labels ADD COLUMN dtype TEXT NOT NULL DEFAULT 'int' CHECK (dtype IN ('double', 'float', 'int', 'bigint'));
ALTER TABLE labels ADD COLUMN frequency TEXT NOT NULL DEFAULT 'daily' CHECK (frequency IN ('hourly', 'daily'));
ALTER TABLE labels ADD COLUMN feature_keys JSONB NOT NULL DEFAULT '["symbol"]'::jsonb;
ALTER TABLE labels ADD CONSTRAINT labels_feature_keys_array CHECK (jsonb_typeof(feature_keys) = 'array' AND jsonb_array_length(feature_keys) > 0);
