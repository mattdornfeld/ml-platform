-- Validation overrides are an ordered, discriminated list.  Earlier resource
-- versions stored a legacy object shape that the strict materializer contract
-- correctly rejects.  Those objects only duplicated built-in validations, so
-- replace them with the canonical empty override list.
UPDATE features
SET validations = '[]'::jsonb
WHERE jsonb_typeof(validations) = 'object';

ALTER TABLE features
ALTER COLUMN validations SET DEFAULT '[]'::jsonb;

ALTER TABLE labels
ADD COLUMN validations JSONB NOT NULL DEFAULT '[]'::jsonb;
