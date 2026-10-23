-- The feature-transform package was consolidated into the public data SDK.
-- Persisted resource definitions must point at the current import contract.
UPDATE features
SET transform = regexp_replace(
    transform,
    '^ml_platform\\.data_sdk\\.features\\.[^.]+\\.',
    'ml_platform.data_sdk.'
)
WHERE transform ~ '^ml_platform\\.data_sdk\\.features\\.[^.]+\\.';
