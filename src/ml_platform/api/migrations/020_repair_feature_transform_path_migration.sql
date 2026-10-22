-- Migration 019 used doubled regular-expression escape characters, which match
-- literal backslashes under PostgreSQL's standard-conforming SQL strings.
UPDATE features
SET transform = regexp_replace(
    transform,
    '^ml_platform\.data_sdk\.features\.[^.]+\.',
    'ml_platform.data_sdk.'
)
WHERE transform ~ '^ml_platform\.data_sdk\.features\.[^.]+\.';
