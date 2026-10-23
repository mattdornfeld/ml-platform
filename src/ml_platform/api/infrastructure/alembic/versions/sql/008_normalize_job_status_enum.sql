ALTER TABLE jobs ALTER COLUMN status DROP DEFAULT;
CREATE TYPE job_status_v2 AS ENUM ('submitted', 'succeeded', 'failed', 'cancelled');
ALTER TABLE jobs ALTER COLUMN status TYPE job_status_v2 USING status::text::job_status_v2;
DROP TYPE job_status;
ALTER TYPE job_status_v2 RENAME TO job_status;
ALTER TABLE jobs ALTER COLUMN status SET DEFAULT 'submitted';
