ALTER TABLE job_queue DROP CONSTRAINT job_queue_check;
ALTER TABLE job_queue ADD CONSTRAINT job_queue_check
    CHECK (completed_at IS NULL OR claimed_at IS NOT NULL);
