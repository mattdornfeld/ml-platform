-- Authoritative queue for materialization jobs claimed by API scheduler instances.
CREATE TABLE job_queue (
    job_id UUID PRIMARY KEY REFERENCES jobs (id),
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    worker_id TEXT,
    CHECK (completed_at IS NULL OR claimed_at IS NOT NULL)
);
