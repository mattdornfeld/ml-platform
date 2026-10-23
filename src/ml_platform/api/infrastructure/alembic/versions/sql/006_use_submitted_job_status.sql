ALTER TABLE jobs ALTER COLUMN status SET DEFAULT 'submitted';
-- Older deployments may still have these legacy values. Cast to text so a
-- fresh schema whose enum never contained them remains migratable.
UPDATE jobs SET status = 'submitted' WHERE status::text IN ('queued', 'running');
